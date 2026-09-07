//! OpenAI-compatible HTTP server for Imparo.
//!
//! Deliberately small: one model, one worker, streaming SSE. Requests serialise, which is
//! honest about the current state -- continuous batching and the KV pool are separate work
//! and this server is the thing that will exercise them.

#![forbid(unsafe_code)]

// The chat codec comes from the PLAN, not from an architecture named here. It has to be
// reachable before the model is built: the template sniff below runs at load.
use imparo_model::chat::{Channel, ChatCodec};
mod host_fit;
mod http;
mod template;

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use imparo_gguf::weights::Weights;
use imparo_model::build_plan;
// The kv_* pool methods are defaults on this trait, not inherent methods.
use imparo_tokenize::Tokenizer;
use serde_json::{Value, json};

struct Engine {
    /// The loaded workflow, whatever architecture the file is.
    ///
    /// Was `Gemma4`. Everything the server and the pool call on it is a `Model` or
    /// `KvPoolMember` method -- checked across both files before widening, not assumed --
    /// so naming the concrete type bought nothing and cost every other model.
    model: Box<dyn imparo_model::Model + Send>,
    tok: Tokenizer,
    /// This architecture's chat format: how a prompt is rendered without a template, and
    /// how the output is split back into reasoning, text and tool calls.
    chat: &'static dyn ChatCodec,
    /// The BOS token's spelling; see `render_chat_prompt` for who owns emitting it.
    bos_text: String,
    /// `chat.user_turn_open()` tokenized once: what the branch-point scan matches.
    /// Empty when the spelling does not survive tokenization, which falls the scan
    /// back to the re-render.
    user_open: Vec<u32>,
    /// Whether the scan was checked against a re-render at load and agreed. False
    /// means this template needs the slow path; see the probe at the call site.
    scan_trusted: bool,
    ctx: usize,
    /// The disk tier (docs/unified-kv-pool.md): durability across restart,
    /// conversation switch-in/out, resource fit. None when IMPARO_KV_DISK=0.
    store: Option<imparo_kv::Store>,
    /// The disk tier's writer. Every write goes through it, in order, off the
    /// request thread; reads still go straight to `store`.
    disk: Option<imparo_kv::disk::DiskQueue>,
    root: imparo_kv::ConfigRoot,
    /// Pool residency mode (default on with a GPU): multi-conversation one-copy
    /// KV. IMPARO_KV_POOL=0 reverts to the single-resident legacy path.
    pool: Option<imparo_kv::pool::PoolMode>,
    disk_cap: u64,
    /// Label the resident KV belongs to (the client's conversation id, or a
    /// content-derived name for keyless traffic). Deletion must be able to drop
    /// the resident record, so the label rides with it.
    resident_conv: String,
    /// Token ids whose KV is resident from the previous request — the pure-append
    /// continuation cache (docs/unified-kv-pool.md). Emptied whenever the cache is
    /// overwritten or a forward fails mid-flight.
    resident: Vec<u32>,
}

#[derive(Debug, Eq, PartialEq)]
enum PoolEligibility {
    DisabledByConfig,
    NoPooledLayers,
    NoBackend,
    BackendNotSelected,
    UnsafeCaps(Vec<imparo_backend::PoolCapsIssue>),
    UnsafeCutGrid {
        finest_cut_tokens: u32,
        grid_tokens: usize,
    },
    Ready(imparo_backend::PoolAddressing),
}

/// Pure server-side admission decision for the common KV pool.
///
/// Environment parsing and model/backend discovery stay at the caller. Keeping the
/// priority here explicit guarantees that `IMPARO_KV_POOL=0` cannot be overridden by a
/// capable backend, and lets the no-backend path remain a quiet no-op.
fn pool_eligibility(
    pool_requested: bool,
    has_pooled_layers: bool,
    backend_selected: bool,
    caps: Option<imparo_backend::PoolCaps>,
) -> PoolEligibility {
    if !pool_requested {
        return PoolEligibility::DisabledByConfig;
    }
    if !has_pooled_layers {
        return PoolEligibility::NoPooledLayers;
    }
    let Some(caps) = caps else {
        return PoolEligibility::NoBackend;
    };
    // v2 derives the placement page/grid from the backend declaration. Keep the
    // independent numerical cut invariant: the page may be coarser than the kernel's
    // finest safe cut, but it must be an exact multiple of it.
    let grid_tokens = imparo_kv::grid_tokens();
    if caps.finest_cut_tokens == 0 || grid_tokens % caps.finest_cut_tokens as usize != 0
    {
        return PoolEligibility::UnsafeCutGrid {
            finest_cut_tokens: caps.finest_cut_tokens,
            grid_tokens,
        };
    }
    let addressing = match caps.validate_for_pool(caps.page_cells) {
        Ok(addressing) => addressing,
        Err(issues) => return PoolEligibility::UnsafeCaps(issues),
    };
    if !backend_selected {
        return PoolEligibility::BackendNotSelected;
    }
    PoolEligibility::Ready(addressing)
}

/// The current explicit mover demotes an outgoing conversation before it promotes
/// the incoming one. At the worst boundary both can occupy a complete device arena:
/// one in Host and one on Device. Requiring room for both prevents a capacity cycle
/// where neither side can move first. This floor is derived from the allocator's
/// runtime capacity; it is not a model or SM constant.
fn host_fit_policy_for_device_blocks(
    capacity_blocks: u32,
) -> Option<host_fit::HostFitPolicy> {
    let device_units = u64::from(capacity_blocks)
        / u64::try_from(imparo_kv::resident::UNIT_BLOCKS).ok()?;
    if device_units == 0 {
        return None;
    }
    let policy = host_fit::HostFitPolicy {
        minimum_units: device_units.checked_mul(2)?,
        ..host_fit::HostFitPolicy::default()
    };
    Some(policy)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path = None;
    let mut port = 8420_u16;
    let mut context_length = 4096_usize;
    let mut cache_key_type: Option<String> = None;
    let mut cache_value_type: Option<String> = None;
    let mut template_file: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next().map(PathBuf::from),
            "--port" => port = args.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "-c" | "--ctx" => {
                context_length = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(context_length);
            }
            "--parallel" | "-md" | "--mmproj" => {
                let _ = args.next();
            }
            // KV cache types, like llama.cpp's -ctk/-ctv: f16 (default) | q4_0 | q8_0.
            // USER CONFIG, not a tuned knob.
            "-ctk" | "--cache-type-k" => cache_key_type = args.next(),
            "-ctv" | "--cache-type-v" => cache_value_type = args.next(),
            // A user-supplied jinja template file overriding the GGUF-embedded one
            // (llama.cpp's --chat-template-file).
            "--chat-template-file" => template_file = args.next().map(PathBuf::from),
            _ => {}
        }
    }
    let path = model_path.ok_or(
        "usage: imparo-server -m MODEL.gguf [--port N] [-c N] \
                                 [--cache-type-k T] [--cache-type-v T]",
    )?;
    // Only configure when a flag was given: with no flags the engine falls back to the
    // IMPARO_CTK/IMPARO_CTV environment (probe binaries and harnesses use that), and
    // pinning f16 here would silently split the host sizing from the Metal path.
    if cache_key_type.is_some() || cache_value_type.is_some() {
        let k = cache_key_type.as_deref().unwrap_or("f16");
        let v = cache_value_type.as_deref().unwrap_or("f16");
        if !imparo_model::host::configure_kv_types(k, v) {
            return Err(
                format!("bad cache type: k={k} v={v} (f16 | q4_0 | q8_0)").into()
            );
        }
        eprintln!("[imparo] kv cache types: k={k} v={v}");
    }

    let t0 = Instant::now();
    imparo_model::host::log_footprint("start");
    let document = imparo_gguf::read(&path)?;
    imparo_model::host::log_footprint("gguf read");
    let plan = build_plan(&document, &path)?;
    // From the PLAN, and before the model exists: the template sniff below runs at load.
    let chat = imparo_model::chat::codec(&plan)?;
    imparo_model::host::log_footprint("plan");
    let mut weights = Weights::open_with(&document, &path)?;
    imparo_model::backend::enable_gpu(
        &mut weights,
        &plan,
        context_length,
        imparo_model::prefill_batch(),
    )?;
    // Chat template (task #15). Source order: --chat-template-file, then the
    // GGUF-embedded tokenizer.chat_template, then this architecture's own codec.
    // A user FILE that fails to compile is fatal (they asked for it); an embedded
    // template that fails to compile falls back with a loud line.
    // Sniff any loaded template for the OUTPUT parser's markers: a template for a
    // different model renders fine while channel/tool-call parsing silently matches
    // nothing, so say so at load instead.
    let sniff = |src: &str, origin: &str| {
        let missing = chat.markers_missing_from(src);
        if !missing.is_empty() {
            eprintln!(
                "[imparo] {origin} chat template never emits {missing:?}; the \
                       output parser expects them -- reasoning/tool-call extraction \
                       will find nothing if the template and model disagree"
            );
        }
    };
    let template = if let Some(tf) = &template_file {
        let src = std::fs::read_to_string(tf)
            .map_err(|e| format!("--chat-template-file {}: {e}", tf.display()))?;
        sniff(&src, "--chat-template-file");
        Some(
            template::Template::compile(&src)
                .map_err(|e| format!("--chat-template-file {}: {e}", tf.display()))?,
        )
    } else if let Some(src) = document.string_value("tokenizer.chat_template") {
        match template::Template::compile(src) {
            Ok(t) => {
                sniff(src, "embedded");
                if imparo_model::log_on() {
                    eprintln!(
                        "[imparo] chat: embedded template in use; tool-call arguments \
                         rendered as {}",
                        if t.requires_object_arguments() {
                            "objects"
                        } else {
                            "the strings given"
                        }
                    );
                }
                Some(t)
            }
            Err(e) => {
                eprintln!(
                    "[imparo] embedded chat template unusable ({e}); falling back \
                           to this architecture's built-in renderer"
                );
                None
            }
        }
    } else {
        None
    };
    let _ = TEMPLATE.set(template);
    // The parsed document holds every vocab string and all metadata -- 13 MiB that nothing
    // reads again, since the tokenizer parses the file itself. Left in scope it lives for
    // the process's whole life.
    drop(document);
    imparo_model::host::log_footprint("weights mmap");
    let tok = Tokenizer::from_gguf(&path)?;
    // The BOS SPELLING, not its id. Templates and fallback renderers emit text, and a
    // codec that knew an id would be carrying vocabulary state it has no business having.
    let bos_text = tok
        .bos
        .map(|b| tok.token_str(b as usize).to_string())
        .unwrap_or_default();
    imparo_model::host::log_footprint("tokenizer");
    // The model file declares its trained context; that is the operational
    // limit for THIS model. The engine itself is length-clean far beyond it.
    let trained = plan.config.context_length as usize;
    if context_length > trained {
        eprintln!(
            "[imparo] -c {context_length} exceeds the model's trained context {trained}; clamping"
        );
        context_length = trained;
    }
    let mut model = imparo_model::load(weights, plan, context_length)?;
    // Device setup and any selected persistent transformed-weight cache are model
    // admission work, not a tax on the first user request. CPU execution and backends
    // without such a cache retain their established path.
    if imparo_model::backend::active().is_some() {
        model.ensure_gpu_ready()?;
    }
    // Startup allocates and frees a lot -- the GGUF document, the merge list, the token
    // strings the blob replaced. Hand it back before the server starts serving.
    imparo_model::host::release_free_memory();
    // Startup left the GGUF metadata and vocab build on malloc's free lists; give them back
    // before the first request rather than carrying them for the process's life.
    imparo_model::host::release_free_memory();
    imparo_model::host::log_footprint("model prepare");
    eprintln!(
        "[imparo] loaded ms={:.0} ctx={context_length} vocab={}",
        t0.elapsed().as_secs_f64() * 1e3,
        tok.vocab_size()
    );
    eprintln!(
        "[imparo] tokenizer approx: tokens={} merges_ranked={}                token_bytes={:.1} MiB",
        tok.vocab_size(),
        tok.rank_count(),
        tok.token_bytes() as f64 / (1 << 20) as f64
    );

    // The disk tier: on by default (durability is the point); IMPARO_KV_DISK=0
    // turns it off, IMPARO_KV_DIR moves it, IMPARO_KV_DISK_CAP_GB caps it (LRU).
    let digest = imparo_model::kv::model_digest(&path).map_err(|e| e.clone())?;
    let root = imparo_model::kv::config_root(
        model.plan(),
        &digest,
        &imparo_model::backend::active()
            .map_or_else(|| "none".to_string(), imparo_backend::Backend::device_tag),
    );
    let disk_base = if std::env::var("IMPARO_KV_DISK").is_ok_and(|v| v == "0") {
        None
    } else {
        Some(std::env::var("IMPARO_KV_DIR").map_or_else(
            |_| {
                std::env::var("HOME")
                    .map_or_else(|_| PathBuf::from("."), PathBuf::from)
                    .join(".imparo")
                    .join("kv")
            },
            PathBuf::from,
        ))
    };
    let store =
        disk_base
            .as_ref()
            .and_then(|dir| match imparo_kv::Store::open(dir, &root) {
                Ok(st) => {
                    eprintln!("[imparo] kv disk tier at {}", dir.display());
                    Some(st)
                }
                Err(e) => {
                    eprintln!("[imparo] kv disk tier DISABLED ({e})");
                    None
                }
            });
    let disk_cap = std::env::var("IMPARO_KV_DISK_CAP_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(16)
        .saturating_mul(1 << 30);
    // IMPARO_KV_PREGROW=1 (debug): grow KV to full capacity even in legacy mode,
    // isolating buffer size from the pool machinery in A/B runs.
    if std::env::var("IMPARO_KV_PREGROW").is_ok_and(|v| v == "1") {
        if let Err(e) = model.kv_prepare_pool() {
            eprintln!("[imparo] kv pregrow failed: {e}");
        }
    }
    // PoolCaps is a safety contract, not backend documentation. In addition to paged
    // reads and the byte-identity grid, validate the address-space/tier topology. This
    // keeps today's discrete CUDA backend on the correct single-resident path until it
    // has both paged attention and an implemented Device -> Host mover. Metal's unified
    // path does not change and, importantly, never pretends to perform a host transfer.
    let caps = imparo_model::backend::active().map(imparo_backend::Backend::pool_caps);
    let eligibility = pool_eligibility(
        !std::env::var("IMPARO_KV_POOL").is_ok_and(|v| v == "0"),
        model.plan().layers.iter().any(|_| true),
        std::env::var("IMPARO_GPU").is_ok_and(|v| v != "0" && !v.is_empty()),
        caps,
    );
    let pool_mode = match eligibility {
        PoolEligibility::DisabledByConfig
        | PoolEligibility::NoPooledLayers
        | PoolEligibility::NoBackend
        | PoolEligibility::BackendNotSelected => None,
        PoolEligibility::UnsafeCaps(issues) => {
            let c = caps.expect("unsafe caps came from an active backend");
            let reasons = issues
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            eprintln!(
                "[imparo] kv pool DISABLED: unsafe backend caps ({reasons}); \
shared_address={}, tiers={:?}; running the single-resident path",
                c.shared_address, c.tiers
            );
            None
        }
        PoolEligibility::UnsafeCutGrid {
            finest_cut_tokens,
            grid_tokens,
        } => {
            eprintln!(
                "[imparo] kv pool DISABLED: attention kernels require a \
{finest_cut_tokens}-token cut, which does not divide the {grid_tokens}-token \
placement grid; running the single-resident path"
            );
            None
        }
        PoolEligibility::Ready(addressing) => match model.kv_prepare_pool() {
            Ok(()) => {
                let layers = model.kv_full_layers();
                let blocks = model.kv_capacity_blocks();
                eprintln!(
                    "[imparo] kv pool: {} full-attention layers, {blocks} blocks/layer",
                    layers.len()
                );
                let strides: std::collections::BTreeMap<u32, (usize, usize)> = model
                    .kv_state_geometry()
                    .into_iter()
                    .filter(|g| matches!(g.kind, imparo_kv::StateKind::Full))
                    .map(|g| (g.layer, (g.k_stride, g.v_stride)))
                    .collect();
                // The pool's own unit size, not a second computation of it. This
                // multiplied the same strides by UNIT_TOKENS (256) while the pool
                // allocates and charges grid units (64), so the Host tier was sized
                // in units four times the ones it would actually hold.
                let unit_bytes =
                    u64::try_from(imparo_kv::pool::unit_bytes(&strides)).ok();
                let host_capacity_bytes = match addressing {
                    imparo_backend::PoolAddressing::Shared => None,
                    imparo_backend::PoolAddressing::ExplicitHostTransfers => {
                        let user_cap = match std::env::var("IMPARO_KV_HOST_MAX_MB") {
                            Ok(value) => value
                                .parse::<u64>()
                                .ok()
                                .and_then(|mb| mb.checked_mul(1 << 20))
                                .map(Some)
                                .ok_or(()),
                            Err(std::env::VarError::NotPresent) => Ok(None),
                            Err(std::env::VarError::NotUnicode(_)) => Err(()),
                        };
                        match user_cap {
                            Err(()) => {
                                eprintln!(
                                    "[imparo] kv Host tier DISABLED: InvalidConfig(\"IMPARO_KV_HOST_MAX_MB\")"
                                );
                                None
                            }
                            Ok(user_cap) => {
                                let profile = imparo_model::backend::active()
                                    .and_then(imparo_backend::Backend::kv_host_profile);
                                let policy = host_fit_policy_for_device_blocks(blocks);
                                let decision = match (profile, unit_bytes, policy) {
                                    (Some(profile), Some(unit_bytes), Some(policy)) => {
                                        host_fit::probe_and_fit_host_tier(
                                            profile,
                                            unit_bytes,
                                            store
                                                .as_ref()
                                                .and(disk_base.as_deref()),
                                            user_cap,
                                            policy,
                                        )
                                    }
                                    (None, _, _) => host_fit::HostFitDecision::Disabled(
                                        host_fit::HostFitDisabled::UnknownOrZero(
                                            host_fit::HostFact::AvailableHostBytes,
                                        ),
                                    ),
                                    (_, None, _) | (_, _, None) => {
                                        host_fit::HostFitDecision::Disabled(
                                            host_fit::HostFitDisabled::ArithmeticOverflow(
                                                host_fit::ArithmeticSite::CapacityBytes,
                                            ),
                                        )
                                    }
                                };
                                match decision {
                                    host_fit::HostFitDecision::Enabled(fit) => {
                                        eprintln!(
                                            "[imparo] kv Host tier ENABLED: {fit:?}"
                                        );
                                        Some(fit.capacity_bytes)
                                    }
                                    host_fit::HostFitDecision::Disabled(reason) => {
                                        eprintln!(
                                            "[imparo] kv Host tier DISABLED: {reason:?}"
                                        );
                                        None
                                    }
                                }
                            }
                        }
                    }
                };
                if addressing == imparo_backend::PoolAddressing::ExplicitHostTransfers
                    && host_capacity_bytes.is_none()
                {
                    None
                } else {
                    Some(imparo_kv::pool::PoolMode::new(
                        addressing,
                        host_capacity_bytes,
                        layers,
                        blocks,
                        model.kv_checkpoint_slack(),
                        strides,
                        root,
                    ))
                }
            }
            Err(e) => {
                eprintln!("[imparo] kv pool DISABLED ({e})");
                None
            }
        },
    };
    // Tokenized ONCE. The branch-point scan needs the ids, and re-encoding a
    // dozen characters per request is pointless; more to the point, doing it here
    // is where a spelling that does not survive tokenization shows up in the log
    // instead of quietly disabling the scan on every request.
    let user_open = tok.encode(chat.user_turn_open(), false);
    if user_open.is_empty() {
        eprintln!(
            "[imparo] chat: {:?} tokenizes to nothing; branch points will be found \
by re-rendering (slower)",
            chat.user_turn_open()
        );
    }
    // Does the scan AGREE with the re-render, for this template and this tokenizer?
    // Asked once, here, with a probe conversation shaped like a real one (system, tools,
    // a finished turn, a new user message). The scan is the fast path and the re-render
    // is what it replaces, so if the two disagree the answer is not "use the fast one" --
    // it is "this template needs the slow one", and the request path is told which.
    let scan_trusted = !user_open.is_empty() && {
        let probe = |n: usize| -> Vec<u32> {
            let msgs: Vec<Value> = vec![
                json!({"role": "system", "content": "You are a probe."}),
                json!({"role": "user", "content": "first question"}),
                json!({"role": "assistant", "content": "first answer"}),
                json!({"role": "user", "content": "second question"}),
            ];
            let tools: Vec<Value> = vec![json!({
                "type": "function",
                "function": {"name": "probe", "description": "A probe tool.",
                             "parameters": {"type": "object", "properties": {}}},
            })];
            let text = render_chat_prompt(
                TEMPLATE.get().and_then(Option::as_ref),
                chat,
                &msgs[..n],
                &tools,
                n == msgs.len(),
                tok.add_bos,
                &bos_text,
                &template_kwargs(&Value::Null),
            );
            tok.encode(&text, true)
        };
        let full = probe(4);
        let head = probe(3);
        let expect = full.iter().zip(&head).take_while(|(a, b)| a == b).count();
        let scanned = last_subsequence(&full, &user_open);
        let ok = scanned == Some(expect);
        if ok {
            // Positively, not by silence: an empty log would also be what a probe
            // that never ran looks like.
            if imparo_model::log_on() {
                eprintln!(
                    "[imparo] chat: turn-opener scan agrees with the re-render at \
{expect} on the probe conversation"
                );
            }
        } else {
            eprintln!(
                "[imparo] chat: the turn-opener scan says {scanned:?} where a \
re-render says {expect}; branch points will be found by re-rendering (slower)"
            );
        }
        ok
    };
    let disk = store.clone().map(imparo_kv::disk::DiskQueue::new);
    let engine = Arc::new(Mutex::new(Engine {
        model,
        tok,
        chat,
        bos_text,
        user_open,
        scan_trusted,
        ctx: context_length,
        store,
        disk,
        root,
        pool: pool_mode,
        disk_cap,
        resident_conv: String::new(),
        resident: Vec::new(),
    }));
    // Graceful shutdown: bytes now move to disk only at switch-out, so a
    // SIGTERM/SIGINT must spill the ACTIVE conversation before the process dies
    // -- that is what keeps restart durability (kv_e2e, the accept gate) intact.
    #[cfg(unix)]
    if let Some(rx) = imparo_model::host::shutdown_pipe() {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            use std::io::Read;
            let mut rx = rx;
            let mut b = [0u8; 1];
            let _ = rx.read(&mut b);
            let mut engine = engine
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let Engine {
                model,
                store,
                disk,
                pool,
                ..
            } = &mut *engine;
            if let Some(pl) = pool.as_mut() {
                if let Some(label) = pl.active.clone() {
                    if let Err(e) =
                        pl.switch_out(&**model, store.as_ref(), disk.as_ref(), &label)
                    {
                        eprintln!("[imparo] shutdown spill: {e}");
                    }
                }
            }
            // Every turn already committed its own state, so this usually has nothing
            // to write -- but "usually" is not durability. The drain is what makes
            // process exit mean the bytes are on disk.
            if let Some(dq) = disk.as_ref() {
                if let Err(e) = dq.drain() {
                    eprintln!("[imparo] kv shutdown barrier: {e}");
                }
            }
            std::process::exit(0);
        });
    }
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    eprintln!("[imparo] listening http://127.0.0.1:{port}");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            if let Err(e) = serve_one(stream, &engine) {
                eprintln!("[imparo] connection error: {e}");
            }
        });
    }
    Ok(())
}

/// The compiled chat template, set once at startup (None = use the built-in renderer).
static TEMPLATE: std::sync::OnceLock<Option<template::Template>> =
    std::sync::OnceLock::new();

/// Renders a chat prompt, and gives the BOS prefix EXACTLY ONE OWNER.
///
/// A template sees the real vocabulary spelling, because that is what the reference
/// contract gives it. But the tokenizer may be configured to prepend BOS at encode, and
/// then the rendered prefix is a second one -- two BOS tokens, which shifts every
/// position and is invisible in the text.
///
/// ```text
/// tokenizer adds BOS   -> strip one rendered prefix; encode(.., true) puts it back
/// tokenizer does not   -> the template or the fallback owns it
/// ```
/// The label a conversation's state is stored under.
///
/// A client id when one is given. Otherwise a CONTENT-derived name: the chain tip, which
/// folds in every earlier unit AND the model configuration, so two keyless conversations
/// share a label only when their state is genuinely interchangeable.
///
/// Below one whole unit there is no tip, so they all take the same bare name -- and that
/// is safe because there is nothing to adopt either: the forward starts at 0, where
/// recurrent state is reset.
///
/// The rule was written out at three call sites with three different hash lists, which is
/// correct (begin sees the prompt, spill sees prompt+generated), and TWO different empty
/// cases -- one `"keyless"`, one `String::new()`. One of those is a label nothing can
/// find again.
/// Index of the LAST occurrence of `needle` in `hay`, or None.
///
/// This is how a turn boundary is found: the conversation's own turn opener,
/// tokenized once at load, matched against the prompt's ids. The answer is the
/// same one a re-render gives -- the token where the last user message's markup
/// begins -- for a scan over integers instead of a second full tokenization.
///
/// A prompt that contains the opener inside a message (a user quoting the chat
/// format at us) moves the checkpoint, and nothing more: a checkpoint is only a
/// position, and what may be RESUMED from it is decided by comparing tokens.
fn last_subsequence(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len())
        .rev()
        .find(|&i| hay[i..i + needle.len()] == *needle)
}

fn conversation_label(conversation: &str, hashes: &[imparo_kv::UnitHash]) -> String {
    if conversation != "default" {
        return conversation.to_string();
    }
    hashes
        .last()
        .map_or_else(|| "keyless".to_string(), |h| format!("keyless-{}", h.hex()))
}

/// The template variables a request sets (`chat_template_kwargs`, llama.cpp's field), with
/// llama-server's policy applied so both servers render the same prompt for the same
/// request: `enable_thinking` is true unless the request says `false` or asks for
/// `reasoning_effort: "none"`. The model's own template defaults `enable_thinking` to
/// false when nothing sets it, which is what the MLX servers render; llama-server and
/// this server default it to true.
fn template_kwargs(body: &Value) -> serde_json::Map<String, Value> {
    let mut kwargs = body
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let effort_none =
        body.get("reasoning_effort").and_then(Value::as_str) == Some("none");
    let thinking = match kwargs.get("enable_thinking") {
        Some(Value::Bool(b)) => *b && !effort_none,
        _ => !effort_none,
    };
    kwargs.insert("enable_thinking".to_string(), Value::Bool(thinking));
    kwargs
}

fn render_chat_prompt(
    template: Option<&template::Template>,
    chat: &dyn ChatCodec,
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
    tokenizer_adds_bos: bool,
    bos_text: &str,
    kwargs: &serde_json::Map<String, Value>,
) -> String {
    let mut prompt = match template {
        Some(t) => {
            match t.render(messages, tools, add_generation_prompt, bos_text, kwargs) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "[imparo] chat template render failed ({e}); using the built-in \
                     renderer for this request (it ignores chat_template_kwargs)"
                    );
                    chat.render(messages, tools, add_generation_prompt, bos_text)
                }
            }
        }
        None => chat.render(messages, tools, add_generation_prompt, bos_text),
    };
    if tokenizer_adds_bos && !bos_text.is_empty() && prompt.starts_with(bos_text) {
        prompt.drain(..bos_text.len());
    }
    prompt
}

fn serve_one(
    mut stream: TcpStream,
    engine: &Arc<Mutex<Engine>>,
) -> std::io::Result<()> {
    let Some(req) = http::read_request(&mut stream)? else {
        return Ok(());
    };
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => http::json(&mut stream, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") => http::json(
            &mut stream,
            200,
            &json!({
                "object": "list",
                "data": [{"id": "imparo", "object": "model", "owned_by": "imparo"}]
            }),
        ),
        ("POST", "/v1/chat/completions") => chat_completions(&mut stream, &req, engine),
        ("POST", "/kv/conversations/erase") => {
            erase_conversations(&mut stream, &req, engine)
        }
        ("POST", "/kv/barrier") => kv_disk_barrier(&mut stream, engine),
        _ => http::json(&mut stream, 404, &json!({"error": "not found"})),
    }
}

/// Wait until every KV write accepted before this request is durable.
///
/// The writer is deliberately asynchronous on the inference path.  A lifecycle
/// harness (and an orchestrator preparing to stop the process) needs an explicit
/// boundary that works on Windows too, where `TerminateProcess` cannot run Rust
/// destructors or the Unix signal-driven shutdown path.
fn kv_disk_barrier(
    stream: &mut TcpStream,
    engine: &Arc<Mutex<Engine>>,
) -> std::io::Result<()> {
    let engine = engine
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(dq) = engine.disk.as_ref() else {
        return http::json(stream, 200, &json!({"durable": true, "disk": false}));
    };
    match dq.drain() {
        Ok(()) => http::json(stream, 200, &json!({"durable": true, "disk": true})),
        Err(error) => {
            http::json(stream, 500, &json!({"durable": false, "error": error}))
        }
    }
}

/// POST /kv/conversations/erase {"ids": ["...", ...]} -- prompt deletion of a
/// conversation SET in one pass: manifests+checkpoints removed, then every unit
/// no surviving manifest references is swept (shared preamble units survive
/// through their other holders; private content goes). The resident record is
/// dropped for erased conversations so their KV cannot be silently reused.
fn erase_conversations(
    stream: &mut TcpStream,
    req: &http::Request,
    engine: &Arc<Mutex<Engine>>,
) -> std::io::Result<()> {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => return http::json(stream, 400, &json!({"error": e.to_string()})),
    };
    let ids: Vec<String> = body
        .get("ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty() {
        return http::json(
            stream,
            400,
            &json!({"error": "ids: non-empty array required"}),
        );
    }
    let mut engine = engine
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if ids.iter().any(|i| *i == engine.resident_conv) {
        engine.resident.clear();
        engine.resident_conv.clear();
    }
    if let Some(pl) = engine.pool.as_mut() {
        if let Err(error) = pl.forget(&ids) {
            eprintln!("[imparo] kv pool erase: {error}");
        }
    }
    let Some(store) = &engine.store else {
        return http::json(stream, 200, &json!({"erased": 0, "disk": false}));
    };
    let _ = store;
    let Some(dq) = engine.disk.as_ref() else {
        return http::json(stream, 200, &json!({"erased": 0, "disk": false}));
    };
    // Through the queue, so it lands behind whatever those conversations' last turns
    // were still writing: a sweep that overtook a commit would delete the units that
    // commit is about to name.
    let swept = match dq.erase_blocking(ids.clone()) {
        Ok(swept) => swept,
        Err(error) => {
            return http::json(stream, 500, &json!({"error": error}));
        }
    };
    http::json(
        stream,
        200,
        &json!({"erased": ids.len(), "units_swept": swept}),
    )
}

fn parse_raw_input_ids(
    body: &Value,
    enabled: bool,
) -> Result<Option<Vec<u32>>, String> {
    if !enabled {
        return Ok(None);
    }
    let Some(value) = body.get("input_ids") else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| "input_ids must be an array".to_string())?;
    if values.is_empty() {
        return Err("input_ids must not be empty".to_string());
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| format!("input_ids[{index}] must be a u32"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn chat_completions(
    stream: &mut TcpStream,
    req: &http::Request,
    engine: &Arc<Mutex<Engine>>,
) -> std::io::Result<()> {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => return http::json(stream, 400, &json!({"error": e.to_string()})),
    };
    let raw_ids = match parse_raw_input_ids(
        &body,
        std::env::var("IMPARO_DEV_RAW_TOKENS").is_ok_and(|value| value == "1"),
    ) {
        Ok(ids) => ids,
        Err(error) => return http::json(stream, 400, &json!({"error": error})),
    };
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // SEED: prefill a preamble -- a system prompt and its tools -- and store it, with no
    // user turn, no generation prompt and no answer. What it buys is the FIRST user turn
    // of every conversation that carries that preamble: a seed's unit chain is short by
    // construction, so it is a prefix of any such prompt and the disk tier can hand it
    // back whole. A preamble stored as an ordinary conversation cannot do that -- its
    // chain runs past the preamble into a user turn and an answer, and a later prompt
    // that shares only the preamble matches a PREFIX of that chain, which the restore
    // refuses (measured: "agrees for 4 of its 5 units, so nothing is adoptable").
    let seed = body.get("seed").and_then(Value::as_bool).unwrap_or(false);
    if seed && raw_ids.is_some() {
        return http::json(
            stream,
            400,
            &json!({"error": "seed cannot be combined with input_ids"}),
        );
    }
    let max_tokens = if seed {
        0
    } else {
        body.get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(128) as usize
    };
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let kwargs = template_kwargs(&body);
    let temperature = body
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    // Addressing only. Per docs/unified-kv-pool.md the conversation key never gates
    // correctness, so it is taken from the client without validation; a wrong one can cost
    // a cache miss, never a wrong token. Nothing reuses KV yet -- this is the prerequisite.
    let conversation = req
        .header("x-conversation-id")
        .unwrap_or("default")
        .to_string();
    // `x-new-conversation: true` -- "whatever this request borrows, it is not continuing
    // it". Only the CLIENT knows this case, and it is not the same as diverging: a fork
    // that carries a conversation's whole history and then goes its own way agrees with
    // the stored chain to the token, so no test on the prompt can tell it from a
    // continuation. It matters only for a keyless client, where the label is derived from
    // content and a continuation is allowed to drop the manifest it stood on; with a
    // conversation id the identity is explicit and nothing is ever dropped implicitly.
    let new_conversation = req
        .header("x-new-conversation")
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "True"));
    // Jinja template when one is loaded, this architecture's codec as the fallback.
    // The template sees the REAL bos spelling and `render_chat_prompt` then strips one
    // prefix when the tokenizer adds its own -- exactly one owner. It used to render
    // with bos empty, which is the same answer only while every tokenizer adds BOS.
    let (adds_bos, bos_text, chat) = {
        let e = engine
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (e.tok.add_bos, e.bos_text.clone(), e.chat)
    };
    let prompt = if raw_ids.is_some() {
        String::new()
    } else {
        render_chat_prompt(
            TEMPLATE.get().and_then(Option::as_ref),
            chat,
            &messages,
            &tools,
            true,
            adds_bos,
            &bos_text,
            &kwargs,
        )
    };
    // IMPARO_DUMP_PROMPT=1: print the rendered prompt between markers, for byte-diffing
    // against llama-server's /apply-template (task #7).
    if std::env::var("IMPARO_DUMP_PROMPT").is_ok_and(|v| v == "1") {
        eprintln!("[prompt-dump-begin]{prompt}[prompt-dump-end]");
    }
    let mut engine = engine
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let t_tok = Instant::now();
    let mut ids = raw_ids.unwrap_or_else(|| engine.tok.encode(&prompt, true));
    if seed {
        // WHERE DOES THE PREAMBLE END? Not at "render without a generation prompt":
        // gemma4's template writes `<|think|>` into the system block only when one is
        // asked for, so a seed rendered that way is not a prefix of any real prompt --
        // measured, the two part 14 characters in. Rather than teach the seed about each
        // template, ask the template: render this preamble with two DIFFERENT user
        // messages and keep the tokens they agree on. That is exactly the text the model
        // sees before the user's own words, whatever the codec spells there.
        //
        // One token of margin, because the last shared token is the one that can merge
        // with what follows it.
        let probe = |txt: &str, e: &Engine| {
            let mut m = messages.clone();
            m.push(json!({"role": "user", "content": txt}));
            let p = render_chat_prompt(
                TEMPLATE.get().and_then(Option::as_ref),
                chat,
                &m,
                &tools,
                true,
                adds_bos,
                &bos_text,
                &kwargs,
            );
            e.tok.encode(&p, true)
        };
        let (a, b) = (probe("alpha probe", &engine), probe("beta probe", &engine));
        let n = a
            .iter()
            .zip(&b)
            .take_while(|(x, y)| x == y)
            .count()
            .saturating_sub(1);
        // The seed's tokens are a PROBE's tokens, cut where the probes part -- not the
        // user-less render's. Rendering the preamble alone puts an assistant opener where
        // a real prompt puts the user's turn, so those tokens are nobody's prefix.
        ids = a[..n].to_vec();
        if imparo_model::log_on() {
            eprintln!("[imparo] seed: the preamble is {n} tokens");
        }
    }
    let ids = ids;
    if std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1") {
        eprintln!(
            "[imparo] tokenize {} chars -> {} ids in {:.2} ms",
            prompt.len(),
            ids.len(),
            t_tok.elapsed().as_secs_f64() * 1e3
        );
    }
    let prompt_tokens = ids.len();
    if prompt_tokens + max_tokens > engine.ctx {
        return http::json(
            stream,
            400,
            &json!({
            "error": format!("prompt {prompt_tokens} + max_tokens {max_tokens} exceeds ctx {}",
                             engine.ctx)}),
        );
    }

    // KV continuation: when the request strictly extends the resident tokens, resume
    // at the deepest 64-ALIGNED position instead of re-prefilling the whole history.
    // 64 is the engine's token-tile width: resuming there is MEASURED byte-identical
    // to a cold prefill (dev_harness/kv_gate.py); unaligned resumes are not, so the
    // resume point snaps down and at most 63 tokens are re-prefilled. Divergence from
    // the resident prefix means the windowed layers' rings may hold aliased positions,
    // so anything but pure append falls back to cold. IMPARO_NO_REUSE=1 is the A/B off.
    let reuse_off = std::env::var("IMPARO_NO_REUSE").is_ok_and(|v| v == "1");
    if !engine.resident.is_empty()
        && std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1")
    {
        let lcp = ids
            .iter()
            .zip(&engine.resident)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "[imparo] kv probe: resident={} ids={} lcp={lcp}",
            engine.resident.len(),
            ids.len()
        );
        if lcp < engine.resident.len() && lcp + 8 >= engine.resident.len() {
            let a = &ids[lcp..(lcp + 6).min(ids.len())];
            let b = &engine.resident[lcp..(lcp + 6).min(engine.resident.len())];
            eprintln!("[imparo] kv probe: diverge ids={a:?} resident={b:?}");
        }
    }
    // POOL MODE: residency, sharing, disk -- one entry point (docs/unified-kv-pool.md).
    let mut pool_pos: Option<usize> = None;
    type PoolBranch = (usize, Vec<imparo_kv::UnitHash>, String, bool);
    let mut pool_branch: Option<PoolBranch> = None;
    // What the state is actually stored under, for the stats line. Without it the log
    // shows the raw header -- `conv=default` for every keyless request -- while the
    // effective label is content-derived and different for each.
    let mut effective_label: Option<String> = None;
    if !reuse_off && engine.pool.is_some() {
        let prof = std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1");
        let t_hash = Instant::now();
        let hashes = imparo_kv::unit_ids(&engine.root, &ids);
        let ms_hash = t_hash.elapsed().as_secs_f64() * 1e3;
        let t_branch = Instant::now();
        // Branch point: the token where the LAST user message starts. Sub-agents
        // share everything before it (system prompt, tools, history), so the prefill
        // pauses there to checkpoint the windowed state under the tip unit's hash --
        // what lets another conversation adopt the shared prefix.
        //
        // Found by SCANNING the ids the prompt already produced for this codec's turn
        // opener. The re-render below is the same answer computed the expensive way --
        // render the messages minus the last user one, tokenize, count the common
        // prefix -- and it runs only when the two were checked at load and DISAGREED
        // (`scan_trusted`), or when the opener does not survive tokenization. The
        // reference matches template delimiters in the stream the same way.
        //
        // NOT rounded to the 256-token unit grid. Units are what conversations
        // SHARE; a checkpoint is where one can resume, and the two are different
        // questions -- rounding cost up to 255 reprocessed tokens per restore.
        let branch_off = std::env::var("IMPARO_KV_BRANCH").is_ok_and(|v| v == "0");
        // The fast path: the last place this conversation's own turn opener appears
        // in the ids the prompt already produced. One pass over integers.
        let scanned = if branch_off {
            None
        } else {
            last_subsequence(&ids, &engine.user_open)
        };
        let re_render = !branch_off && !engine.scan_trusted;
        let branch_pos = scanned.unwrap_or_else(|| {
            messages
                .iter()
                .take(if re_render { usize::MAX } else { 0 })
                .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                .filter(|&i| i > 0)
                .map_or(0, |i| {
                    let head = render_chat_prompt(
                        TEMPLATE.get().and_then(Option::as_ref),
                        chat,
                        &messages[..i],
                        &tools,
                        false,
                        adds_bos,
                        &bos_text,
                        &kwargs,
                    );
                    let head_ids = engine.tok.encode(&head, true);
                    ids.iter()
                        .zip(&head_ids)
                        .take_while(|(a, b)| a == b)
                        .count()
                })
        });
        // Where the two conversations diverge, moved to a point a resume can reproduce
        // (imparo_model::kv::resume_point states that rule and why).
        //
        // This used to snap to the 256-token UNIT at q4_0/q8_0 instead, because two
        // conversations with the IDENTICAL prompt answered differently once the branch
        // was 704 rather than 512. That was a width-keyed precision path in the engine,
        // not the pool; it is fixed (HALF_A_MIN, imparo_metal.mm) and every cache type
        // branches on the 64 grid now.
        //
        // A checkpoint is named by the last WHOLE unit below it, so a branch inside the
        // first unit has nothing to hang on: hence the >= 256 floor.
        let branch_pos = imparo_model::kv::resume_point(branch_pos, ids.len());
        let branch_pos = if branch_pos >= 256 { branch_pos } else { 0 };
        let ms_branch = t_branch.elapsed().as_secs_f64() * 1e3;
        let t_label = Instant::now();
        let upper = ids.len() + max_tokens + 1;
        // A keyless client is identified by what its prompt CONTINUES: if a resident
        // conversation's tokens are a prefix of this prompt, this is that conversation
        // and it keeps its name. Only when nothing matches does it get a fresh
        // content-derived one.
        // `x-new-conversation` skips the question entirely: the client says this prompt
        // starts something, so it gets its own identity even though it may agree with a
        // resident conversation to the last token.
        let continued = engine
            .pool
            .as_ref()
            .filter(|_| conversation == "default" && !new_conversation)
            .and_then(|pl| pl.label_for_prefix(&ids));
        if imparo_model::log_on() && conversation == "default" {
            eprintln!(
                "[imparo] kv label: continued={:?} content={}",
                continued.as_deref(),
                conversation_label(&conversation, &hashes)
            );
        }
        let label =
            continued.unwrap_or_else(|| conversation_label(&conversation, &hashes));
        let ms_label = t_label.elapsed().as_secs_f64() * 1e3;
        effective_label = Some(label.clone());
        let t_begin = Instant::now();
        let Engine {
            model,
            store,
            disk,
            pool,
            ..
        } = &mut *engine;
        if let Some(pl) = pool.as_mut() {
            match pl.begin(
                &mut **model,
                store.as_ref(),
                disk.as_ref(),
                &label,
                &ids,
                upper,
                conversation == "default",
                new_conversation,
            ) {
                // A checkpoint boundary is wherever a turn ended, which is not a grid
                // point; resuming below it recomputes a few tokens and is exact.
                Ok(r) => pool_pos = Some(imparo_model::kv::resume_point(r, ids.len())),
                Err(e) => {
                    eprintln!("[imparo] kv pool begin failed ({e}); cold path");
                    pool_pos = Some(0);
                }
            }
        }
        // The host-side cost of deciding what to reuse, which is paid on EVERY
        // request before a single kernel runs. `branch` is the expensive one by
        // construction: it re-renders the conversation minus its last user message
        // and tokenizes it a second time.
        if prof {
            eprintln!(
                "[imparo] kv decide: hash={ms_hash:.2} branch={ms_branch:.2} label={ms_label:.2} begin={:.2} ms for {} tokens",
                t_begin.elapsed().as_secs_f64() * 1e3,
                ids.len()
            );
        }
        if branch_pos > 0 {
            // Capture eagerly ONLY when the boundary could slide out of ring
            // reach before a switch-out (long replies); otherwise it is recorded
            // after the forward and captured lazily -- zero copies on the hot path.
            // EAGER when the device cannot still produce a valid checkpoint for the
            // branch later. A recurrent model reports zero slack, so it is always
            // eager: its conv state is overwritten by the next token.
            let eager = upper.saturating_sub(branch_pos) > model.kv_checkpoint_slack();
            pool_branch = Some((branch_pos, hashes, label, eager));
        }
    }

    let start_pos: usize = if pool_pos.is_some() {
        0 // decided below; legacy path skipped
    } else if !reuse_off
        && !engine.resident.is_empty()
        && ids.len() >= engine.resident.len()
        && ids[..engine.resident.len()] == engine.resident[..]
    {
        imparo_model::kv::resume_point(engine.resident.len(), ids.len())
    } else {
        0
    };

    // DISK RESTORE LIVES IN THE POOL PATH, and only there.
    //
    // A second restore used to sit here for the single-resident path. It could not
    // work: it read the checkpoint through `Store::checkpoint_for`, i.e. a file beside
    // the manifest, and nothing has ever written one -- the only writer is
    // `put_checkpoint`, which is content-addressed at `ckpt/<hash>` and reached through
    // `Manifest::ckpts[].blob`. So it returned None on every request and the branch was
    // unreachable; the `restored` flag it set was discarded a few lines later.
    //
    // Rebuilding it is not the fix either. What the pool stores per boundary is a chain
    // LINK (`delta_blob`), and turning links back into a state is `assemble_chain` in
    // the pool -- a second copy here would be the same code with its own bugs. With the
    // pool off there is no disk restore, and the pool is on wherever the backend
    // declares `paged_reads`.
    let start_pos = pool_pos.unwrap_or(start_pos);

    let id = format!("chatcmpl-imparo-{prompt_tokens}");
    if streaming {
        http::sse_headers(stream)?;
        http::sse(
            stream,
            &json!({
            "id": id, "object": "chat.completion.chunk", "model": "imparo",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": null},
                         "finish_reason": null}]}),
        )?;
    }

    let t_prefill = Instant::now();
    // ONE logits buffer for the whole request. Allocating a fresh vocab-sized `Vec` per
    // step costs 1 MiB of fresh anonymous pages every decode step -- 256 page faults per
    // token at this vocab, plus the heap high-water they leave behind.
    let mut logits: Vec<f32> = Vec::new();
    // A cold prefill overwrites the cache; the resident record is stale from this
    // point until the request completes, and a mid-flight failure leaves it empty.
    engine.resident.clear();
    let mut prefill_from = start_pos;
    if let Some((branch, bhashes, blabel, true)) = &pool_branch {
        let branch = *branch;
        if imparo_model::log_on() {
            eprintln!(
                "[imparo] kv branch eager at {branch}: prefill_from={prefill_from} len={}",
                ids.len()
            );
        }
        // `>=`, not `>`. A conversation continuing ITSELF resumes exactly at its new
        // user message -- both are the same 64-grid point -- so `>` meant a model that
        // is always eager (recurrent: zero slack) recorded a branch on its FIRST turn
        // and never again. There is nothing to forward in that case, but there is
        // something to checkpoint: the cache is at `filled == prefill_from == branch`
        // right now, which is exactly the state the branch names. Below it, the
        // recurrent half would already have advanced past the boundary, so that case
        // still records nothing.
        if branch >= prefill_from && branch < ids.len() {
            // Pause the prefill at the branch point -- where the last user message
            // starts -- and checkpoint the windowed state: this is what lets a
            // sub-agent adopt the shared prefix WITH valid windows.
            if branch > prefill_from {
                if let Err(e) = engine.model.forward_into(
                    &ids[prefill_from..branch],
                    prefill_from,
                    &mut logits,
                ) {
                    return http::json(stream, 500, &json!({"error": e}));
                }
            }
            // A RESIDENT record: its tokens are compared by `matching_ckpts` against
            // the stretch above the last 256-TILE, so that is the grid they are cut
            // on. The manifest's tail is a different measurement -- above the last
            // EXTENT -- and `commit_to_disk` derives it there rather than inheriting
            // this one.
            let units = branch / imparo_kv::grid_tokens();
            if let Some(tip) = bhashes.get(units.wrapping_sub(1)) {
                let tip = *tip;
                let tokens = ids[units * imparo_kv::grid_tokens()..branch].to_vec();
                let Engine { model, pool, .. } = &mut *engine;
                if let Some(pl) = pool.as_mut() {
                    let prev = pl.prev_ckpt_at(blabel, bhashes, &ids, branch - 1);
                    pl.note_ckpt(&**model, blabel, tip, prev, branch, tokens);
                }
            }
            prefill_from = branch;
        }
    }
    if let Err(e) =
        engine
            .model
            .forward_into(&ids[prefill_from..], prefill_from, &mut logits)
    {
        return http::json(stream, 500, &json!({"error": e}));
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    // IMPARO_KV_DIGEST=1: the cache as the prefill left it, BEFORE a single token is
    // generated -- so two conversations with the same prompt can be compared over the
    // whole prompt, not just the part they share.
    {
        let Engine { model, pool, .. } = &mut *engine;
        if let (Some(pl), Some(l)) = (pool.as_ref(), effective_label.as_deref()) {
            pl.digest(&**model, l, prompt_tokens);
        }
    }
    // Read the submission profile at the prefill boundary so prefill and decode are
    // attributed separately -- they have completely different shapes and averaging them
    // would hide whichever one is stalling.
    imparo_model::host::prof_log("prefill", prefill_ms);

    let end_sequence_token = engine.tok.eos;
    let end_turn_token = engine.tok.eot;
    let mut generated: Vec<u32> = Vec::new();
    let mut emitted = String::new();
    // The generated text's bytes, appended per token: decoding the WHOLE prefix each step
    // was quadratic in generation length (review #116, D7).
    let mut gen_bytes: Vec<u8> = Vec::new();
    let t_decode = Instant::now();
    // Counts forwards, not tokens. The first token is free -- it comes from the prefill
    // logits -- so N tokens cost N-1 decode steps, and llama.cpp reports its decode rate
    // over exactly that (n_gen - 1). Dividing N tokens by N-1 forwards would report a rate
    // 1/(N-1) too high against it.
    let mut decode_steps = 0_usize;
    // Between two forwards the GPU is idle. `gpu_busy` sits ~0.7 ms/token under wall, so
    // whatever runs here is on the critical path just as much as a kernel is.
    let (mut t_sample, mut t_detok, mut t_send) = (0.0_f64, 0.0_f64, 0.0_f64);
    // Read once, outside the loop: the clock reads themselves are the probe's cost, so
    // gating only the print would leave six of them per token in the measured build.
    let probe = std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1");
    // The FIRST pick is made on the host from the prefill logits; every later one comes
    // back from `forward_next`, which runs the same greedy pick on the GPU and returns
    // only the index -- the vocab-size logits never cross to the host during decode.
    let t_s = probe.then(Instant::now);
    let (mut sent_reasoning, mut sent_visible) = (0usize, 0usize);
    let mut next = sample(&logits, temperature);
    if let Some(t) = t_s {
        t_sample += t.elapsed().as_secs_f64() * 1e3;
    }
    // PIPELINED DECODE (docs/decode-turnaround.md): the step that consumes `next` is
    // queued before `next` is emitted, and the step after it is queued before this one's
    // pick is read, so the GPU never waits for the host between tokens. The host learns
    // each token one step late; on a stop the one step still queued is discarded.
    let pipelined = engine.model.decode_pipelined() && max_tokens > 1;
    let mut queued = false;
    if pipelined {
        match engine.model.queue_step(Some(next), prompt_tokens) {
            Ok(()) => queued = true,
            Err(e) => return http::json(stream, 500, &json!({"error": e})),
        }
    }
    for step in 0..max_tokens {
        if Some(next) == end_sequence_token || Some(next) == end_turn_token {
            break;
        }
        generated.push(next);
        let t_d = probe.then(Instant::now);
        engine
            .tok
            .decode_bytes_into(&generated[generated.len() - 1..], &mut gen_bytes);
        let piece = String::from_utf8_lossy(&gen_bytes).into_owned();
        if let Some(t) = t_d {
            t_detok += t.elapsed().as_secs_f64() * 1e3;
        }
        let t_e = probe.then(Instant::now);
        if piece.len() > emitted.len() {
            emitted = piece;
            if streaming {
                // Split on every prefix so reasoning never streams as `content`; the
                // holdback keeps a marker that straddles two deltas from being emitted
                // as visible text and then silently reclassified.
                let (r, v) = chat.split_channels(&emitted);
                let r_safe = r.len() - chat.holdback(&r, Channel::Reasoning);
                let v_safe = v.len() - chat.holdback(&v, Channel::Visible);
                if r_safe > sent_reasoning {
                    http::sse(
                        stream,
                        &json!({
                        "id": id, "object": "chat.completion.chunk", "model": "imparo",
                        "choices": [{"index": 0,
                                     "delta": {"reasoning_content": &r[sent_reasoning..r_safe]},
                                     "finish_reason": null}]}),
                    )?;
                    sent_reasoning = r_safe;
                }
                if v_safe > sent_visible {
                    http::sse(
                        stream,
                        &json!({
                        "id": id, "object": "chat.completion.chunk", "model": "imparo",
                        "choices": [{"index": 0, "delta": {"content": &v[sent_visible..v_safe]},
                                     "finish_reason": null}]}),
                    )?;
                    sent_visible = v_safe;
                }
            }
        }
        if let Some(t) = t_e {
            t_send += t.elapsed().as_secs_f64() * 1e3;
        }
        // The last token needs no forward: its logits would never be read. Computing them
        // anyway spent a full decode step per request.
        if generated.len() >= max_tokens {
            break;
        }
        let pos = prompt_tokens + step;
        if pipelined {
            // The step at `pos` is already queued. Its pick is token step+1; the step
            // after it produces token step+2, wanted only if that token can be emitted.
            let want_more = step + 2 < max_tokens;
            if want_more && engine.model.queue_step(None, pos + 1).is_err() {
                break;
            }
            match engine.model.wait_step() {
                Ok(id) => next = id,
                Err(_) => {
                    queued = want_more;
                    break;
                }
            }
            queued = want_more;
        } else {
            match engine.model.forward_next(next, pos) {
                Ok(id) => next = id,
                Err(_) => break,
            }
        }
        decode_steps += 1;
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    // The cache now holds the prompt plus every token that went through a forward:
    // decode_steps of the generated tokens (the last generated token is never
    // forwarded — its logits would be unread — so it is not resident).
    engine.resident = ids;
    engine
        .resident
        .extend_from_slice(&generated[..decode_steps.min(generated.len())]);
    engine.resident_conv.clone_from(&conversation);
    let (reasoning, body) = chat.split_channels(&emitted);
    let (visible, calls) = chat.parse_tool_calls(&body);
    let tool_calls: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (n, a))| {
            json!({
        "id": format!("call_{n}_{i}"), "type": "function",
        "function": {"name": n, "arguments": a}})
        })
        .collect();
    // The assistant message a client will send back next turn, which is what the next
    // prompt renders -- so it is also what this turn has to have RECORDED.
    let mut message = json!({"role": "assistant", "content": &visible});
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning.clone());
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls.clone());
    }

    let completion_tokens = generated.len();
    let finish = if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    // Report the same `timings` block llama.cpp does. Without it a harness comparing the
    // two engines has to split prefill from decode by wall clock on our side and by the
    // server's own numbers on theirs -- two different rulers, which charged us for decode
    // time the fork was not charged for and understated our prefill by a third.
    let processed = prompt_tokens - start_pos;
    let timings = json!({
        "prompt_n": processed,
        "prompt_ms": prefill_ms,
        "prompt_per_second": processed as f64 / (prefill_ms / 1e3).max(1e-9),
        "predicted_n": completion_tokens,
        "predicted_ms": decode_ms,
        "predicted_per_second": decode_steps as f64 / (decode_ms / 1e3).max(1e-9),
    });
    // cached_tokens = the reused prefix, same field llama-server reports. What lets
    // a speed harness (dev_harness/bracket.py) reject cache-contaminated legs on
    // THIS engine instead of only on the reference.
    let usage = json!({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                       "total_tokens": prompt_tokens + completion_tokens,
                       "prompt_tokens_details": {"cached_tokens": start_pos}});

    // THE RESPONSE GOES OUT FIRST. Everything below keeps the engine for this request
    // but no longer feeds the answer: the pipelined step queued behind the stop is
    // discarded and the turn's rows are written through. Measured on the response path
    // before this ordering (IMPARO_PROF=1 `[prof] tail`): E4B 84-96 ms per response,
    // LFM2 47-51 ms -- two to four decode tokens the client waited for after its last
    // delta (most of it the turn-close forward, since deleted; see below). A write
    // failure still commits the turn; the error is returned after.
    // The request's log line goes out BEFORE the response: kv_gates and speed_gate read
    // it as soon as the client has its answer. `free_blocks` is therefore the pool's
    // count before this turn's seal.
    let free_blocks = engine.pool.as_ref().map_or(String::new(), |pl| {
        format!(" free_blocks={}", pl.free_blocks())
    });
    eprintln!(
        "[imparo] conv={} prompt={prompt_tokens} reused={start_pos} \
               gen={completion_tokens}{free_blocks} \
               prefill_ms={prefill_ms:.0} ({:.1} tok/s) decode_ms={decode_ms:.0} ({:.2} tok/s)",
        effective_label.as_deref().unwrap_or(&conversation),
        (prompt_tokens - start_pos) as f64 / (prefill_ms / 1e3).max(1e-9),
        decode_steps as f64 / (decode_ms / 1e3).max(1e-9)
    );
    let sent = (|| -> std::io::Result<()> {
        if streaming {
            if reasoning.len() > sent_reasoning {
                http::sse(
                    stream,
                    &json!({
                    "id": id, "object": "chat.completion.chunk", "model": "imparo",
                    "choices": [{"index": 0,
                                 "delta": {"reasoning_content": &reasoning[sent_reasoning..]},
                                 "finish_reason": null}]}),
                )?;
            }
            if body.len() > sent_visible {
                http::sse(
                    stream,
                    &json!({
                    "id": id, "object": "chat.completion.chunk", "model": "imparo",
                    "choices": [{"index": 0, "delta": {"content": &body[sent_visible..]},
                                 "finish_reason": null}]}),
                )?;
            }
            if !tool_calls.is_empty() {
                http::sse(
                    stream,
                    &json!({
                    "id": id, "object": "chat.completion.chunk", "model": "imparo",
                    "choices": [{"index": 0, "delta": {"tool_calls": tool_calls},
                                 "finish_reason": null}]}),
                )?;
            }
            http::sse(
                stream,
                &json!({
                "id": id, "object": "chat.completion.chunk", "model": "imparo",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage, "timings": timings}),
            )?;
            stream.write_all(b"data: [DONE]\n\n")?;
            stream.flush()
        } else {
            http::json(
                stream,
                200,
                &json!({
                "id": id, "object": "chat.completion", "model": "imparo",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage, "timings": timings}),
            )
        }
    })();
    let t_tail = std::time::Instant::now();
    if pipelined && queued {
        // The step queued behind the last confirmed one ran for a token that will not be
        // emitted; its position is not advanced into and its state advance is undone.
        if let Err(e) = engine.model.discard_step() {
            eprintln!("[imparo] discard_step: {e}");
        }
    }
    let tail_discard_ms = t_tail.elapsed().as_secs_f64() * 1e3;
    // THE RECORDED STREAM STAYS SHORT. Two tokens of this turn never get a KV row here:
    // the last generated one (its logits would be unread) and the stop token the loop
    // dropped. A turn close used to forward them -- a full weight pass for 4-5 tokens,
    // 63 ms on E4B and 44 ms on LFM2 per response -- and the rows landed above the
    // 64-token grid cut, where the next request's resume point re-runs them anyway
    // (`resume_point`; docs/evidence/bracket/2026-09-05-response-tail.md section 3).
    // The next prompt carries those tokens (the template renders the closed turn), so
    // it prefills them together with its user message. Every token in `engine.resident`
    // has a row, which is what a resume past it needs; a re-render that parts from the
    // recorded stream inside this turn is the next request's own rewind.
    let t_commit = std::time::Instant::now();
    // WRITE-THROUGH at the turn boundary: commit the resident conversation's
    // sealed units + checkpoint. put_unit skips units the store already holds, so
    // a turn costs only its new tail units plus the superseding checkpoint.
    // (Synchronous for now; the async overlap is a measured-later optimization.)
    if engine.pool.is_some() {
        let mut wrote = false;
        let final_tokens = engine.resident.clone();
        let hashes_full = imparo_kv::unit_ids(&engine.root, &final_tokens);
        // The SAME label this request began under. Recomputing it here read the hash
        // of the last sealed unit of prompt+generated, while `begin` had read the
        // prompt's -- so a keyless turn whose generation crossed a unit boundary ended
        // under a name that did not exist, `end` returned silently, and the turn
        // sealed nothing at all.
        let label = effective_label
            .clone()
            .unwrap_or_else(|| conversation_label(&conversation, &hashes_full));
        let Engine {
            model,
            store,
            disk,
            pool,
            ..
        } = &mut *engine;
        if let Some(pl) = pool.as_mut() {
            if let Some((branch, bhashes, blabel, false)) = &pool_branch {
                // deferred branch point: record now that the forward SUCCEEDED
                let units = branch / imparo_kv::grid_tokens();
                if imparo_model::log_on() {
                    eprintln!("[imparo] kv branch deferred at {branch}: units={units}");
                }
                if let Some(tip) = bhashes.get(units.wrapping_sub(1)) {
                    // `final_tokens` starts with the prompt, so it carries the
                    // same tokens `ids` did below the branch (`ids` itself moved
                    // into engine.resident above).
                    let ids = &final_tokens;
                    let prev = pl.prev_ckpt_at(blabel, bhashes, ids, branch - 1);
                    pl.note_branch(
                        blabel,
                        *tip,
                        prev,
                        *branch,
                        ids[units * imparo_kv::grid_tokens()..*branch].to_vec(),
                    );
                }
            }
            if let Err(e) = pl.end(&label, final_tokens, &hashes_full) {
                eprintln!("[imparo] kv pool end: {e}");
            }
            // WRITE-THROUGH, here and not at switch-out: by the time this conversation
            // is evicted or the process is asked to stop, its units and its checkpoint
            // are already on disk, so neither is the moment a long conversation
            // discovers it has to write everything it ever computed.
            match pl.commit_to_disk(&**model, store.as_ref(), disk.as_ref(), &label) {
                Ok(w) => wrote = w,
                Err(e) => eprintln!("[imparo] kv write-through: {e}"),
            }
        }
        // Only when something was written: a GC pass stats every unit file in the store
        // twice, and finds nothing new when nothing was added.
        if wrote {
            if let Some(dq) = engine.disk.as_ref() {
                dq.gc(engine.disk_cap);
            }
        }
    } else if let Some(store) = &engine.store {
        if let Some(state) = engine.model.kv_spill() {
            // One call, shared with `imparo-forward --spill`: the extents, the anchor
            // link and the manifest all come from the store. This was a copy of that
            // code, and the copy is where the second blob format came from.
            let staged = match store.stage_whole(
                &engine.root,
                &engine.resident[..state.boundary],
                &state,
            ) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("[imparo] kv spill: {e}");
                    return Ok(());
                }
            };
            let label = if conversation == "default" {
                conversation_label(&conversation, &staged.unit_hashes())
            } else {
                conversation.clone()
            };
            if !label.is_empty() {
                // The legacy path is only reached with an explicit label, so never keyless.
                if let Err(e) =
                    store.commit(&label, &staged.manifest(false), &staged.at)
                {
                    eprintln!("[imparo] kv commit: {e}");
                }
                if let Err(e) = store.gc(engine.disk_cap) {
                    eprintln!("[imparo] kv gc: {e}");
                }
            }
        }
    }
    if probe {
        eprintln!(
            "[prof] tail discard={tail_discard_ms:.1} commit={:.1} \
             total={:.1} ms (stop={})",
            t_commit.elapsed().as_secs_f64() * 1e3,
            t_tail.elapsed().as_secs_f64() * 1e3,
            if generated.len() >= max_tokens {
                "max_tokens"
            } else {
                "eos"
            },
        );
        let n = decode_steps.max(1) as f64;
        eprintln!(
            "[imparo] cpu-between-forwards per token: sample={:.3} ms detok={:.3} ms \
                   emit={:.3} ms  total={:.3} ms",
            t_sample / n,
            t_detok / n,
            t_send / n,
            (t_sample + t_detok + t_send) / n
        );
    }
    imparo_model::host::prof_log("decode ", decode_ms);
    // A request boundary is the one place where returning free pages costs nothing: the
    // next request will fault back in only what it actually uses.
    imparo_model::host::release_free_memory();
    // Sample after a request has run: the startup samples are taken before any GPU buffer
    // exists, so they miss the KV pool and the activation buffers entirely -- and those
    // are exactly what the prefill batch size trades against speed.
    // Hand back the request's transients: the decoded strings, the JSON, the token vecs.
    // The allocator keeps them in its arena for reuse, and `phys_footprint` charges for
    // them either way, so a server that has served N requests carries N requests' worth of
    // arena unless it asks. After `decode_ms` is taken, so it costs no measured time.
    imparo_model::host::release_free_memory();
    imparo_model::host::log_footprint("after request");

    sent
}

/// Greedy at temperature 0; otherwise softmax sampling with a fixed stream of
/// deterministic pseudo-randomness is deliberately NOT provided -- the harness measures
/// at temperature 0 and a sampler is separate work.
///
/// Used for the FIRST token only (the pick from the prefill logits); decode steps sample
/// on the GPU inside `forward_next`, which applies the same rule.
fn sample(logits: &[f32], _temperature: f64) -> u32 {
    imparo_model::ops::argmax_f32(logits)
}

#[cfg(test)]
mod pool_eligibility_tests {
    use super::{
        PoolEligibility, host_fit_policy_for_device_blocks, parse_raw_input_ids,
        pool_eligibility,
    };
    use imparo_backend::{PoolAddressing, PoolCaps, PoolCapsIssue, Tier};
    use serde_json::json;

    const SHARED: PoolCaps = PoolCaps {
        page_cells: 64,
        finest_cut_tokens: 16,
        paged_reads: true,
        shared_address: true,
        tiers: &[Tier::Unified, Tier::Disk],
    };
    const DISCRETE: PoolCaps = PoolCaps {
        page_cells: 64,
        finest_cut_tokens: 16,
        paged_reads: true,
        shared_address: false,
        tiers: &[Tier::Device, Tier::Host, Tier::Disk],
    };
    const CURRENT_CUDA: PoolCaps = PoolCaps {
        page_cells: 64,
        finest_cut_tokens: 16,
        paged_reads: true,
        shared_address: false,
        tiers: &[Tier::Device],
    };

    #[test]
    fn invalid_caps_disable_pool_with_every_reason() {
        assert_eq!(
            pool_eligibility(true, true, true, Some(CURRENT_CUDA)),
            PoolEligibility::UnsafeCaps(vec![
                PoolCapsIssue::DiscreteAddressNeedsDeviceHostDisk
            ])
        );
    }

    #[test]
    fn valid_shared_caps_enable_pool_without_a_transfer_route() {
        assert_eq!(
            pool_eligibility(true, true, true, Some(SHARED)),
            PoolEligibility::Ready(PoolAddressing::Shared)
        );
    }

    #[test]
    fn valid_discrete_caps_enable_explicit_host_transfers() {
        assert_eq!(
            pool_eligibility(true, true, true, Some(DISCRETE)),
            PoolEligibility::Ready(PoolAddressing::ExplicitHostTransfers)
        );
    }

    #[test]
    fn explicit_pool_disable_has_priority_over_invalid_backend_caps() {
        assert_eq!(
            pool_eligibility(false, true, true, Some(CURRENT_CUDA)),
            PoolEligibility::DisabledByConfig
        );
    }

    #[test]
    fn no_backend_preserves_the_quiet_disabled_behavior() {
        assert_eq!(
            pool_eligibility(true, true, true, None),
            PoolEligibility::NoBackend
        );
    }

    #[test]
    fn no_layers_and_unselected_backend_remain_disabled() {
        assert_eq!(
            pool_eligibility(true, false, true, Some(SHARED)),
            PoolEligibility::NoPooledLayers
        );
        assert_eq!(
            pool_eligibility(true, true, false, Some(SHARED)),
            PoolEligibility::BackendNotSelected
        );
    }

    #[test]
    fn explicit_host_fit_can_swap_two_complete_device_working_sets() {
        let unit_blocks = imparo_kv::resident::UNIT_BLOCKS as u32;
        assert!(host_fit_policy_for_device_blocks(unit_blocks - 1).is_none());
        assert_eq!(
            host_fit_policy_for_device_blocks(4 * unit_blocks)
                .expect("four device units")
                .minimum_units,
            8
        );
        assert_eq!(
            host_fit_policy_for_device_blocks(16 * unit_blocks)
                .expect("sixteen device units")
                .minimum_units,
            32
        );
    }

    #[test]
    fn raw_input_ids_are_disabled_by_default_and_fail_closed_when_enabled() {
        let body = json!({"input_ids": [2, 1001, 1002]});
        assert_eq!(parse_raw_input_ids(&body, false).unwrap(), None);
        assert_eq!(
            parse_raw_input_ids(&body, true).unwrap(),
            Some(vec![2, 1001, 1002])
        );
        assert!(parse_raw_input_ids(&json!({"input_ids": []}), true).is_err());
        assert!(parse_raw_input_ids(&json!({"input_ids": [2, -1]}), true).is_err());
        assert!(
            parse_raw_input_ids(&json!({"input_ids": [u64::from(u32::MAX) + 1]}), true)
                .is_err()
        );
    }
}
