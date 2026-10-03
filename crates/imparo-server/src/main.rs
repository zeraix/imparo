//! OpenAI-compatible HTTP server for Imparo.
//!
//! Deliberately small: one model, one worker, streaming SSE. Requests serialise, which is
//! honest about the current state -- continuous batching and the KV pool are separate work
//! and this server is the thing that will exercise them.

#![forbid(unsafe_code)]

// The chat codec comes from the PLAN, not from an architecture named here. It has to be
// reachable before the model is built: the template sniff below runs at load.
use imparo_model::chat::ChatCodec;
#[cfg(feature = "speculative")]
mod draft_pairing;
mod host_fit;
mod http;
mod sched;
mod template;

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
    #[cfg(feature = "speculative")]
    draft_spec: Option<Arc<imparo_model::speculative::DraftSpec>>,
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
    /// The same check for the FIRST turn opener, where the system prompt and tool list end.
    /// False means no checkpoint is taken there: that branch point has no slow path.
    first_scan_trusted: bool,
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
    /// Conversations that can decode at once, each in its own slot (co-batched decode,
    /// docs/continuous-batching.md). 1 runs requests one at a time.
    slots: usize,
}

#[cfg(feature = "speculative")]
impl Drop for Engine {
    fn drop(&mut self) {
        if let Err(e) = self.model.clear_draft_cache() {
            // Do not unmap shared weights after a failed GPU retirement.
            eprintln!("fatal draft cache retirement: {e}");
            std::process::abort();
        }
    }
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
    let mut draft_path: Option<PathBuf> = None;
    let mut draft_pairing_path: Option<PathBuf> = None;
    let mut draft_mask_token: Option<u32> = None;
    let mut kv_idle_s: Option<u64> = None;
    let mut parallel: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-m" | "--model" => model_path = args.next().map(PathBuf::from),
            // A DSpark drafter file, paired with the target at start (no manifest).
            "--draft" => {
                draft_path = Some(PathBuf::from(
                    args.next().ok_or("--draft requires a drafter file")?,
                ));
            }
            // A Gemma4 MTP pairing manifest (the CUDA drafter).
            "--draft-pairing" => {
                draft_pairing_path = Some(PathBuf::from(
                    args.next()
                        .ok_or("--draft-pairing requires a manifest path")?,
                ));
            }
            "--draft-mask-token" => {
                draft_mask_token = Some(
                    args.next()
                        .ok_or("--draft-mask-token requires an integer")?
                        .parse()
                        .map_err(|_| "invalid --draft-mask-token")?,
                );
            }
            "--port" => port = args.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "-c" | "--ctx" => {
                context_length = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(context_length);
            }
            // Requests that decode at once, like llama.cpp's --parallel.
            "-np" | "--parallel" => {
                parallel = Some(
                    args.next()
                        .ok_or("--parallel requires a number of requests")?
                        .parse()
                        .map_err(|_| "invalid --parallel")?,
                );
            }
            "-md" | "--mmproj" => {
                let _ = args.next();
            }
            // KV cache types, like llama.cpp's -ctk/-ctv: f16 (default) | q4_0 | q8_0.
            // USER CONFIG, not a tuned knob.
            "-ctk" | "--cache-type-k" => cache_key_type = args.next(),
            "-ctv" | "--cache-type-v" => cache_value_type = args.next(),
            // A user-supplied jinja template file overriding the GGUF-embedded one
            // (llama.cpp's --chat-template-file).
            "--chat-template-file" => template_file = args.next().map(PathBuf::from),
            // Seconds without a request before idle conversations' KV goes back to the
            // system (0: after every request). Default: the backend's residency hold.
            "--kv-idle-s" => {
                kv_idle_s = Some(
                    args.next()
                        .ok_or("--kv-idle-s requires a number of seconds")?
                        .parse()
                        .map_err(|_| "invalid --kv-idle-s")?,
                );
            }
            _ => {}
        }
    }
    #[cfg(not(feature = "speculative"))]
    if draft_path.is_some() || draft_pairing_path.is_some() || draft_mask_token.is_some() {
        return Err("draft pairing requires the speculative build feature".into());
    }
    let path = model_path.ok_or(
        "usage: imparo-server -m MODEL.gguf [--port N] [-c N] \
                                 [--cache-type-k T] [--cache-type-v T] [--kv-idle-s S] [--parallel N] [--draft DSPARK.gguf [--draft-mask-token ID]] [--draft-pairing GEMMA4_MTP.json]",
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
    #[cfg(feature = "speculative")]
    let pairing = draft_pairing::load(
        &path,
        &document,
        &plan.config.architecture,
        draft_path.as_deref(),
        draft_pairing_path.as_deref(),
        draft_mask_token,
    )?;
    // A paired drafter's file is mapped right after the target's, in the same range: its
    // tensors sit past the target's, and the weight placement covers them before any kernel
    // reads one.
    #[cfg(feature = "speculative")]
    let mut weights = match &pairing {
        Some((draft, _)) => Weights::open_with_appended(&document, &path, draft)?,
        None => Weights::open_with(&document, &path)?,
    };
    #[cfg(not(feature = "speculative"))]
    let mut weights = Weights::open_with(&document, &path)?;
    #[cfg(feature = "speculative")]
    let appended = match &pairing {
        Some((_, spec)) => spec.appended_spans(&weights, plan.config.n_layers)?,
        None => Vec::new(),
    };
    // The drafter's caches, attention dims and feature rows are sized from the plan, so the plan
    // carries the drafter before the device is enabled. Without it every request's attach is
    // refused ("the target's plan does not carry this drafter").
    #[cfg(feature = "speculative")]
    let plan = match &pairing {
        Some((_, spec)) => imparo_model::ModelPlan {
            drafter: spec.drafter_plan(&weights, plan.config.n_layers)?,
            ..plan
        },
        None => plan,
    };
    #[cfg(not(feature = "speculative"))]
    let appended: Vec<(u64, u64)> = Vec::new();
    imparo_model::backend::enable_gpu_with_appended(
        &mut weights,
        &plan,
        context_length,
        &appended,
    )?;
    #[cfg(feature = "cuda-speculative")]
    draft_pairing::apply_knobs()?;
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
    // Whether a backend holds the weights -- asked of the model, not re-read from IMPARO_GPU.
    // Reading the variable here kept its old meaning (unset = CPU) after unset came to mean
    // GPU, so a server started without it ran every request on the single-resident path.
    let eligibility = pool_eligibility(
        !std::env::var("IMPARO_KV_POOL").is_ok_and(|v| v == "0"),
        model.plan().layers.iter().any(|_| true),
        model.weights().gpu_enabled(),
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
        PoolEligibility::Ready(addressing) => match model.kv_prepare_pool_for_slots(
            if std::env::var("IMPARO_NO_REUSE").is_ok_and(|v| v == "1") {
                1
            } else {
                parallel.unwrap_or(8).max(1)
            },
            model.slot_state_bytes(),
        ) {
            Ok(()) => {
                let layers = model.kv_full_layers();
                let blocks = model.kv_capacity_blocks();
                eprintln!(
                    "[imparo] kv pool: {} full-attention layers, {blocks} blocks/layer",
                    layers.len()
                );
                let positions = blocks as usize * imparo_kv::page_cells();
                let ctx = model.kv_runtime().capacity.max(1);
                eprintln!(
                    "[imparo] kv pool tier: {positions} positions, {} conversations at the \
full {ctx}-token context",
                    positions / ctx
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
                        ctx.div_ceil(imparo_kv::page_cells()),
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
    //
    // The FIRST opener gets the same question against a render of the system message alone:
    // where the system prompt and tool list end, the branch point every conversation with this
    // system prompt shares. It has no slow path, so a disagreement only turns that point off.
    let (scan_trusted, first_scan_trusted) = if user_open.is_empty() {
        (false, false)
    } else {
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
        let system = probe(1);
        let expect_first = full.iter().zip(&system).take_while(|(a, b)| a == b).count();
        let first = first_subsequence(&full, &user_open);
        let first_ok = first == Some(expect_first);
        if !first_ok {
            eprintln!(
                "[imparo] chat: the first turn opener is at {first:?} where the system \
message alone renders {expect_first} tokens; no checkpoint at the system prompt's end"
            );
        }
        (ok, first_ok)
    };
    // CO-BATCHED DECODE (docs/continuous-batching.md): a slot per conversation that decodes
    // at once. Only with the pool, whose page tables give each conversation blocks of its
    // own; 8 unless --parallel says otherwise. A model or cache the co-batched step does not
    // serve yet runs requests one at a time, as before.
    let slots = if pool_mode.is_none()
        || std::env::var("IMPARO_NO_REUSE").is_ok_and(|v| v == "1")
    {
        1
    } else {
        parallel.unwrap_or(8).max(1)
    };
    let mut pool_mode = pool_mode;
    let slots = if slots > 1 {
        match model.set_slots(u32::try_from(slots)?) {
            Ok(()) => {
                // A slot's own state (rings, recurrent buffers) is device memory beside the
                // pool's blocks: every slot but the first is charged its pages while it holds
                // them (docs/continuous-batching.md, section 4).
                let state = model.slot_state_bytes();
                let pages = pool_mode.as_mut().map_or(0, |pl| {
                    let pages = usize::try_from(state)
                        .unwrap_or(usize::MAX)
                        .div_ceil(pl.page_bytes().max(1));
                    pl.set_slot_pages(pages);
                    if let Some(bytes) = imparo_model::backend::active()
                        .and_then(imparo_backend::Backend::slot_state_budget_bytes)
                    {
                        let state_units = usize::try_from(bytes).unwrap_or(usize::MAX)
                            / pl.page_bytes().max(1);
                        assert!(pl.set_slot_state_budget(state_units));
                        eprintln!("[imparo] co-batched state budget: {:.1} MiB outside committed KV pages", bytes as f64 / 1048576.0);
                    }
                    pages
                });
                eprintln!(
                    "[imparo] co-batched decode: {slots} slots, {:.1} MiB of state each beyond the first ({pages} pages)",
                    state as f64 / f64::from(1u32 << 20)
                );
                slots
            }
            Err(e) => {
                eprintln!(
                    "[imparo] co-batched decode off ({e}); requests run one at a time"
                );
                1
            }
        }
    } else {
        1
    };
    let disk = store.clone().map(imparo_kv::disk::DiskQueue::new);
    let engine = Arc::new(Mutex::new(Engine {
        model,
        #[cfg(feature = "speculative")]
        draft_spec: pairing.map(|(_, spec)| Arc::new(spec)),
        tok,
        chat,
        bos_text,
        user_open,
        scan_trusted,
        first_scan_trusted,
        ctx: context_length,
        store,
        disk,
        root,
        pool: pool_mode,
        disk_cap,
        resident_conv: String::new(),
        resident: Vec::new(),
        slots,
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
            let mut engine = sched::lock_engine(&engine);
            let Engine {
                model,
                store,
                disk,
                pool,
                ..
            } = &mut *engine;
            if let Some(pl) = pool.as_mut() {
                // A request still running is lost with the process; its conversation is
                // not written half made.
                if let Some(label) = pl.active.clone().filter(|l| !pl.is_pinned(l)) {
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
    // THE KV TIER'S RELEASE. Idle conversations stay resident while the server is busy, so a
    // switch back costs nothing; they leave when it has been idle for the backend's release
    // window, or when the system reports memory pressure.
    {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || kv_trimmer(&engine, kv_idle_s));
    }
    // THE ENGINE LOOP: the one thread that runs chat requests (sched.rs). A connection's
    // thread reads its request, hands it over as a job and sends what the loop writes back.
    let (jobs, queue) = std::sync::mpsc::channel::<Job>();
    let submitted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    {
        let engine = Arc::clone(&engine);
        let submitted = Arc::clone(&submitted);
        std::thread::spawn(move || sched::run(&engine, &queue, &submitted));
    }
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    eprintln!("[imparo] listening http://127.0.0.1:{port}");
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let engine = Arc::clone(&engine);
        let jobs = jobs.clone();
        let submitted = Arc::clone(&submitted);
        std::thread::spawn(move || {
            if let Err(e) = serve_one(stream, &engine, &jobs, &submitted) {
                eprintln!("[imparo] connection error: {e}");
            }
        });
    }
    Ok(())
}

/// When the last request finished, in milliseconds on `process_ms`'s clock; 0 before any.
static LAST_REQUEST_END_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn process_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    u64::try_from(START.get_or_init(Instant::now).elapsed().as_millis())
        .unwrap_or(u64::MAX)
}

fn note_request_end() {
    LAST_REQUEST_END_MS
        .store(process_ms().max(1), std::sync::atomic::Ordering::Relaxed);
}

/// Releases idle conversations' KV (`PoolMode::trim`) once the server has been idle for the
/// backend's release window -- the one after which the weights' residency lets go too -- or
/// as soon as the system reports memory pressure. Once per idle stretch: a trim with nothing
/// resident gives nothing back, and the next request starts a new stretch.
///
/// Only where the backend's committed KV follows the blocks in use does a release give memory
/// back. Fixed arenas also use a configured idle window to release private CoBatch
/// slot buffers. The idle release hands conversations to the disk tier, so without
/// one only memory pressure releases.
///
/// `--kv-idle-s`, else IMPARO_KV_IDLE_S, sets the window (0: release right after every
/// request); unset, it is the backend's residency hold.
fn kv_trimmer(engine: &Mutex<Engine>, kv_idle_s: Option<u64>) {
    let Some(be) = imparo_model::backend::active() else {
        return;
    };
    let (has_disk, has_extra_slots) = {
        let engine = sched::lock_engine(engine);
        (engine.disk.is_some(), engine.slots > 1)
    };
    let window = if has_disk {
        kv_idle_s
            .or_else(|| {
                std::env::var("IMPARO_KV_IDLE_S")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
            })
            .map(std::time::Duration::from_secs)
            .or_else(|| be.idle_release_after())
    } else {
        None
    };
    // A fixed KV arena still owns independently releasable conversation buffers.
    // Honor the configured idle window for those slots too; kv_release remains a
    // no-op on fixed arenas, while release_slot gives their private state back.
    if !be.kv_commits_on_demand() && !(has_extra_slots && window.is_some()) {
        return;
    }
    match window {
        Some(w) => eprintln!(
            "[imparo] kv idle release: after {} s without a request, or on memory pressure",
            w.as_secs()
        ),
        None => eprintln!("[imparo] kv idle release: on memory pressure only"),
    }
    process_ms();
    let mut released_after = 0_u64;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(250));
        let pressure = be.take_memory_pressure();
        let last = LAST_REQUEST_END_MS.load(std::sync::atomic::Ordering::Relaxed);
        let idle_for =
            std::time::Duration::from_millis(process_ms().saturating_sub(last));
        // Idle means no request running either: a co-batch can decode for longer than the
        // window without one ending.
        let due = window
            .is_some_and(|w| last != 0 && last != released_after && idle_for >= w)
            && sched::IN_FLIGHT.load(std::sync::atomic::Ordering::Relaxed) == 0;
        if !pressure && !due {
            continue;
        }
        // This thread never exits: what the trim autoreleases is freed with this scope.
        let _pool = imparo_model::host::AutoreleaseScope::open();
        let mut engine = sched::lock_engine(engine);
        // A request may have run while this waited for the lock: its end starts a new
        // stretch, and only pressure releases before that one has passed the window.
        let last_now = LAST_REQUEST_END_MS.load(std::sync::atomic::Ordering::Relaxed);
        if !pressure && last_now != last {
            continue;
        }
        let Engine {
            model, pool, slots, ..
        } = &mut *engine;
        let Some(pl) = pool.as_mut() else {
            return;
        };
        let t0 = Instant::now();
        let before = be.kv_committed_bytes();
        match pl.trim(&mut **model) {
            Ok(t) => {
                if imparo_model::log_on() || t.conversations > 0 {
                    eprintln!(
                        "[imparo] kv trim ({}): {} conversation(s) released, blocks {} -> {}, \
committed {:.1} -> {:.1} MiB in {:.1} ms",
                        if pressure { "memory pressure" } else { "idle" },
                        t.conversations,
                        t.blocks_before,
                        t.blocks_after,
                        before as f64 / f64::from(1u32 << 20),
                        be.kv_committed_bytes() as f64 / f64::from(1u32 << 20),
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            Err(e) => eprintln!("[imparo] kv trim: {e}"),
        }
        // A slot no conversation occupies gives its buffers back; its next request makes them
        // again. The selected slot keeps its own: a lone client's next turn runs there.
        let gpu_before = be.allocated_bytes();
        let mut released = 0;
        for s in 0..*slots {
            if s == pl.selected_slot() || pl.occupant(s).is_some() {
                continue;
            }
            match model.release_slot(u32::try_from(s).unwrap_or(u32::MAX)) {
                Ok(true) => {
                    // Its rings went with its buffers: no conversation resumes from them.
                    pl.slot_released(s);
                    released += 1;
                }
                Ok(false) => {}
                Err(e) => eprintln!("[imparo] slot {s} release: {e}"),
            }
        }
        if imparo_model::log_on() || released > 0 {
            eprintln!(
                "[imparo] slot release ({}): {released} slot(s) gave their buffers back, \
device {:.1} -> {:.1} MiB",
                if pressure { "memory pressure" } else { "idle" },
                gpu_before as f64 / f64::from(1u32 << 20),
                be.allocated_bytes() as f64 / f64::from(1u32 << 20)
            );
        }
        released_after = last_now;
    }
}

/// The compiled chat template, set once at startup (None = use the built-in renderer).
static TEMPLATE: std::sync::OnceLock<Option<template::Template>> =
    std::sync::OnceLock::new();

/// Bytes of `b` that are SETTLED: everything except a trailing UTF-8 sequence still arriving.
///
/// `from_utf8_lossy` turns an incomplete trailing sequence into U+FFFD, and the next token
/// replaces that with the real character -- so the decoded string is not byte-monotone
/// across prefixes, while every streaming offset assumes it is:
///
/// ```text
///   bytes ..X F0 9F 9F      lossy -> "..X\u{FFFD}"   len 68, streamed through 68
///   bytes ..X F0 9F 9F 9F   lossy -> "..X🟢"          the char now spans 65..69
///   -> &v[68..] panics: "start byte index 68 is not a char boundary"
/// ```
///
/// Measured on a Qwen3.8-27B streamed answer that contained an emoji; it killed the request
/// thread. Cutting the incomplete tail makes the string a strict prefix of the next one, and
/// the character simply arrives with the token that finishes it.
///
/// An INVALID lead byte is not held: it will never complete, so the lossy decode shows its
/// replacement character now and that never changes.
fn settled_len(b: &[u8]) -> usize {
    let n = b.len();
    for back in 1..=4.min(n) {
        let c = b[n - back];
        if c & 0b1100_0000 == 0b1000_0000 {
            continue; // a continuation byte; its lead is further back
        }
        // ASCII and a byte that cannot lead a sequence both count as one.
        let need = match c {
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => 1,
        };
        return if back >= need { n } else { n - back };
    }
    n
}

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

/// Where the FIRST turn opener starts: the end of the system prompt and tool list.
fn first_subsequence(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| hay[i..i + needle.len()] == *needle)
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
    jobs: &std::sync::mpsc::Sender<Job>,
    submitted: &std::sync::atomic::AtomicUsize,
) -> std::io::Result<()> {
    // NAGLE OFF. A streamed reply is one small SSE frame per token, and Nagle holds a small
    // write until the previous one is ACKed -- so frames leave in clumps and the CLIENT sees
    // inter-token gaps the server never had. That is invisible to our own `timings` (which
    // measure the decode loop) and lands squarely in a client-side tok/s, which is the number
    // an engine comparison is read on. Costs nothing: the frames are written once each and
    // there is nothing to coalesce that we want coalesced.
    let _ = stream.set_nodelay(true);
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
        ("POST", "/v1/chat/completions") => {
            // The engine loop runs it; this thread sends what the loop writes until the
            // loop is done with the request and drops its end of the channel.
            let (tx, rx) = std::sync::mpsc::channel();
            submitted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let job = Job {
                req,
                out: http::Outbox::new(tx),
            };
            if jobs.send(job).is_err() {
                submitted.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                return http::json(
                    &mut stream,
                    503,
                    &json!({"error": "engine stopped"}),
                );
            }
            http::pump(&mut stream, &rx)
        }
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
    let engine = sched::lock_engine(engine);
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
    let mut engine = sched::lock_engine(engine);
    // A conversation a request is decoding cannot go from under it; the client erases it
    // once its response is in.
    let running: Vec<&String> = ids
        .iter()
        .filter(|i| engine.pool.as_ref().is_some_and(|pl| pl.is_running(i)))
        .collect();
    if !running.is_empty() {
        return http::json(
            stream,
            409,
            &json!({"error": "a request is running for these conversations", "ids": running}),
        );
    }
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

/// Read-only service witness, enabled only by an explicit local dump directory.
/// File I/O is outside reported prefill/decode intervals and is never a speed gate.
fn dump_service_prefill_witness(
    model: &dyn imparo_model::Model,
    logits: &[f32],
) -> std::io::Result<Option<PathBuf>> {
    let Some(root) = std::env::var_os("IMPARO_SERVICE_STATE_DUMP_DIR") else {
        return Ok(None);
    };
    static INDEX: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    let index = INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let folder = PathBuf::from(root).join(format!("request-{index:03}"));
    std::fs::create_dir_all(&folder)?;
    let raw: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(folder.join("logits.raw"), raw)?;
    let boundary = model.kv_runtime().filled;
    let live = model
        .kv_recurrent_blob(boundary)
        .ok_or_else(|| std::io::Error::other("missing service recurrent state"))?;
    std::fs::write(folder.join("live.raw"), &live)?;
    let checkpoint = model.kv_recurrent_note();
    if let Some((_, bytes)) = &checkpoint {
        std::fs::write(folder.join("checkpoint.raw"), bytes)?;
    }
    let meta = json!({"live_boundary":boundary,"logit_count":logits.len(),
        "recurrent_elements":model.plan().recurrent_elems(),
        "checkpoint_boundary":checkpoint.as_ref().map(|(at,_)|*at),
        "checkpoint_bytes":checkpoint.as_ref().map(|(_,bytes)|bytes.len())});
    std::fs::write(folder.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(Some(folder))
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

/// What reads the cache as a prefill left it, before a token is generated: the service
/// witness (IMPARO_SERVICE_STATE_DUMP_DIR) and the digest (IMPARO_KV_DIGEST=1), so two
/// conversations with the same prompt can be compared over the whole prompt. Then the
/// prefill's time goes to the profile log, apart from decode's: their shapes differ, and
/// averaging them would hide whichever one is stalling.
fn prefill_probes(
    model: &dyn imparo_model::Model,
    pool: Option<&imparo_kv::pool::PoolMode>,
    label: Option<&str>,
    ids: &[u32],
    logits: &[f32],
    prefill_ms: f64,
) -> std::io::Result<Option<PathBuf>> {
    let service_witness = dump_service_prefill_witness(model, logits)?;
    if let Some(folder) = &service_witness {
        std::fs::write(folder.join("input-ids.json"), serde_json::to_vec(ids)?)?;
    }
    if let (Some(pl), Some(label)) = (pool, label) {
        pl.digest(model, label, ids.len());
    }
    imparo_model::host::prof_log("prefill", prefill_ms);
    Ok(service_witness)
}

/// A chat request the engine loop serves: the request and where its response goes.
struct Job {
    req: http::Request,
    out: http::Outbox,
}

/// What became of a job the engine loop ran.
enum Outcome {
    /// Its response went out whole, or it failed; nothing of it is left running.
    Done,
    /// It decodes on as a row of the co-batch.
    Row(Box<sched::Row>),
    /// Rows were running when it was admitted: its prompt is prefilled a chunk per loop
    /// iteration, between their decode steps, and it joins them after its first token.
    Prefill(Box<sched::Prefilling>),
    /// The pool cannot hold it beside the rows running (docs/continuous-batching.md,
    /// section 4). Nothing of it ran; it waits at the head of the line.
    Wait(Job),
}

fn done(sent: std::io::Result<()>) -> std::io::Result<Outcome> {
    sent.map(|()| Outcome::Done)
}

/// A request's fixed facts from its prefill on: what its finish reads.
struct Turn {
    out: http::Outbox,
    seed: bool,
    prompt_tokens: usize,
    /// The reused prefix: where the prefill started.
    start_pos: usize,
    /// The client's conversation id, `default` for none.
    conversation: String,
    /// What the conversation's state is stored under; None without the pool.
    label: Option<String>,
    /// The prompt's tokens.
    ids: Vec<u32>,
    starts_in_reasoning: bool,
    pool_branches: Vec<PoolBranch>,
}

/// A branch point: its position, the prompt's unit hashes, the label, and whether it was
/// captured while the prefill passed it (eager) or is recorded at the turn's end.
type PoolBranch = (usize, Vec<imparo_kv::UnitHash>, String, bool);

/// A prompt's prefill: its segments -- to each eager branch point, where it pauses to
/// checkpoint the state, then to the prompt's end -- run a chunk at a time. A request that
/// starts with nothing running runs it to its end in one go; one admitted beside running rows
/// runs a chunk per loop iteration, between their decode steps (docs/continuous-batching.md,
/// section 3). Both make the same calls, so the cache a request leaves does not depend on
/// which way it was prefilled.
struct PromptPrefill {
    /// The pauses, ascending: a branch point, the prompt's unit hashes, the label its
    /// checkpoint is stored under.
    pauses: Vec<(usize, Vec<imparo_kv::UnitHash>, String)>,
    /// The prompt anchor: the grid point the next turn resumes from.
    anchor: Option<usize>,
    /// Whether the last segment keeps the recurrent state aside at `anchor` (a model with
    /// recurrent state); a model without it checkpoints the anchor's KV units alone.
    arm: bool,
    /// Where the running segment starts, and its forward once a chunk of it has run.
    at: usize,
    running: Option<imparo_model::Prefill>,
    /// Pauses passed.
    passed: usize,
}

impl PromptPrefill {
    /// The prefill of a `len`-token prompt from `from`: the eager branch points at or above
    /// `from` and below the end pause it. `anchor` is kept when it lies at or above the last
    /// pause: only the last segment keeps a recurrent state aside.
    fn new(
        branches: &[PoolBranch],
        from: usize,
        len: usize,
        anchor: Option<usize>,
        arm: bool,
    ) -> Self {
        let pauses: Vec<_> = branches
            .iter()
            .filter(|(branch, .., eager)| *eager && *branch >= from && *branch < len)
            .map(|(branch, hashes, label, _)| (*branch, hashes.clone(), label.clone()))
            .collect();
        let anchor =
            anchor.filter(|at| *at >= pauses.last().map_or(from, |(b, ..)| *b));
        Self {
            pauses,
            anchor,
            arm,
            at: from,
            running: None,
            passed: 0,
        }
    }

    /// After the last chunk: the prompt anchor, captured on the way, noted as the next
    /// turn's resume point (see `chat_completions`).
    fn note_anchor(
        &self,
        model: &dyn imparo_model::Model,
        pool: Option<&mut imparo_kv::pool::PoolMode>,
        root: &imparo_kv::ConfigRoot,
        label: Option<&str>,
        ids: &[u32],
    ) {
        let (Some(anchor), Some(pl), Some(label)) = (self.anchor, pool, label) else {
            return;
        };
        let hashes = imparo_kv::unit_ids(root, ids);
        if let Some(at) = pl.note_prompt_checkpoint(model, label, &hashes, ids, anchor) {
            if imparo_model::log_on() {
                eprintln!("[imparo] prompt replay anchor={at} prompt={}", ids.len());
            }
        }
    }

    /// Runs the prompt's next chunk, and checkpoints each pause the prefill reaches. True
    /// once every token has run. The request's slot must be the selected one.
    fn step(
        &mut self,
        model: &mut dyn imparo_model::Model,
        mut pool: Option<&mut imparo_kv::pool::PoolMode>,
        ids: &[u32],
        logits: &mut Vec<f32>,
        mut observer: Option<&mut imparo_model::PrefillChunkObserver<'_>>,
    ) -> std::io::Result<bool> {
        let mut ran = false;
        loop {
            let end = self.pauses.get(self.passed).map_or(ids.len(), |(b, ..)| *b);
            if end > self.at {
                if ran {
                    return Ok(false);
                }
                let tokens = &ids[self.at..end];
                let forward = match &mut self.running {
                    Some(forward) => forward,
                    running => {
                        let armed = if end == ids.len() && self.arm { self.anchor } else { None };
                        running.insert(
                            model
                                .prefill_begin(tokens.len(), self.at, armed)
                                .map_err(std::io::Error::other)?,
                        )
                    }
                };
                model
                    .prefill_chunk(forward, tokens, logits, observer.as_deref_mut())
                    .map_err(std::io::Error::other)?;
                ran = true;
                if !forward.done() {
                    return Ok(false);
                }
                if let Some(forward) = self.running.take() {
                    model.prefill_end(forward).map_err(std::io::Error::other)?;
                }
            }
            self.at = end;
            let Some((branch, hashes, label)) = self.pauses.get(self.passed) else {
                return Ok(true);
            };
            // A RESIDENT record: its tokens are compared by `matching_ckpts` against the
            // stretch above the last 256-TILE, so that is the grid they are cut on. The
            // manifest's tail is a different measurement -- above the last EXTENT -- and
            // `commit_to_disk` derives it there rather than inheriting this one.
            let units = branch / imparo_kv::grid_tokens();
            if let (Some(tip), Some(pl)) =
                (hashes.get(units.wrapping_sub(1)), pool.as_mut())
            {
                let tokens = ids[units * imparo_kv::grid_tokens()..*branch].to_vec();
                let prev = pl.prev_ckpt_at(label, hashes, ids, branch - 1);
                pl.note_ckpt(&*model, label, *tip, prev, *branch, tokens);
            }
            self.passed += 1;
        }
    }
}

/// A reply as it is generated: its tokens, their text, and what of the text has been sent.
///
/// STREAMING FEEDS THE NEW BYTES, NOT THE WHOLE ANSWER (the shape oMLX's output parser
/// uses). Splitting and parsing all of `emitted` every token is O(n) per token and so
/// O(n^2) over a response. Instead each stage asks its codec how many TRAILING bytes are
/// still undecided; everything before that is settled forever, classified once, and never
/// looked at again.
///
/// ```text
///   raw ─[unsettled_in_raw]→ split_channels ─→ reasoning ──────────────────→ delta
///                                           └→ visible ─[unsettled_in_visible]→ parse → delta
/// ```
///
/// TWO CUTS, NOT ONE, because they are different questions. Asking "is a call open?" about
/// RAW text answers it for the reasoning channel too, where a `<tool_call>` the model wrote
/// in its thinking is prose and will never close -- and holding it holds everything after
/// it, including the visible answer, to the end of generation.
#[allow(clippy::struct_excessive_bools)] // independent facts about one reply, not one state
struct Reply {
    id: String,
    streaming: bool,
    max_tokens: usize,
    chat: &'static dyn ChatCodec,
    /// The end-of-sequence and end-of-turn tokens: either ends the reply unsent.
    stops: Vec<u32>,
    generated: Vec<u32>,
    /// The generated text's bytes, appended per token: decoding the WHOLE prefix each step
    /// was quadratic in generation length (review #116, D7).
    gen_bytes: Vec<u8>,
    emitted: String,
    fed_raw: usize,
    inside: bool,
    fed_vis: usize,
    vis_split: String,
    reasoning_out: String,
    visible_out: String,
    sent_reasoning: usize,
    sent_visible: usize,
    /// The token's SSE frames, serialised here and written once (task #202).
    frames: Vec<u8>,
    /// The turn can end inside the TEXT, not only at an end token -- see
    /// `ChatCodec::turn_ends_at`. Set when it has, so the reply stops after this token's
    /// bytes have been truncated and streamed.
    turn_ended: bool,
    /// IMPARO_PROF=1, read once: the clock reads themselves are the probe's cost.
    probe: bool,
    t_detok: f64,
    t_send: f64,
}

impl Reply {
    /// A reply before its first token. `inside`: the prompt leaves the reasoning channel
    /// open.
    fn new(
        id: String,
        streaming: bool,
        max_tokens: usize,
        chat: &'static dyn ChatCodec,
        stops: Vec<u32>,
        inside: bool,
        probe: bool,
    ) -> Self {
        Self {
            id,
            streaming,
            max_tokens,
            chat,
            stops,
            generated: Vec::new(),
            gen_bytes: Vec::new(),
            emitted: String::new(),
            fed_raw: 0,
            inside,
            fed_vis: 0,
            vis_split: String::new(),
            reasoning_out: String::new(),
            visible_out: String::new(),
            sent_reasoning: 0,
            sent_visible: 0,
            frames: Vec::with_capacity(512),
            turn_ended: false,
            probe,
            t_detok: 0.0,
            t_send: 0.0,
        }
    }

    /// Hands `next` to the reply. A stop token ends it unsent; any other token is appended,
    /// and when streaming, its settled text goes out in one write. Returns whether the reply
    /// has ended: a stop token, a turn handed over inside the text, or `max_tokens` tokens.
    fn take<W: std::io::Write + ?Sized>(
        &mut self,
        next: u32,
        tok: &Tokenizer,
        stream: &mut W,
    ) -> std::io::Result<bool> {
        if self.stops.contains(&next) {
            return Ok(true);
        }
        let chat = self.chat;
        self.generated.push(next);
        let t_d = self.probe.then(Instant::now);
        tok.decode_bytes_into(
            &self.generated[self.generated.len() - 1..],
            &mut self.gen_bytes,
        );
        let piece =
            String::from_utf8_lossy(&self.gen_bytes[..settled_len(&self.gen_bytes)])
                .into_owned();
        if let Some(t) = t_d {
            self.t_detok += t.elapsed().as_secs_f64() * 1e3;
        }
        let t_e = self.probe.then(Instant::now);
        if piece.len() > self.emitted.len() {
            self.emitted = piece;
            // WHERE THE MODEL HANDS THE TURN OVER, THE TURN IS OVER. gemma4 finishes a tool
            // call by writing `<|tool_response>`, the opener of the block a tool RESULT
            // fills; run past it and the model writes the tool's answer itself. Truncating
            // HERE rather than after the loop is what makes the stream right too: a delta
            // already sent cannot be taken back over SSE.
            //
            // Searched from `fed_raw`, not from 0: the cut below holds any suffix that could
            // still grow into a marker, so a COMPLETE handover marker can only lie in the
            // bytes it has not settled yet. A `find` over the whole answer every token is
            // the same O(n^2) the split and the parse were just cured of.
            if let Some(at) = chat.turn_ends_at(&self.emitted[self.fed_raw..]) {
                // The UNTRUNCATED bytes, under the same gate as the dump below: after the
                // truncation nothing else in the process holds what the model actually
                // wrote, and that is exactly what a reader of this probe came for.
                if std::env::var("IMPARO_DUMP_GEN").is_ok_and(|v| v == "1") {
                    eprintln!("[gen-raw-begin]{}[gen-raw-end]", self.emitted);
                    let abs = self.fed_raw + at;
                    eprintln!(
                        "[imparo] turn ends at byte {abs}; the rest is the caller's"
                    );
                }
                self.emitted.truncate(self.fed_raw + at);
                self.turn_ended = true;
            }
            // THE RAW CUT RUNS IN BOTH MODES, because both need `fed_raw`: streaming
            // classifies and sends what it settles, and the turn-end search above uses it as
            // the floor that keeps itself off the whole answer. Non-streaming only advances.
            let raw_tail = &self.emitted[self.fed_raw..];
            let settled = &raw_tail[..raw_tail.len() - chat.unsettled_in_raw(raw_tail)];
            if !settled.is_empty() {
                if self.streaming {
                    // Stage 1: reasoning never streams as `content`.
                    let (r, v) = chat.split_channels(settled, self.inside);
                    self.reasoning_out.push_str(&r);
                    self.vis_split.push_str(&v);
                    self.inside = chat.channel_state_after(settled, self.inside);
                }
                self.fed_raw += settled.len();
            }
            if self.streaming {
                // Stage 2: a tool call is not visible text either. Streaming used to skip
                // this, so a client received the whole call as `content` deltas AND again as
                // `tool_calls`, while the same request non-streaming returned `content: ""`.
                // The calls are dropped here and read from the final parse; only the TEXT
                // between them is streamed.
                let vis_tail = &self.vis_split[self.fed_vis..];
                let ready =
                    &vis_tail[..vis_tail.len() - chat.unsettled_in_visible(vis_tail)];
                if !ready.is_empty() {
                    self.visible_out.push_str(&chat.parse_tool_calls(ready).0);
                    self.fed_vis += ready.len();
                }
                // Everything accumulated above is settled, so a delta is simply the part
                // not yet sent. Nothing is held back HERE any more -- both cuts happened
                // upstream, each on the text its own question is about.
                // ONE WRITE PER TOKEN: both deltas are framed into `frames` and go out
                // together (task #202); the buffer is reused across the response.
                let (r, v) = (&self.reasoning_out, &self.visible_out);
                if r.len() > self.sent_reasoning {
                    http::sse_frame(
                        &mut self.frames,
                        &json!({
                "id": self.id, "object": "chat.completion.chunk", "model": "imparo",
                "choices": [{"index": 0,
                             "delta": {"reasoning_content": &r[self.sent_reasoning..]},
                             "finish_reason": null}]}),
                    )?;
                    self.sent_reasoning = r.len();
                }
                if v.len() > self.sent_visible {
                    http::sse_frame(
                        &mut self.frames,
                        &json!({
                "id": self.id, "object": "chat.completion.chunk", "model": "imparo",
                "choices": [{"index": 0, "delta": {"content": &v[self.sent_visible..]},
                             "finish_reason": null}]}),
                    )?;
                    self.sent_visible = v.len();
                }
                http::sse_send(stream, &mut self.frames)?;
            }
        }
        if let Some(t) = t_e {
            self.t_send += t.elapsed().as_secs_f64() * 1e3;
        }
        // The last token needs no forward: its logits would never be read. Computing them
        // anyway spent a full decode step per request.
        Ok(self.turn_ended || self.generated.len() >= self.max_tokens)
    }
}

/// What a request's generation left: its reply, and the counts and timings its finish
/// reports.
struct Generation {
    prefill_ms: f64,
    service_witness: Option<PathBuf>,
    reply: Reply,
    /// Counts forwards, not tokens. The first token is free -- it comes from the prefill
    /// logits -- so N tokens cost N-1 decode steps, and llama.cpp reports its decode rate
    /// over exactly that (n_gen - 1). Dividing N tokens by N-1 forwards would report a rate
    /// 1/(N-1) too high against it.
    decode_steps: usize,
    t_decode: Instant,
    decode_ms: f64,
    t_sample: f64,
    pipelined: bool,
    queued: bool,
    /// What the drafter did, when one was attached.
    drafted: Option<Drafting>,
    /// The request stopped decoding alone to join the co-batch: the token its first
    /// co-batched step starts from, and whether the reply has it already.
    handover: Option<(u32, bool)>,
}

/// A request's decode with the drafter attached: its verify rounds, its one-token steps with
/// no draft verified, and the draft tokens the rounds verified and accepted.
struct Drafting {
    rounds: usize,
    plain: usize,
    tally: imparo_model::speculative::DraftTally,
}

fn chat_completions(
    job: Job,
    engine: &mut Engine,
    sched: &sched::Sched<'_>,
) -> std::io::Result<Outcome> {
    let Job { req, mut out } = job;
    let stream = &mut out;
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            return done(http::json(stream, 400, &json!({"error": e.to_string()})));
        }
    };
    let raw_ids = match parse_raw_input_ids(
        &body,
        std::env::var("IMPARO_DEV_RAW_TOKENS").is_ok_and(|value| value == "1"),
    ) {
        Ok(ids) => ids,
        Err(error) => return done(http::json(stream, 400, &json!({"error": error}))),
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
        return done(http::json(
            stream,
            400,
            &json!({"error": "seed cannot be combined with input_ids"}),
        ));
    }
    // The client's cap on the reply, under either OpenAI spelling. None: no cap -- the reply
    // runs to a stop token or to the end of the context, as llama.cpp's does. The request
    // reserves nothing for it up front, so no cap costs nothing (docs/memory-tiers-and-fit.md
    // section 12.4).
    let requested_max = if seed {
        Some(0)
    } else {
        body.get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .map(|v| usize::try_from(v).unwrap_or(usize::MAX))
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
    let mut new_conversation = req
        .header("x-new-conversation")
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "True"));
    // Jinja template when one is loaded, this architecture's codec as the fallback.
    // The template sees the REAL bos spelling and `render_chat_prompt` then strips one
    // prefix when the tokenizer adds its own -- exactly one owner. It used to render
    // with bos empty, which is the same answer only while every tokenizer adds BOS.
    let (adds_bos, bos_text, chat) =
        (engine.tok.add_bos, engine.bos_text.clone(), engine.chat);
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
    // DOES THE PROMPT LEAVE THE REASONING CHANNEL OPEN? Read once, here, from the text that
    // was actually rendered -- a template branches on `enable_thinking`, so the same model
    // ends its prompt inside the channel in one branch and outside it in the other. Both
    // split sites below take this; neither may assume it. (A raw-ids request renders no
    // prompt, so there is no channel to be inside.)
    let starts_in_reasoning =
        !prompt.is_empty() && chat.prompt_ends_in_reasoning(&prompt);
    // IMPARO_DUMP_PROMPT=1: print the rendered prompt between markers, for byte-diffing
    // against llama-server's /apply-template (task #7).
    if std::env::var("IMPARO_DUMP_PROMPT").is_ok_and(|v| v == "1") {
        eprintln!("[prompt-dump-begin]{prompt}[prompt-dump-end]");
    }
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
        let (a, b) = (
            probe("alpha probe", &*engine),
            probe("beta probe", &*engine),
        );
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
    let max_tokens = requested_max.unwrap_or(engine.ctx.saturating_sub(prompt_tokens));
    if prompt_tokens.saturating_add(max_tokens) > engine.ctx {
        return done(http::json(
            stream,
            400,
            &json!({
            "error": format!("prompt {prompt_tokens} + max_tokens {max_tokens} exceeds ctx {}",
                             engine.ctx)}),
        ));
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
    // Ascending: the system prompt's end when this prefill passes it, then the last user message.
    let mut pool_branches: Vec<PoolBranch> = Vec::new();
    // What the state is actually stored under, for the stats line. Without it the log
    // shows the raw header -- `conv=default` for every keyless request -- while the
    // effective label is content-derived and different for each.
    let mut effective_label: Option<String> = None;
    // Rows one decode step may write past the committed position; set with the pool's
    // reservation, read by the decode loop's `grow_room`.
    let mut lookahead = 0_usize;
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
        // THE SYSTEM PROMPT'S END: where the FIRST user message starts. Every conversation with
        // this system prompt and tool list shares everything below it, while the branch point
        // above is shared only by conversations that also share this one's history. A first
        // request that carries several messages passes both and seals both. The last user
        // message stays a branch point too: each turn's is the root the pool keeps one
        // checkpoint per turn by, so moving it would drop the older turns.
        let system_end = (!branch_off && engine.first_scan_trusted)
            .then(|| first_subsequence(&ids, &engine.user_open))
            .flatten()
            .map(|at| imparo_model::kv::resume_point(at, ids.len()))
            .filter(|&at| at >= 256 && at < branch_pos);
        let ms_branch = t_branch.elapsed().as_secs_f64() * 1e3;
        let t_label = Instant::now();
        // THE RESERVATION: every cache slot this request's forwards write. A plain decode
        // writes up to the reply's last token. A speculative round also lays its verify tree
        // over the slots after its start -- one slot per node, whatever the node's depth, up to
        // ROW_LAYOUT_MAX_ROWS -- so a tree near the end of the reply reaches past the reply. A
        // page the pool did not reserve still has an entry in the device's table (the identity
        // value, or an older table's), and a row written there lands on a block another page
        // owns: measured, a 56-row tree at 847 wrote positions 896..902 over this
        // conversation's own page at 832. The slots stop at the capacity; a round too close to
        // it drafts a chain instead (`DsparkProvider::tree_proposal`).
        // THE REQUEST RESERVES ITS PROMPT, NOT ITS BUDGET: decode grows the conversation's
        // blocks as it reaches them (`grow_room` in the decode loop). `lookahead` is the most
        // rows one step writes past the committed position -- a tree round's verify rows, and
        // the pipelined decode's queued step -- so the room is always there before the step
        // that writes it. The slots stop at the capacity; a round too close to it drafts a
        // chain instead (`DsparkProvider::tree_proposal`).
        let capacity = engine.model.kv_runtime().capacity;
        #[cfg(feature = "speculative")]
        let verify_slots = if engine.draft_spec.is_some()
            && !sched.others()
            && draft_pairing::request_enabled(&body)
            && temperature == 0.0
        {
            imparo_backend::ROW_LAYOUT_MAX_ROWS
        } else {
            0
        };
        #[cfg(not(feature = "speculative"))]
        let verify_slots = 0;
        lookahead = verify_slots + 2;
        let upper = (ids.len() + lookahead).min(capacity);
        // Where the reply can reach: what decides whether a branch point's state must be
        // captured now, before decode runs the rings past it.
        let reply_end = (ids.len() + max_tokens + verify_slots).min(capacity);
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
        // A keyless label a running request holds -- two prompts that agree up to their
        // last whole unit get the same one -- names a conversation of its own here: a label
        // is addressing only, so a fresh one can cost reuse, never a token. It borrows what
        // it restores from and supersedes nothing.
        let label = if conversation == "default"
            && engine.pool.as_ref().is_some_and(|pl| pl.is_running(&label))
        {
            new_conversation = true;
            sched::fresh_label(&label)
        } else {
            label
        };
        let ms_label = t_label.elapsed().as_secs_f64() * 1e3;
        effective_label = Some(label.clone());
        let t_begin = Instant::now();
        let Engine {
            model,
            store,
            disk,
            pool,
            slots,
            ..
        } = &mut *engine;
        if let Some(pl) = pool.as_mut() {
            // With co-batching the request decodes in a slot of its own: the one its
            // conversation occupies, else the selected one when it is free, else an empty one.
            if *slots > 1 {
                let Some(s) = sched::choose_slot(pl, &label, *slots) else {
                    return done(http::json(
                        stream,
                        503,
                        &json!({"error": "every slot is running a request"}),
                    ));
                };
                // A slot's first request makes its device buffers, and the tier gives up the
                // pages they take first. When running conversations hold those pages the
                // request waits at the head of the line; with nothing running it is refused
                // like a full house, as when the buffers cannot be made.
                match pl.charge_slot(s, &label) {
                    Ok(true) => {}
                    Ok(false) if sched.running > 0 => {
                        if imparo_model::log_on() {
                            eprintln!(
                                "[imparo] cobatch wait: conv={label} slot={s} needs room for \
the slot's own state"
                            );
                        }
                        return Ok(Outcome::Wait(Job { req, out }));
                    }
                    Ok(false) => {
                        return done(http::json(
                            stream,
                            503,
                            &json!({"error": "no room for another slot's state"}),
                        ));
                    }
                    Err(e) => {
                        return done(http::json(stream, 503, &json!({"error": e})));
                    }
                }
                if let Err(e) =
                    model.select_slot(u32::try_from(s).map_err(std::io::Error::other)?)
                {
                    return done(http::json(stream, 503, &json!({"error": e})));
                }
                pl.select_slot(s);
                // ADMISSION BY FIT (docs/continuous-batching.md, section 4): beside running rows,
                // the pages this request takes and one more per running row must be pages the
                // pool can obtain. Otherwise nothing of it runs and it waits at the head of the
                // line; with nothing running there is no check.
                if sched.running > 0 {
                    let need = upper.div_ceil(imparo_kv::grid_tokens()) + sched.running;
                    let obtainable = pl.obtainable_units();
                    if obtainable < need {
                        if imparo_model::log_on() {
                            eprintln!(
                                "[imparo] cobatch wait: conv={label} needs {need} pages, \
{obtainable} obtainable"
                            );
                        }
                        return Ok(Outcome::Wait(Job { req, out }));
                    }
                }
            }
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
                // The pool installed state AT this legal grid boundary. Never
                // resnap only the position after recurrent state was restored.
                Ok(r) => pool_pos = Some(r),
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
            let eager =
                |at: usize| reply_end.saturating_sub(at) > model.kv_checkpoint_slack();
            // The system prompt's end only when this prefill passes it: a request restored
            // past it stands on the checkpoint the request that first prefilled it sealed.
            if let Some(at) = system_end.filter(|&at| at >= pool_pos.unwrap_or(0)) {
                pool_branches.push((at, hashes.clone(), label.clone(), eager(at)));
            }
            pool_branches.push((branch_pos, hashes, label, eager(branch_pos)));
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
    // The single-resident path needs the same exact recurrent restoration as
    // Pool.begin. A snapped token position alone does not rewind ShortConv.
    let start_pos = if pool_pos.is_none()
        && start_pos > 0
        && engine.model.plan().recurrent_elems() > 0
        && engine.model.kv_runtime().filled != start_pos
    {
        let before = engine.model.kv_runtime().filled;
        let needed = engine.model.plan().recurrent_elems() as usize * 4;
        let window_ok = !engine
            .model
            .plan()
            .layers
            .iter()
            .any(|l| matches!(l.attention, imparo_model::Attention::Window { .. }))
            || engine.model.kv_window_reuse_allowed(start_pos) == Some(true);
        if let Some((at, recurrent)) =
            engine.model.kv_recurrent_note().filter(|(at, b)| {
                *at == start_pos
                    && b.len() == needed
                    && start_pos <= before
                    && window_ok
            })
        {
            engine
                .model
                .kv_resume(&imparo_kv::KvState {
                    boundary: at,
                    full: Vec::new(),
                    window: Vec::new(),
                    recurrent,
                })
                .map_err(std::io::Error::other)?;
            eprintln!("[imparo] legacy recurrent restore {before}->{at}");
            at
        } else {
            eprintln!(
                "[imparo] legacy recurrent checkpoint unavailable at {start_pos}; cold prefill"
            );
            0
        }
    } else {
        start_pos
    };

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
    let prefill_from = start_pos;
    // THE BRANCH PAUSES: an eager branch point -- where a user message starts -- is
    // checkpointed from the live state, so the prefill stops there first and then goes on.
    // A pause is part of the prefill below, not a forward run ahead of it: rows forwarded
    // ahead of the prefill never reach a drafter's history, and such a request could not draft.
    //
    // `>=`, not `>` (`PromptPrefill::new`). A conversation continuing ITSELF resumes exactly at
    // its new user message -- both are the same 64-grid point -- so `>` meant a model that is
    // always eager (recurrent: zero slack) recorded a branch on its FIRST turn and never again.
    // There is nothing to forward in that case, but there is something to checkpoint: the
    // cache is at `filled == prefill_from == branch` right now, which is exactly the state the
    // branch names. Below it, the recurrent half would already have advanced past the
    // boundary, so that case still records nothing.
    if imparo_model::log_on() {
        for (branch, ..) in pool_branches.iter().filter(|(.., eager)| *eager) {
            eprintln!(
                "[imparo] kv branch eager at {branch}: prefill_from={prefill_from} len={}",
                ids.len()
            );
        }
    }
    // THE PROMPT ANCHOR: a boundary the NEXT turn can resume from. The next turn's prompt is
    // this prompt, the client's copy of this turn's reply, and a new message. The client's copy
    // need not be the stream this request generates -- a reasoning model's history drops its
    // thinking -- so the next prompt is only sure to agree with this one up to THIS PROMPT's
    // end: the anchor is that prompt's last usable grid boundary. For 513 tokens it is 448:
    // 512 would leave a one-token tail. EVERY model keeps one. A recurrent state cannot be
    // rewound, so a model with one keeps the state at the anchor aside before decode moves
    // past it -- captured inside its existing batch, without another cut; it lies at or above
    // the last pause, so only the last segment arms it. An attention-only model's KV below the
    // anchor is never rewritten, so its anchor is the KV units alone, noted after the prefill.
    let mut prompt = PromptPrefill::new(
        &pool_branches,
        prefill_from,
        ids.len(),
        Some(imparo_model::kv::resume_point(ids.len(), ids.len())),
        engine.model.plan().recurrent_elems() > 0,
    );
    #[cfg(feature = "speculative")]
    let cache_draft =
        std::env::var("IMPARO_LAB_DRAFT_CACHE_RESUME").as_deref() != Ok("0");
    // A paired server that serves a request without its drafter says which check declined it: a
    // silent decline is indistinguishable from a slow drafter in every measurement downstream.
    #[cfg(feature = "speculative")]
    let draft_spec = engine.draft_spec.as_ref().and_then(|spec| {
        let declined = if sched.others() {
            Some("other requests are running or waiting")
        } else if !draft_pairing::request_enabled(&body) {
            Some("the request turned drafting off")
        } else if temperature != 0.0 {
            Some("the request samples")
        } else if prefill_from != 0
            && !(cache_draft
                && spec.can_draft_from(
                    &ids,
                    prefill_from,
                    !new_conversation && engine.resident_conv == conversation,
                ))
        {
            Some("the drafter's history cannot serve the reused prefix")
        } else if !spec.can_start(
            ids.len(),
            max_tokens,
            engine.model.kv_runtime().capacity,
        ) {
            Some("the start check (block, cell room, output budget or capacity)")
        } else {
            None
        };
        match declined {
            None => Some(Arc::clone(spec)),
            Some(why) => {
                eprintln!(
                    "[imparo] draft declined: {why} (prompt={} reused={prefill_from})",
                    ids.len()
                );
                None
            }
        }
    });
    // BESIDE RUNNING ROWS the prompt is prefilled a chunk per loop iteration, between their
    // decode steps, and the request joins them after its first token (docs/continuous-batching.md,
    // section 3). From here it holds its slot and its pages, so it counts as running: pinned.
    if sched.running > 0 {
        if let (Some(pl), Some(label)) =
            (engine.pool.as_mut(), effective_label.as_deref())
        {
            pl.pin(label);
        }
        let slot = engine
            .pool
            .as_ref()
            .map_or(0, imparo_kv::pool::PoolMode::selected_slot);
        let stops = [engine.tok.eos, engine.tok.eot]
            .into_iter()
            .flatten()
            .collect();
        let probe = std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1");
        return Ok(Outcome::Prefill(Box::new(sched::Prefilling {
            reply: Reply::new(
                id,
                streaming,
                max_tokens,
                chat,
                stops,
                starts_in_reasoning,
                probe,
            ),
            turn: Turn {
                out,
                seed,
                prompt_tokens,
                start_pos,
                conversation,
                label: effective_label,
                ids,
                starts_in_reasoning,
                pool_branches,
            },
            prompt,
            slot,
            logits,
            t_prefill,
            temperature,
        })));
    }
    let mut outstanding_pipeline = false;
    let generation_result = {
        let Engine {
            model: target,
            tok,
            pool,
            root,
            store,
            disk,
            ..
        } = &mut *engine;
        // Positions the conversation's blocks cover; decode grows them as it reaches them.
        let mut room = match (pool.as_ref(), effective_label.as_deref()) {
            (Some(pl), Some(label)) => pl.room(label),
            _ => usize::MAX,
        };
        let mut generate = |model: &mut dyn imparo_model::Model,
                            mut draft: Option<
            &mut dyn imparo_model::speculative::DraftProvider,
        >|
         -> std::io::Result<_> {
            if let Some(p) = draft.as_deref_mut() {
                p.initialize_at(prefill_from)
                    .map_err(std::io::Error::other)?;
                p.set_capture(true).map_err(std::io::Error::other)?;
            }
            // The prefill, all of it here, a chunk at a time. A drafter observes every chunk,
            // so its history stays contiguous across a pause.
            #[cfg(feature = "speculative")]
            let (mut committed, mut chunks) = (prefill_from, 0_usize);
            {
                #[cfg(feature = "speculative")]
                let drafting = draft.is_some();
                #[cfg(feature = "speculative")]
                let mut observe = |pos: usize, chunk: &[u32]| -> Result<(), String> {
                    let Some(p) = draft.as_deref_mut() else {
                        return Ok(());
                    };
                    if pos != committed {
                        return Err("noncontiguous draft Prefill history".into());
                    }
                    p.commit(pos, chunk)?;
                    committed = pos + chunk.len();
                    chunks += 1;
                    Ok(())
                };
                #[cfg(feature = "speculative")]
                let mut observer: Option<
                    &mut imparo_model::PrefillChunkObserver<'_>,
                > = if drafting { Some(&mut observe) } else { None };
                #[cfg(not(feature = "speculative"))]
                let mut observer: Option<
                    &mut imparo_model::PrefillChunkObserver<'_>,
                > = None;
                while !prompt.step(
                    model,
                    pool.as_mut(),
                    &ids,
                    &mut logits,
                    observer.as_deref_mut(),
                )? {}
            }
            #[cfg(feature = "speculative")]
            if draft.is_some() {
                if committed != ids.len() {
                    return Err(std::io::Error::other(
                        "incomplete draft prompt history",
                    ));
                }
                eprintln!("[imparo] draft prompt_history={committed} chunks={chunks}");
            }
            // The prompt anchor, captured before decode advances the recurrent note past the
            // prompt; its cost counts as prefill time.
            prompt.note_anchor(
                &*model,
                pool.as_mut(),
                root,
                effective_label.as_deref(),
                &ids,
            );
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
            let service_witness = prefill_probes(
                &*model,
                pool.as_ref(),
                effective_label.as_deref(),
                &ids,
                &logits,
                prefill_ms,
            )?;

            let t_decode = Instant::now();
            let mut decode_steps = 0_usize;
            // Between two forwards the GPU is idle. `gpu_busy` sits ~0.7 ms/token under wall, so
            // whatever runs here is on the critical path just as much as a kernel is.
            let mut t_sample = 0.0_f64;
            let mut t_forward = 0.0_f64;
            // Read once, outside the loop: the clock reads themselves are the probe's cost, so
            // gating only the print would leave six of them per token in the measured build.
            let probe = std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1");
            // The FIRST pick is made on the host from the prefill logits; every later one comes
            // back from `forward_next`, which runs the same greedy pick on the GPU and returns
            // only the index -- the vocab-size logits never cross to the host during decode.
            let t_s = probe.then(Instant::now);
            let mut reply = Reply::new(
                id.clone(),
                streaming,
                max_tokens,
                chat,
                [tok.eos, tok.eot].into_iter().flatten().collect(),
                starts_in_reasoning,
                probe,
            );
            let mut next = sample(&logits, temperature);
            if let Some(t) = t_s {
                t_sample += t.elapsed().as_secs_f64() * 1e3;
            }
            let mut cursor = if let Some(provider) = draft.as_deref_mut() {
                Some(
                    imparo_model::speculative::GreedyCursor::new(
                        model,
                        provider,
                        prompt_tokens,
                        next,
                        max_tokens,
                        &reply.stops,
                    )
                    .map_err(std::io::Error::other)?,
                )
            } else {
                None
            };
            // PIPELINED DECODE (docs/decode-turnaround.md): the step that consumes `next` is
            // queued before `next` is emitted, and the step after it is queued before this one's
            // pick is read, so the GPU never waits for the host between tokens. The host learns
            // each token one step late; on a stop the one step still queued is discarded. Not for
            // a request that joins the co-batch after its first token.
            let pipelined = cursor.is_none()
                && model.decode_pipelined()
                && max_tokens > 1
                && !sched.others();
            let mut queued = false;
            if pipelined {
                match model.queue_step(Some(next), prompt_tokens) {
                    Ok(()) => {
                        queued = true;
                        outstanding_pipeline = true;
                    }
                    Err(e) => return Err(std::io::Error::other(e)),
                }
            }
            // Set when another request is running or waiting and this one reaches a point a
            // plain decode step can carry on from: it leaves to decode in the co-batch.
            let mut handover: Option<(u32, bool)> = None;
            for step in 0..max_tokens {
                // Room for what this step writes, before it writes it: the pool hands out the
                // next chunk of blocks and the storage is committed (prepared ahead, so no
                // wait). Nothing was reserved for the reply up front.
                let rt = model.kv_runtime();
                let reach = (rt.filled + lookahead).min(rt.capacity);
                if reach > room {
                    if let (Some(pl), Some(label)) =
                        (pool.as_mut(), effective_label.as_deref())
                    {
                        room = pl
                            .grow_room(
                                model,
                                label,
                                reach,
                                store.as_ref(),
                                disk.as_ref(),
                            )
                            .map_err(std::io::Error::other)?;
                    }
                }
                if let (Some(c), Some(provider)) =
                    (cursor.as_mut(), draft.as_deref_mut())
                {
                    if let Some(token) =
                        c.next(model, provider).map_err(std::io::Error::other)?
                    {
                        next = token;
                    } else {
                        decode_steps = c.consumed;
                        break;
                    }
                    decode_steps = c.consumed;
                }
                if reply.take(next, tok, stream)? {
                    break;
                }
                if let Some(c) = &cursor {
                    // Between two rounds the token just handed out is the next round's
                    // anchor, which the cache does not hold yet: a plain step can take it.
                    if c.between_rounds() && sched.others() {
                        handover = Some((next, true));
                        break;
                    }
                    continue;
                }
                let pos = prompt_tokens + step;
                let hand = sched.others();
                if pipelined {
                    // The step at `pos` is already queued. Its pick is token step+1; the step
                    // after it produces token step+2, wanted only if that token can be emitted
                    // and this request goes on alone.
                    let want_more = !hand && step + 2 < max_tokens;
                    if want_more && model.queue_step(None, pos + 1).is_err() {
                        break;
                    }
                    if let Ok(id) = model.wait_step() {
                        next = id;
                    } else {
                        queued = want_more;
                        outstanding_pipeline = queued;
                        break;
                    }
                    queued = want_more;
                    outstanding_pipeline = queued;
                } else if hand {
                    handover = Some((next, true));
                    break;
                } else {
                    // THE FORWARD, TIMED APART FROM THE LOOP AROUND IT. decode_ms brackets
                    // the whole loop, so it cannot say whether a gap to another engine is
                    // the engine or the serving loop -- and on this model that is exactly
                    // the open question.
                    let t_fwd = std::time::Instant::now();
                    let r = model.forward_next(next, pos);
                    t_forward += t_fwd.elapsed().as_secs_f64() * 1e3;
                    match r {
                        Ok(id) => next = id,
                        Err(_) => break,
                    }
                }
                decode_steps += 1;
                if hand {
                    // Nothing is queued past the step just waited for; its pick is the
                    // co-batch's first input and the reply does not have it yet.
                    handover = Some((next, false));
                    break;
                }
            }
            let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
            if decode_steps > 0 && std::env::var("IMPARO_DECODE_SPLIT").is_ok() {
                eprintln!(
                    "[imparo] decode split: steps={decode_steps} total={:.3} ms/token                      forward={:.3} loop={:.3}",
                    decode_ms / decode_steps as f64,
                    t_forward / decode_steps as f64,
                    (decode_ms - t_forward) / decode_steps as f64
                );
            }
            if let Some(c) = &cursor {
                if c.consumed != decode_steps
                    || model.kv_runtime().filled != prompt_tokens + decode_steps
                {
                    return Err(std::io::Error::other(
                        "service cursor committed position mismatch",
                    ));
                }
                eprintln!(
                    "[imparo] draft blocks={} calls={} sequential={} delivered={} consumed={} filled={}",
                    c.verified_blocks,
                    c.draft_calls,
                    c.sequential_steps,
                    reply.generated.len(),
                    c.consumed,
                    model.kv_runtime().filled
                );
                if let Some(histogram) = &c.verified_consumed_histogram {
                    eprintln!("[imparo] draft-consumed-histogram {histogram:?}");
                }
            }
            Ok(Generation {
                prefill_ms,
                service_witness,
                reply,
                decode_steps,
                t_decode,
                decode_ms,
                t_sample,
                pipelined,
                queued,
                drafted: cursor.as_ref().map(|c| Drafting {
                    rounds: c.verified_blocks,
                    plain: c.sequential_steps,
                    tally: c.drafted,
                }),
                handover,
            })
        };
        #[cfg(feature = "speculative")]
        {
            if let Some(spec) = draft_spec.as_deref() {
                let mut output = None;
                let mut finished = None;
                let scoped = if cache_draft
                    && matches!(spec, imparo_model::speculative::DraftSpec::Dspark(_))
                {
                    let ran = target.with_cached_draft(
                        spec,
                        &ids,
                        prefill_from,
                        &mut |model, provider| {
                            let result = generate(model, Some(provider));
                            let status = result
                                .as_ref()
                                .map(|_| ())
                                .map_err(ToString::to_string);
                            output = Some(result);
                            finished = Some(Instant::now());
                            status
                        },
                    );
                    match ran {
                        Ok(false) => {
                            output = Some(generate(&mut **target, None));
                            finished = Some(Instant::now());
                            Ok(())
                        }
                        Ok(true) => Ok(()),
                        Err(e) => Err(e),
                    }
                } else {
                    target.with_draft(spec, &mut |model, provider| {
                        output = Some(generate(model, Some(provider)));
                        finished = Some(Instant::now());
                        Ok(())
                    })
                };
                if let Err(e) = scoped {
                    Err(std::io::Error::other(e))
                } else {
                    let mut result =
                        output.expect("draft scope did not invoke generation");
                    if let (Ok(value), Some(at)) = (&mut result, finished) {
                        value.decode_ms += at.elapsed().as_secs_f64() * 1e3;
                    }
                    result
                }
            } else {
                generate(&mut **target, None)
            }
        }
        #[cfg(not(feature = "speculative"))]
        {
            generate(&mut **target, None)
        }
    };
    let generation = match generation_result {
        Ok(generation) => generation,
        Err(e) => {
            return done(abandon(
                engine,
                effective_label.as_deref(),
                outstanding_pipeline,
                streaming,
                stream,
                e,
            ));
        }
    };
    let turn = Turn {
        out,
        seed,
        prompt_tokens,
        start_pos,
        conversation,
        label: effective_label,
        ids,
        starts_in_reasoning,
        pool_branches,
    };
    if let Some((next, taken)) = generation.handover {
        return Ok(Outcome::Row(Box::new(sched::Row::join(
            engine, turn, generation, next, taken,
        ))));
    }
    finish(engine, turn, generation, sched.running == 0)?;
    Ok(Outcome::Done)
}

/// A request prefilled beside running rows (`sched::Prefilling`), once its last chunk has run:
/// the prompt anchor noted and the probes run. Returns the prefill's milliseconds and the
/// service witness's folder. The request's slot is the selected one.
fn prefill_ended(
    engine: &mut Engine,
    p: &sched::Prefilling,
) -> std::io::Result<(f64, Option<PathBuf>)> {
    let Engine {
        model, pool, root, ..
    } = &mut *engine;
    p.prompt.note_anchor(
        &**model,
        pool.as_mut(),
        root,
        p.turn.label.as_deref(),
        &p.turn.ids,
    );
    let prefill_ms = p.t_prefill.elapsed().as_secs_f64() * 1e3;
    let witness = prefill_probes(
        &**model,
        pool.as_ref(),
        p.turn.label.as_deref(),
        &p.turn.ids,
        &p.logits,
        prefill_ms,
    )?;
    Ok((prefill_ms, witness))
}

/// ... and then its first token, picked from the prefill's logits: it becomes a row whose
/// reply does not have that token yet.
fn first_row(
    engine: &mut Engine,
    p: sched::Prefilling,
    prefill_ms: f64,
    service_witness: Option<PathBuf>,
) -> sched::Row {
    let next = sample(&p.logits, p.temperature);
    let generation = Generation {
        prefill_ms,
        service_witness,
        reply: p.reply,
        decode_steps: 0,
        t_decode: Instant::now(),
        decode_ms: 0.0,
        t_sample: 0.0,
        pipelined: false,
        queued: false,
        drafted: None,
        handover: None,
    };
    sched::Row::join(engine, p.turn, generation, next, false)
}

/// A request that failed part way: its conversation is forgotten rather than stored half
/// made, and a client still waiting for a response is told. The request's slot is the
/// selected one.
fn abandon(
    engine: &mut Engine,
    label: Option<&str>,
    outstanding_pipeline: bool,
    streaming: bool,
    stream: &mut http::Outbox,
    e: std::io::Error,
) -> std::io::Result<()> {
    // A verified block may be ahead of delivery. Abandon the transient
    // claim; never pretend changing filled rolls recurrent state back.
    if outstanding_pipeline {
        engine.model.discard_step().map_err(std::io::Error::other)?;
    }
    let filled = engine.model.kv_runtime().filled;
    engine.resident.clear();
    engine.resident_conv.clear();
    engine.model.kv_set_filled(0);
    if let (Some(pool), Some(label)) = (engine.pool.as_mut(), label) {
        pool.forget(&[label.to_owned()])
            .map_err(std::io::Error::other)?;
    }
    eprintln!(
        "[imparo] generation abandoned committed={filled} resident=0 filled=0: {e}"
    );
    // SSE headers are already sent. Do not write a second HTTP response.
    if streaming {
        return Err(e);
    }
    http::json(stream, 500, &json!({"error": e.to_string()}))
}

/// A request's end: its response, the pipelined step it left queued, and its turn written
/// through. The request's slot is the selected one. `quiet`: nothing else is running, so
/// handing free memory back to the system delays no one.
fn finish(
    engine: &mut Engine,
    turn: Turn,
    generation: Generation,
    quiet: bool,
) -> std::io::Result<()> {
    let Turn {
        mut out,
        seed,
        prompt_tokens,
        start_pos,
        conversation,
        label: effective_label,
        ids,
        starts_in_reasoning,
        pool_branches,
    } = turn;
    let Generation {
        prefill_ms,
        service_witness,
        reply,
        decode_steps,
        decode_ms,
        t_sample,
        pipelined,
        queued,
        drafted,
        ..
    } = generation;
    let Reply {
        id,
        streaming,
        max_tokens,
        chat,
        generated,
        emitted,
        sent_reasoning,
        sent_visible,
        mut frames,
        probe,
        t_detok,
        t_send,
        ..
    } = reply;
    let stream = &mut out;
    // The cache now holds the prompt plus every token that went through a forward:
    // decode_steps of the generated tokens (the last generated token is never
    // forwarded — its logits would be unread — so it is not resident).
    // A textual stop can leave accepted, un-emitted draft inputs ahead of delivery.
    // Never publish that partial transcript as a resumable target/draft state.
    let discard_partial_draft = decode_steps > generated.len();
    engine.resident = ids;
    engine
        .resident
        .extend_from_slice(&generated[..decode_steps.min(generated.len())]);
    engine.resident_conv.clone_from(&conversation);
    // IMPARO_DUMP_GEN=1: the assistant's TURN, before any split or parse -- the companion to
    // IMPARO_DUMP_PROMPT. A codec defect shows up as a marker landing in the wrong half, and
    // nothing else on the request path prints it. When the turn ended inside the text, the
    // bytes past the handover were printed above as `gen-raw`.
    if std::env::var("IMPARO_DUMP_GEN").is_ok_and(|v| v == "1") {
        eprintln!("[gen-dump-begin]{emitted}[gen-dump-end]");
    }
    let (reasoning, body) = chat.split_channels(&emitted, starts_in_reasoning);
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

    if let Some(folder) = &service_witness {
        std::fs::write(
            folder.join("generated.json"),
            serde_json::to_vec(&generated)?,
        )?;
    }
    let completion_tokens = generated.len();
    let finish = if !seed && completion_tokens >= max_tokens {
        "length"
    } else if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    // Report the same `timings` block llama.cpp does. Without it a harness comparing the
    // two engines has to split prefill from decode by wall clock on our side and by the
    // server's own numbers on theirs -- two different rulers, which charged us for decode
    // time the fork was not charged for and understated our prefill by a third.
    let processed = prompt_tokens - start_pos;
    let mut timings = json!({
        "prompt_n": processed,
        "prompt_ms": prefill_ms,
        "prompt_per_second": processed as f64 / (prefill_ms / 1e3).max(1e-9),
        "predicted_n": completion_tokens,
        "predicted_ms": decode_ms,
        "predicted_per_second": decode_steps as f64 / (decode_ms / 1e3).max(1e-9),
    });
    // THIS REQUEST'S DRAFTING, present whenever the drafter was attached: the draft tokens the
    // target verified past each round's anchor and the ones it accepted (llama.cpp calls them
    // draft_n and draft_n_accepted), the verify rounds, the one-token steps that verified no draft,
    // and the two counts by the source that proposed each token. The decode commits
    // draft_accepted + draft_rounds + plain_rounds tokens to the cache: each round its accepted
    // drafts and its anchor, each plain step one token.
    if let Some(d) = drafted {
        let counts = |t: imparo_model::speculative::Tally| json!({"verified": t.verified, "accepted": t.accepted});
        let total = d.tally.total();
        timings["draft_verified"] = json!(total.verified);
        timings["draft_accepted"] = json!(total.accepted);
        timings["draft_rounds"] = json!(d.rounds);
        timings["plain_rounds"] = json!(d.plain);
        timings["draft_sources"] = json!({
            "drafter": counts(d.tally.drafter),
            "ngram": counts(d.tally.ngram),
            "agreed": counts(d.tally.agreed),
        });
    }
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
            // The tail's frames and the terminator go out as one write (task #202).
            if reasoning.len() > sent_reasoning {
                http::sse_frame(
                    &mut frames,
                    &json!({
                    "id": id, "object": "chat.completion.chunk", "model": "imparo",
                    "choices": [{"index": 0,
                                 "delta": {"reasoning_content": &reasoning[sent_reasoning..]},
                                 "finish_reason": null}]}),
                )?;
            }
            // FLUSH FROM THE PARSED TEXT, the same string the loop streamed from. Sending
            // `body` (split but unparsed) here would put every tool call back into the
            // stream at the end, which is most of what this defect was.
            if visible.len() > sent_visible {
                http::sse_frame(
                    &mut frames,
                    &json!({
                    "id": id, "object": "chat.completion.chunk", "model": "imparo",
                    "choices": [{"index": 0, "delta": {"content": &visible[sent_visible..]},
                                 "finish_reason": null}]}),
                )?;
            }
            if !tool_calls.is_empty() {
                http::sse_frame(
                    &mut frames,
                    &json!({
                    "id": id, "object": "chat.completion.chunk", "model": "imparo",
                    "choices": [{"index": 0,
                                 "delta": {"tool_calls": streamed_tool_calls(&tool_calls)},
                                 "finish_reason": null}]}),
                )?;
            }
            http::sse_frame(
                &mut frames,
                &json!({
                "id": id, "object": "chat.completion.chunk", "model": "imparo",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage, "timings": timings}),
            )?;
            frames.extend_from_slice(b"data: [DONE]\n\n");
            http::sse_send(stream, &mut frames)
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
    // Optional local quality witness from tokens already held on the CPU.
    // Emit only after the response is sent; no GPU state dump in timed requests.
    if std::env::var("IMPARO_LAB_GENERATED_TOKEN_IDS").is_ok_and(|v| v == "1") {
        eprintln!("[generated-token-ids] {}", json!(&generated));
    }
    let t_tail = std::time::Instant::now();
    if pipelined && queued {
        // The step queued behind the last confirmed one ran for a token that will not be
        // emitted; its position is not advanced into and its state advance is undone.
        if let Err(e) = engine.model.discard_step() {
            eprintln!("[imparo] discard_step: {e}");
        }
    }
    let tail_discard_ms = t_tail.elapsed().as_secs_f64() * 1e3;
    if discard_partial_draft {
        #[cfg(feature = "speculative")]
        engine
            .model
            .clear_draft_cache()
            .map_err(std::io::Error::other)?;
        engine.resident.clear();
        engine.resident_conv.clear();
        engine.model.kv_set_filled(0);
        if let (Some(pool), Some(label)) =
            (engine.pool.as_mut(), effective_label.as_deref())
        {
            pool.forget(&[label.to_owned()])
                .map_err(std::io::Error::other)?;
        }
        return sent;
    }
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
            for (branch, bhashes, blabel, _) in
                pool_branches.iter().filter(|(.., eager)| !eager)
            {
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
    if quiet {
        // A request boundary is the one place where returning free pages costs nothing: the
        // next request will fault back in only what it actually uses. Not while other
        // requests run: the loop would hold their next step for it.
        // Hand back the request's transients: the decoded strings, the JSON, the token vecs.
        // The allocator keeps them in its arena for reuse, and `phys_footprint` charges for
        // them either way, so a server that has served N requests carries N requests' worth
        // of arena unless it asks. After `decode_ms` is taken, so it costs no measured time.
        imparo_model::host::release_free_memory();
        // Sample after a request has run: the startup samples are taken before any GPU
        // buffer exists, so they miss the KV pool and the activation buffers entirely -- and
        // those are exactly what the prefill batch size trades against speed.
        imparo_model::host::log_footprint("after request");
    }

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

/// The response's tool calls as stream deltas: each carries its `index`, because a client
/// joins streamed call deltas by index. Sent without one, three `get_weather` calls came
/// back to the client as one call named `get_weatherget_weatherget_weather` whose
/// arguments were not JSON, and the history it sent next did not render.
fn streamed_tool_calls(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, call)| {
            let mut call = call.clone();
            call["index"] = json!(i);
            call
        })
        .collect()
}

#[cfg(test)]
mod streamed_tool_call_tests {
    use super::streamed_tool_calls;
    use serde_json::json;

    #[test]
    fn every_streamed_call_carries_its_index() {
        let calls = [
            json!({"id": "call_get_weather_0", "type": "function",
                   "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}),
            json!({"id": "call_get_weather_1", "type": "function",
                   "function": {"name": "get_weather", "arguments": "{\"city\":\"Bergen\"}"}}),
        ];
        let streamed = streamed_tool_calls(&calls);
        assert_eq!(streamed.len(), 2);
        for (i, (s, c)) in streamed.iter().zip(&calls).enumerate() {
            assert_eq!(s["index"], json!(i));
            assert_eq!(s["function"], c["function"]);
            assert_eq!(s["id"], c["id"]);
        }
    }
}

#[cfg(test)]
mod streamed_text_tests {
    use super::settled_len;

    /// The decoded text must be a strict PREFIX of what the next token decodes to, or a
    /// stream offset taken now lands inside a character later.
    #[test]
    fn an_incomplete_trailing_sequence_is_held_until_it_completes() {
        let emoji = "🟢".as_bytes(); // F0 9F 9F A2, four bytes
        let mut b = b"ok ".to_vec();
        assert_eq!(settled_len(&b), 3);
        for (i, byte) in emoji.iter().enumerate() {
            b.push(*byte);
            let want = if i + 1 == emoji.len() { b.len() } else { 3 };
            assert_eq!(settled_len(&b), want, "after {} emoji bytes", i + 1);
        }
        // Every prefix decodes to a prefix of the next -- the property the stream needs.
        let mut prev = String::new();
        for n in 1..=b.len() {
            let now = String::from_utf8_lossy(&b[..settled_len(&b[..n])]).into_owned();
            assert!(now.starts_with(&prev), "{now:?} does not extend {prev:?}");
            prev = now;
        }
    }

    #[test]
    fn complete_text_is_never_held() {
        assert_eq!(settled_len(b""), 0);
        assert_eq!(settled_len("plain ascii".as_bytes()), 11);
        assert_eq!(settled_len("caf\u{e9}".as_bytes()), 5);
        assert_eq!(settled_len("\u{4e16}\u{754c}".as_bytes()), 6);
    }

    /// An INVALID lead byte never completes, so holding it would stall the stream forever.
    /// The lossy decode shows its replacement character now, and that never changes.
    #[test]
    fn an_invalid_lead_byte_is_not_held() {
        let b = b"ok \xff";
        assert_eq!(settled_len(b), b.len());
    }
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

#[cfg(test)]
mod turn_opener_scan_tests {
    use super::{first_subsequence, last_subsequence};

    /// The two branch points of one prompt: the system prompt's end is the FIRST opener, the
    /// last user message the LAST; with one user message they are the same position.
    #[test]
    fn first_and_last_opener_are_the_system_end_and_the_last_user_message() {
        let open = [7, 8];
        // system 1..4 | user (7 8) 5 | assistant 6 | user (7 8) 9
        let ids = [1, 2, 3, 4, 7, 8, 5, 6, 7, 8, 9];
        assert_eq!(first_subsequence(&ids, &open), Some(4));
        assert_eq!(last_subsequence(&ids, &open), Some(8));
        let one = [1, 2, 3, 7, 8, 5];
        assert_eq!(
            first_subsequence(&one, &open),
            last_subsequence(&one, &open)
        );
        assert_eq!(first_subsequence(&one, &[]), None);
        assert_eq!(first_subsequence(&[7], &open), None);
    }
}
