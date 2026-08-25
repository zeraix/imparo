//! Dev tool: build a model plan from a GGUF file and CHECK IT against the file's tensors.
//!
//! The point is not to print a plan. It is to prove the plan agrees with the weights: a
//! wrong head-dimension assignment produces a plausible plan and a broken model, and the
//! only cheap way to catch it is to compare against the shapes actually in the file.

use std::collections::BTreeMap;
use std::path::PathBuf;

use imparo_model::{Attention, KvSource, build_plan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: imparo-plan MODEL.gguf")?,
    );
    let t0 = std::time::Instant::now();
    let document = imparo_gguf::read(&path)?;
    let read_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = std::time::Instant::now();
    let plan = build_plan(&document, &path)?;
    let plan_ms = t1.elapsed().as_secs_f64() * 1e3;

    let c = &plan.config;
    println!(
        "load  gguf_v{} tensors={} metadata={} read_ms={:.1} plan_ms={:.1}",
        document.version,
        document.tensors.len(),
        document.metadata.len(),
        read_ms,
        plan_ms
    );
    println!(
        "model arch={} layers={} embd={} ff={} heads={} kv_heads={} ctx={} vocab={} eps={:e}",
        c.architecture,
        c.n_layers,
        c.n_embd,
        c.n_ff,
        c.n_heads,
        c.n_kv_heads,
        c.context_length,
        c.vocab_size,
        c.norm_eps
    );
    println!(
        "embed scale_sqrt={} per_layer_dim={:?}",
        plan.embed.scale_by_sqrt_embd, plan.embed.per_layer_dim
    );
    println!(
        "out   final_norm={} softcap={:?} tied={}",
        plan.output.final_norm, plan.output.logit_softcap, plan.output.tied_embeddings
    );

    // attention kind census
    let mut kinds: BTreeMap<&str, u32> = BTreeMap::new();
    for l in &plan.layers {
        *kinds.entry(l.attention.label()).or_insert(0) += 1;
    }
    println!(
        "attn  {}",
        kinds
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let pooled = plan.pooled_layers();
    println!(
        "kv    pooled_layers={} of {} -> {:?}",
        pooled.len(),
        c.n_layers,
        pooled
    );
    for (kind, bytes) in plan.kv_bytes_per_token(2) {
        println!("kv    {kind}: {bytes} bytes/token across all owning layers (fp16)");
    }

    // ---- the check that matters: plan vs the weights actually in the file ----
    let dims: BTreeMap<&str, &Vec<u64>> = document
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), &t.dimensions))
        .collect();
    let mut mismatches = 0_u32;
    let mut checked = 0_u32;
    // A tensor that is ABSENT must be checked too. The first version of this loop did
    // `else { continue }`, so 18 layers were silently skipped -- and those were exactly
    // the layers whose KV-sharing the plan got wrong. It printed 90 checked / 0
    // mismatches while observing nothing about them. Presence is now part of the check.
    for layer in &plan.layers {
        // A BLOCK THAT DOES NOT ATTEND IS CHECKED AGAINST ITS OWN TENSORS. Demanding
        // attn_q/k/v from a recurrent block reported 66 mismatches on a model whose plan
        // was correct -- the check, not the plan, did not know the block kind.
        //
        // WHICH tensors those are is ARCHITECTURE-specific, not a property of the block
        // kind: LFM2 spells them shortconv.*, a state-space model spells them ssm_*. The
        // shared plan carries the state SIZE and stops there, so this table is keyed on the
        // architecture and an unlisted one is checked for nothing rather than checked
        // against a guess.
        let (suffixes, hd) = if layer.attention.is_attention() {
            let hd = layer.attention.head_dim();
            let owns = layer.kv_source == KvSource::Own;
            let expect_q = u64::from(c.n_heads) * u64::from(hd);
            let expect_kv = u64::from(c.n_kv_heads) * u64::from(hd);
            (
                vec![
                    ("attn_q", expect_q, true),
                    ("attn_k", expect_kv, owns),
                    ("attn_v", expect_kv, owns),
                ],
                hd,
            )
        } else {
            let w = u64::from(c.n_embd);
            let per: Vec<(&str, u64, bool)> = match c.architecture.as_str() {
                // conv is (l_cache, width) with the taps fastest-varying, so the LAST
                // dimension is the channel count; in_proj carries b, c and x concatenated.
                "lfm2" => vec![
                    ("shortconv.conv", w, true),
                    ("shortconv.in_proj", 3 * w, true),
                    ("shortconv.out_proj", w, true),
                ],
                _ => vec![],
            };
            (per, 0)
        };
        for (suffix, expect, required) in suffixes {
            let name = format!("blk.{}.{suffix}.weight", layer.index);
            let found = dims.get(name.as_str());
            checked += 1;
            match (found, required) {
                (None, true) => {
                    mismatches += 1;
                    if mismatches <= 8 {
                        println!(
                            "MISSING  {name} -- plan says this layer is a {}",
                            layer.attention.label()
                        );
                    }
                }
                (Some(d), false) => {
                    mismatches += 1;
                    if mismatches <= 8 {
                        println!(
                            "UNEXPECTED {name} dims={d:?} -- plan says KV is shared"
                        );
                    }
                }
                (Some(d), true) => {
                    let out = d.last().copied().unwrap_or(0);
                    if out != expect {
                        mismatches += 1;
                        if mismatches <= 8 {
                            println!(
                                "MISMATCH {name} dims={d:?} out={out} expected={expect} \
                                      ({} head_dim={hd})",
                                layer.attention.label()
                            );
                        }
                    }
                }
                (None, false) => {}
            }
        }
    }
    println!("check tensors_checked={checked} mismatches={mismatches}");

    // KV sharing census
    let shared = plan
        .layers
        .iter()
        .filter(|l| l.kv_source != KvSource::Own)
        .count();
    println!(
        "check kv_shared_layers={shared} kv_own_layers={}",
        plan.layers.len() - shared
    );

    // per-layer detail, first and last of each kind, to eyeball rope/window
    let mut seen: BTreeMap<&str, u32> = BTreeMap::new();
    for l in &plan.layers {
        let n = seen.entry(l.attention.label()).or_insert(0);
        *n += 1;
        if *n <= 2 {
            match l.attention {
                Attention::Full {
                    head_dim,
                    rope_base,
                    rope_dim,
                } => println!(
                    "layer {:2} full   head_dim={head_dim} rope_base={rope_base:e} rope_dim={rope_dim} kv={:?}",
                    l.index, l.kv_source
                ),
                Attention::Window {
                    head_dim,
                    rope_base,
                    rope_dim,
                    window,
                } => println!(
                    "layer {:2} window head_dim={head_dim} rope_base={rope_base:e} rope_dim={rope_dim} window={window} kv={:?}",
                    l.index, l.kv_source
                ),
                Attention::Recurrent { r_elems, s_elems } => println!(
                    "layer {:2} recur  state r={r_elems} s={s_elems} elements/layer",
                    l.index
                ),
            }
        }
    }

    if mismatches > 0 {
        return Err(format!("{mismatches} plan/weight shape mismatches").into());
    }
    println!("OK plan agrees with weights");
    Ok(())
}
