//! Where a model's weights live: the common runtime's placement across the fast and slow
//! tiers (docs/memory-tiers-and-fit.md). Computed once at load from the plan, the
//! configured context, the prefill batch and the backend's budget; every backend receives
//! the same answer and implements its tiers its own way.
//!
//! The arithmetic:
//!
//! ```text
//! reserve       = KV for the configured context (per layer from the plan)
//!               + activations for the largest prefill chunk
//!               + scratch + a small fixed margin
//! weight budget = fast tier - reserve
//! placement     = shared tensors in the fast tier first, then whole layers 0.. while they
//!                 fit, the rest in the slow tier from the end; row-gathered tensors bound only
//!                 from what is left after every layer, host-staged otherwise
//! ```

use std::collections::BTreeMap;

use imparo_backend::{TierBudget, WeightPlacement, WeightSegment, WeightTier};
use imparo_gguf::weights::Tensor;

use crate::ModelPlan;

/// The fixed margin over the computed reserve: page rounding, the backend's own small
/// buffers, the pool's tables. A number, not a fraction of the budget, so a model that fits
/// is never pushed into streaming by a percentage of a big machine.
pub const RESERVE_MARGIN_BYTES: u64 = 256 << 20;

/// `IMPARO_FAST_TIER_MB`: replace the backend's budget. `IMPARO_FAST_TIER_HEADROOM_MB`: leave
/// that much of it for other apps. Both are the user's configuration of the total budget;
/// the reserve is never configured, it is computed.
#[must_use]
pub fn configured_budget(reported: Option<u64>) -> Option<u64> {
    let mut total = std::env::var("IMPARO_FAST_TIER_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(|mb| mb << 20)
        .or(reported)?;
    if let Some(head) = std::env::var("IMPARO_FAST_TIER_HEADROOM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        total = total.saturating_sub(head << 20);
    }
    Some(total)
}

/// The KV bytes the configured context needs on the fast tier: full-attention layers hold
/// `capacity` rows, windowed layers their ring, and the recurrent state (plus its snapshot
/// twin) is fixed. K and V both.
#[must_use]
pub fn kv_reserve_bytes(plan: &ModelPlan, capacity: usize, max_batch: usize) -> u64 {
    let mut bytes = 0_u64;
    for g in crate::kv::state_geometry(plan, max_batch) {
        let rows = match g.kind {
            imparo_kv::StateKind::Full => capacity,
            imparo_kv::StateKind::Window { ring, .. } => ring,
        } as u64;
        bytes += rows * (g.k_stride as u64 + g.v_stride as u64);
    }
    // Negotiated planes + one boundary snapshot, from the same static backend
    // layout used by WorkflowState and the allocator. The requested model routes
    // alone can overstate the plane count on a fixed-state graph backend.
    let recur = u64::from(plan.recurrent_elems()) * 4;
    bytes + recur * (u64::from(plan.effective_recur_planes()) + 1)
}

/// The activation bytes the largest prefill chunk needs, from the architecture's own buffer
/// list at that batch, page-rounded the way the backend rounds. Zero when no backend is
/// active (nothing to reserve on the host path).
#[must_use]
pub fn activation_reserve_bytes(
    plan: &ModelPlan,
    capacity: usize,
    max_batch: usize,
) -> u64 {
    if crate::backend::active().is_none() {
        return 0;
    }
    let b = crate::gpu_support::batch_floor(max_batch);
    // EVERY ARCHITECTURE THAT RUNS ON A BACKEND MUST HAVE AN ARM HERE. qwen35 had none for its
    // whole life: it wrote `buffer_requirements` and nothing ever called it, so the fit reserved
    // ZERO and said so in its own line -- "activations 0 at batch 512" -- while the same run
    // allocated 241.2 MiB of activations and 598.5 MiB of recurrent state. The fast tier was
    // therefore promoted ~840 MiB past what the machine had, which is one of the ways the tier
    // stops fitting; a tier that does not fit blocks the host inside one requestResidency (122 s
    // measured, see the residency notes).
    //
    // This is the shape task #181 named: a per-kind match with a default arm. The default arm is
    // where a new architecture goes to fail silently, so it does not answer zero any more -- an
    // unknown architecture says so, once, and the wrong number is visible instead of plausible.
    let arch = plan.config.architecture.as_str();
    let reqs = match arch {
        "gemma4" => crate::gemma4::workflow_gpu::buffer_requirements(plan, b, capacity),
        "lfm2" => crate::lfm2::workflow_gpu::buffer_requirements(plan, b, capacity),
        "qwen35" => crate::qwen35::workflow_gpu::buffer_requirements(plan, b, capacity),
        _ => {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                eprintln!(
                    "[imparo] fit: architecture {arch} has no activation reservation; the fast \
                     tier will be promoted past the activations this model still allocates. \
                     Add its arm to activation_reserve_bytes."
                );
            });
            Vec::new()
        }
    };
    crate::gpu_support::layout_bytes(&reqs)
}

/// The layer a tensor belongs to, from its GGUF name (`blk.N.…`), or None for a shared
/// tensor (embeddings, output norm, lm-head, the per-layer projection tables).
#[must_use]
pub fn layer_of(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (n, _) = rest.split_once('.')?;
    n.parse().ok()
}

/// Spans closer than this merge: a gap between two tensors is alignment padding (GGUF
/// pads tensors to `general.alignment`, 32 bytes by default), never a tensor, and a
/// segment may cover padding. A page is far above any padding and far below any tensor.
const MERGE_GAP: u64 = 64 << 10;

fn merge_adjacent(mut spans: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    spans.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
    for (off, bytes) in spans {
        if let Some(last) = out.last_mut() {
            let end = last.0 + last.1;
            if off >= end && off - end <= MERGE_GAP {
                last.1 = off + bytes - last.0;
                continue;
            }
        }
        out.push((off, bytes));
    }
    out
}

/// Where a row-gathered tensor -- one read by rows indexed by token id, such as gemma4's
/// per-layer token embeddings -- goes (docs/memory-tiers-and-fit.md section 4;
/// docs/decode-turnaround.md section 1.2 for why it decides whether decode pipelines).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowGatheredPolicy {
    /// Never bound: the host copies each token's rows into a device buffer. The smallest
    /// wired set (E4B: 2.9 GB at rest against 4.5), and decode keeps its host round trip,
    /// because the next token's rows cannot be staged before the token is known.
    HostStaged,
    /// Bound in the fast tier when it still fits AFTER every layer; host-staged
    /// otherwise. The device gathers rows itself, so decode can run pipelined.
    BindIfFits,
}

/// `IMPARO_ROW_GATHERED`: `staged` keeps row-gathered tensors off the device (memory
/// first); anything else, or unset, binds them when they fit (speed first). A
/// configuration of the fit, like the budget overrides above.
#[must_use]
pub fn configured_row_gathered_policy() -> RowGatheredPolicy {
    match std::env::var("IMPARO_ROW_GATHERED").as_deref() {
        Ok("staged") => RowGatheredPolicy::HostStaged,
        _ => RowGatheredPolicy::BindIfFits,
    }
}

/// The placement itself, from the tensor table alone: pure, so it is testable without a
/// model file. `row_gathered` names the plan's row-gathered tensors; `weight_budget` is what the
/// fast tier has left after the reserve (u64::MAX when the tier is unknown).
#[must_use]
pub fn fit(
    tensors: &BTreeMap<String, Tensor>,
    n_layers: u32,
    row_gathered: &[&str],
    budget: TierBudget,
    policy: RowGatheredPolicy,
) -> WeightPlacement {
    let mut shared: Vec<(u64, u64)> = Vec::new();
    let mut row_gathered_spans: Vec<(u64, u64)> = Vec::new();
    let mut per_layer: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
    for (name, t) in tensors {
        let span = (t.offset as u64, t.bytes as u64);
        if row_gathered.contains(&name.as_str()) {
            row_gathered_spans.push(span);
        } else if let Some(l) = layer_of(name) {
            per_layer.entry(l).or_default().push(span);
        } else {
            shared.push(span);
        }
    }
    let mut segments: Vec<WeightSegment> = Vec::new();
    let mut fast_bytes: u64 = 0;
    for (off, bytes) in merge_adjacent(shared) {
        fast_bytes += bytes;
        segments.push(WeightSegment {
            offset: off,
            bytes,
            tier: WeightTier::Fast,
        });
    }
    // Whole layers in order while they fit; once one does not, every later one goes to
    // the slow tier too.
    let mut fast_layers = 0_u32;
    let mut overflowed = false;
    for (layer, spans) in per_layer {
        let spans = merge_adjacent(spans);
        let layer_bytes: u64 = spans.iter().map(|s| s.1).sum();
        if !overflowed && fast_bytes + layer_bytes <= budget.weight_budget {
            fast_bytes += layer_bytes;
            fast_layers += 1;
            for (off, bytes) in spans {
                segments.push(WeightSegment {
                    offset: off,
                    bytes,
                    tier: WeightTier::Fast,
                });
            }
        } else {
            overflowed = true;
            for (off, bytes) in spans {
                segments.push(WeightSegment {
                    offset: off,
                    bytes,
                    tier: WeightTier::Slow { layer },
                });
            }
        }
    }
    // Tables LAST: a table is bound only from what the fast tier has left after every
    // layer fitted, so it never pushes a layer into the slow tier. A staged table costs
    // the decode loop its host round trip; a bound one costs its bytes wired.
    for (off, bytes) in merge_adjacent(row_gathered_spans) {
        let bind = policy == RowGatheredPolicy::BindIfFits
            && !overflowed
            && fast_bytes + bytes <= budget.weight_budget;
        if bind {
            fast_bytes += bytes;
        }
        segments.push(WeightSegment {
            offset: off,
            bytes,
            tier: if bind {
                WeightTier::Fast
            } else {
                WeightTier::HostStaged
            },
        });
    }
    segments.sort_by_key(|s| s.offset);
    // ONE BUFFER FOR THE FAST TIER where the file allows it. The fast tier is whole layers
    // in file order, so its segments are adjacent except where a table sits between them;
    // adjacent fast segments merge into one. A backend then binds one resource for the
    // whole fast tier instead of one per layer -- Metal's per-command-buffer bookkeeping
    // is per distinct resource, and a decode token that touched 31 weight buffers instead
    // of 1 read consistently ~1% slower (2026-09-04). Slow segments stay per layer: that
    // is the granularity a prefetch ring or a per-layer CPU-compute choice works in.
    let mut merged: Vec<WeightSegment> = Vec::with_capacity(segments.len());
    for s in segments {
        if let Some(last) = merged.last_mut() {
            let end = last.offset + last.bytes;
            if last.tier == WeightTier::Fast
                && s.tier == WeightTier::Fast
                && s.offset >= end
                && s.offset - end <= MERGE_GAP
            {
                last.bytes = s.offset + s.bytes - last.offset;
                continue;
            }
        }
        merged.push(s);
    }
    WeightPlacement {
        segments: merged,
        budget,
        fast_layers,
        total_layers: n_layers,
    }
}

/// The placement for a loaded model: budget from the backend (and the user's overrides),
/// reserve from the plan, fit from the tensor table.
#[must_use]
pub fn plan_placement(
    tensors: &BTreeMap<String, Tensor>,
    plan: &ModelPlan,
    capacity: usize,
    max_batch: usize,
    reported_budget: Option<u64>,
) -> WeightPlacement {
    let total = configured_budget(reported_budget);
    let reserve_kv = kv_reserve_bytes(plan, capacity, max_batch);
    let reserve_activations = activation_reserve_bytes(plan, capacity, max_batch);
    let reserve_scratch = 0;
    let margin = RESERVE_MARGIN_BYTES;
    let weight_budget = total.map_or(u64::MAX, |t| {
        t.saturating_sub(reserve_kv + reserve_activations + reserve_scratch + margin)
    });
    let budget = TierBudget {
        total,
        reserve_kv,
        reserve_activations,
        reserve_scratch,
        margin,
        weight_budget,
    };
    fit(
        tensors,
        plan.config.n_layers,
        plan.weight_residency.row_gathered,
        budget,
        configured_row_gathered_policy(),
    )
}

/// One line at load: the decision and the arithmetic behind it. Always printed -- a model
/// with layers in the slow tier must say so, and a model that fits must be seen to.
pub fn describe(p: &WeightPlacement, capacity: usize, max_batch: usize) -> String {
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let fast = p.bytes_in(|t| t == WeightTier::Fast);
    let slow = p.bytes_in(|t| matches!(t, WeightTier::Slow { .. }));
    let staged = p.bytes_in(|t| t == WeightTier::HostStaged);
    let b = &p.budget;
    let total = b
        .total
        .map_or_else(|| "unknown".to_string(), |t| format!("{:.0} MiB", mib(t)));
    format!(
        "fit: fast tier {total}, reserve {:.0} MiB (kv {:.0} for ctx {capacity}, \
         activations {:.0} at batch {max_batch}, scratch {:.0}, margin {:.0}); \
         fast tier {}/{} layers = {:.0} MiB, slow tier {:.0} MiB, host-staged rows {:.0} MiB",
        mib(b.reserve_kv + b.reserve_activations + b.reserve_scratch + b.margin),
        mib(b.reserve_kv),
        mib(b.reserve_activations),
        mib(b.reserve_scratch),
        mib(b.margin),
        p.fast_layers,
        p.total_layers,
        mib(fast),
        mib(slow),
        mib(staged)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(offset: u64, bytes: u64) -> Tensor {
        Tensor {
            offset: offset as usize,
            bytes: bytes as usize,
            ggml_type: 2,
            ne: [1, 1, 1, 1],
            n_dims: 2,
        }
    }

    /// Four layers of 100 bytes, a 50-byte shared head, a 1000-byte table; the layers'
    /// two tensors each are contiguous, so a layer is one segment.
    fn table() -> BTreeMap<String, Tensor> {
        let mut t = BTreeMap::new();
        t.insert("token_embd.weight".into(), tensor(0, 50));
        t.insert("big.table".into(), tensor(50, 1000));
        for l in 0..4u64 {
            let base = 1050 + l * 100;
            t.insert(format!("blk.{l}.a.weight"), tensor(base, 60));
            t.insert(format!("blk.{l}.b.weight"), tensor(base + 60, 40));
        }
        t
    }

    fn budget(weight_budget: u64) -> TierBudget {
        TierBudget {
            weight_budget,
            ..TierBudget::default()
        }
    }

    #[test]
    fn a_table_takes_fast_room_last_when_it_fits() {
        // budget unknown: layers and the table bind, and the file order makes ONE segment
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(u64::MAX),
            RowGatheredPolicy::BindIfFits,
        );
        assert_eq!(p.fast_layers, 4);
        assert_eq!(p.bytes_in(|t| t == WeightTier::Fast), 1450);
        assert_eq!(p.bytes_in(|t| t == WeightTier::HostStaged), 0);
        assert_eq!(p.segments.len(), 1);
        // layers fit (450) but the table would not (1450 > 1400): staged, layers untouched
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(1400),
            RowGatheredPolicy::BindIfFits,
        );
        assert_eq!(p.fast_layers, 4);
        assert_eq!(p.bytes_in(|t| t == WeightTier::Fast), 450);
        assert_eq!(p.bytes_in(|t| t == WeightTier::HostStaged), 1000);
        // a layer overflowed: the table is staged whatever room is left
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(260),
            RowGatheredPolicy::BindIfFits,
        );
        assert_eq!(p.fast_layers, 2);
        assert_eq!(p.bytes_in(|t| t == WeightTier::HostStaged), 1000);
    }

    #[test]
    fn everything_fits_when_the_tier_is_unknown() {
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(u64::MAX),
            RowGatheredPolicy::HostStaged,
        );
        assert_eq!(p.fast_layers, 4);
        assert_eq!(p.bytes_in(|t| t == WeightTier::Fast), 450);
        assert_eq!(p.bytes_in(|t| t == WeightTier::HostStaged), 1000);
        assert_eq!(p.bytes_in(|t| matches!(t, WeightTier::Slow { .. })), 0);
        // shared Fast, the host-staged rows, then the four layers merged into ONE fast segment
        assert_eq!(p.segments.len(), 3);
        assert_eq!(p.segments[2].bytes, 400);
    }

    #[test]
    fn layers_go_to_the_slow_tier_from_the_end_once_the_budget_runs_out() {
        // shared 50 + two layers 200 = 250 fits in 260; the third layer would make 350
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(260),
            RowGatheredPolicy::HostStaged,
        );
        assert_eq!(p.fast_layers, 2);
        let slow: Vec<u32> = p
            .segments
            .iter()
            .filter_map(|s| match s.tier {
                WeightTier::Slow { layer } => Some(layer),
                _ => None,
            })
            .collect();
        assert_eq!(slow, vec![2, 3]);
        assert_eq!(p.bytes_in(|t| t == WeightTier::Fast), 250);
    }

    #[test]
    fn a_table_never_counts_against_the_budget() {
        // budget below the table's size: the table stays host-staged, the layers still fit
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(450),
            RowGatheredPolicy::HostStaged,
        );
        assert_eq!(p.fast_layers, 4);
        assert_eq!(p.bytes_in(|t| t == WeightTier::HostStaged), 1000);
    }

    #[test]
    fn segments_are_sorted_and_disjoint() {
        let p = fit(
            &table(),
            4,
            &["big.table"],
            budget(260),
            RowGatheredPolicy::HostStaged,
        );
        for w in p.segments.windows(2) {
            assert!(w[0].offset + w[0].bytes <= w[1].offset, "{w:?}");
        }
    }

    #[test]
    fn layer_names_parse() {
        assert_eq!(layer_of("blk.17.ffn_up.weight"), Some(17));
        assert_eq!(layer_of("token_embd.weight"), None);
        assert_eq!(layer_of("blk.x.weight"), None);
    }
}
