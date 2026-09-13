//! Greedy verification on the existing workflow and device state owner.
use crate::kv::KvPoolMember;
use crate::window_history::WindowHistory;
use crate::{Architecture, Model, OutputDemand, Workflow};
use imparo_backend::BufId;

/// Accepted inputs include the anchor; the correction/bonus is not consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GreedyVerification {
    pub consumed: usize,
    pub next_token: u32,
}

fn validate<M: Model + ?Sized>(
    m: &M,
    tokens: &[u32],
    start: usize,
) -> Result<(), String> {
    if tokens.is_empty() {
        return Err("greedy verification requires an anchor".into());
    }
    if start != m.state().kv_rt.filled {
        return Err(format!(
            "verification start {start} differs from filled {}",
            m.state().kv_rt.filled
        ));
    }
    if !m.state().queued.is_empty() || m.state().pipe.is_some() {
        return Err("verification requires a drained decode queue".into());
    }
    if m.state().verification_prefix_tokens != 0 {
        return Err("nested greedy verification is unsupported".into());
    }
    crate::validate_all_logits(m, tokens, start)
}

pub(crate) fn verify_sequential<M: Model + ?Sized>(
    m: &mut M,
    tokens: &[u32],
    start: usize,
) -> Result<GreedyVerification, String> {
    validate(m, tokens, start)?;
    for (i, &token) in tokens.iter().enumerate() {
        let next = m.forward_next(token, start + i)?;
        if i + 1 == tokens.len() || next != tokens[i + 1] {
            return Ok(GreedyVerification {
                consumed: i + 1,
                next_token: next,
            });
        }
    }
    unreachable!("nonempty anchor checked")
}

fn select(
    logits: &[f32],
    tokens: &[u32],
    vocab: usize,
) -> Result<GreedyVerification, String> {
    if logits.len() != tokens.len() * vocab || logits.iter().any(|v| !v.is_finite()) {
        return Err(
            "verification logits have an invalid length or nonfinite value".into(),
        );
    }
    for (i, row) in logits.chunks_exact(vocab).enumerate() {
        let next = imparo_cpu::ops::argmax_f32(row);
        if i + 1 == tokens.len() || next != tokens[i + 1] {
            return Ok(GreedyVerification {
                consumed: i + 1,
                next_token: next,
            });
        }
    }
    Err("verification produced no logits".into())
}

/// Opt-in laboratory witness of the logits already read by the real verifier.
/// It neither changes inputs nor selects a numerical route; disabled builds have
/// no hook. Enabled dumps add I/O and must never be used as timing evidence.
#[cfg(feature = "cuda-owner-lab")]
fn dump_verification_logits(
    logits: &[f32],
    tokens: &[u32],
    start: usize,
    vocab: usize,
) -> Result<(), String> {
    use std::sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    };
    static ROOT: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    static INDEX: AtomicU64 = AtomicU64::new(0);
    let Some(root) = ROOT.get_or_init(|| {
        std::env::var_os("IMPARO_VERIFY_LOGITS_DUMP_DIR").map(Into::into)
    }) else {
        return Ok(());
    };
    let folder = root.join(format!(
        "block-{:06}",
        INDEX.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    let raw: Vec<u8> = logits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(folder.join("logits.raw"), raw).map_err(|e| e.to_string())?;
    let meta = serde_json::json!({"start":start,"input_ids":tokens,"rows":tokens.len(),"vocab":vocab,
        "logit_count":logits.len(),"scope":"actual verifier full logits before accepted-prefix selection; not timing evidence"});
    std::fs::write(
        folder.join("meta.json"),
        serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

/// Explicit, bounded readback of the path that actually ran. It never selects
/// logits mode, disables graphs, or replaces a device-selected token.
#[cfg(feature = "cuda-owner-lab")]
pub(crate) mod target_witness {
    use super::*;
    use std::{path::PathBuf, sync::OnceLock};
    struct Config {
        root: PathBuf,
        start: usize,
        end: usize,
    }
    fn config() -> Result<Option<&'static Config>, String> {
        static CONFIG: OnceLock<Result<Option<Config>, String>> = OnceLock::new();
        CONFIG
            .get_or_init(|| {
                let Some(root) = std::env::var_os("IMPARO_TARGET_WITNESS_DIR")
                    .filter(|v| !v.is_empty())
                else {
                    return Ok(None);
                };
                let parse = |key| {
                    std::env::var(key)
                        .map_err(|e| format!("{key}: {e}"))?
                        .parse::<usize>()
                        .map_err(|e| format!("{key}: {e}"))
                };
                let start = parse("IMPARO_TARGET_WITNESS_START")?;
                let rows = parse("IMPARO_TARGET_WITNESS_ROWS")?;
                if !(1..=2).contains(&rows) {
                    return Err("target witness requires 1 or 2 rows".into());
                }
                let end = start
                    .checked_add(rows)
                    .ok_or("target witness range overflow")?;
                Ok(Some(Config {
                    root: root.into(),
                    start,
                    end,
                }))
            })
            .as_ref()
            .map(|v| v.as_ref())
            .map_err(Clone::clone)
    }
    fn save(
        cfg: &Config,
        route: &str,
        pos: usize,
        logits: &[f32],
        recurrent: &[f32],
        mut meta: serde_json::Value,
    ) -> Result<(), String> {
        if logits.is_empty() || logits.iter().chain(recurrent).any(|v| !v.is_finite()) {
            return Err("target witness has empty logits or nonfinite values".into());
        }
        let cpu_pick = imparo_cpu::ops::argmax_f32(logits);
        meta["cpu_argmax"] = cpu_pick.into();
        meta["cpu_argmax_logit"] = logits[cpu_pick as usize].into();
        meta["absolute_position"] = pos.into();
        meta["diagnostic_only"] = true.into();
        meta["selection_replaced"] = false.into();
        let folder = cfg.root.join(route).join(format!("pos-{pos}"));
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        let raw =
            |x: &[f32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        std::fs::write(folder.join("logits.raw"), raw(logits))
            .map_err(|e| e.to_string())?;
        std::fs::write(folder.join("recurrent.raw"), raw(recurrent))
            .map_err(|e| e.to_string())?;
        std::fs::write(
            folder.join("meta.json"),
            serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
    pub(crate) fn m1<A: Architecture>(
        w: &Workflow<A>,
        token: u32,
        start: usize,
        pick: &[f32],
    ) -> Result<(), String> {
        let Some(cfg) = config()? else {
            return Ok(());
        };
        if start < cfg.start || start >= cfg.end {
            return Ok(());
        }
        if pick.len() != 1 || pick[0].to_bits() >= w.plan.config.vocab_size {
            return Err("target M1 witness invalid actual pick".into());
        }
        let (_, plane_out) = w.state.recur_decode_planes();
        if plane_out >= w.state.recur_planes {
            return Err("target witness plane out of range".into());
        }
        let n = w.plan.recurrent_elems() as usize;
        let mut logits = vec![0.0; w.plan.config.vocab_size as usize];
        let mut recurrent = vec![0.0; n];
        let be = crate::gpu_support::be();
        be.read(BufId::Logits, 0, &mut logits);
        if n != 0 {
            be.read(
                BufId::Recur,
                (n as u64) * u64::from(plane_out),
                &mut recurrent,
            );
        }
        be.end()
            .map_err(|rc| format!("target M1 witness readback rc={rc}"))?;
        save(
            cfg,
            "m1",
            start,
            &logits,
            &recurrent,
            serde_json::json!({
                "route":"actual M1, including any graph replay","input_tokens":[token],
                "physical_rows":1,"logit_row":0,"actual_selected_id":pick[0].to_bits(),
                "actual_selected_source":"existing device argmax result",
                "recurrent_source":"live Recur after actual M1 before cursor commit",
                "recurrent_plane":plane_out,"recurrent_elements":n,"vocab":logits.len()
            }),
        )
    }
    pub(crate) fn device_packet<A: Architecture>(
        w: &Workflow<A>,
        tokens: &[u32],
        start: usize,
        packet: &[f32],
    ) -> Result<(), String> {
        let Some(cfg) = config()? else {
            return Ok(());
        };
        let end = start
            .checked_add(tokens.len())
            .ok_or("target witness block overflow")?;
        if end <= cfg.start || start >= cfg.end {
            return Ok(());
        }
        if packet.len() != 3 {
            return Err("target witness invalid device packet".into());
        }
        let packet: Vec<u32> = packet.iter().map(|x| x.to_bits()).collect();
        let consumed = packet[1] as usize;
        if packet[0] != 0 || consumed == 0 || consumed > tokens.len() {
            return Err("target witness invalid consumed count".into());
        }
        let n = w.plan.recurrent_elems() as usize;
        let vocab = w.plan.config.vocab_size as usize;
        let be = crate::gpu_support::be();
        for pos in start.max(cfg.start)..end.min(cfg.end) {
            let row = pos - start;
            // Snapshot k is stored in slot k+1. The prefix after input row r
            // therefore lives at (r+2)*n; the final full prefix has no extra slot.
            let slot = row
                .checked_add(2)
                .ok_or("target witness snapshot overflow")?;
            if n != 0
                && (row + 1 >= tokens.len() || slot >= w.state.gpu_recur_snap_slots)
            {
                return Err(
                    "target witness requested an unavailable prefix snapshot".into()
                );
            }
            let actual_selected = if row + 1 < consumed {
                Some(tokens[row + 1])
            } else if row + 1 == consumed {
                Some(packet[2])
            } else {
                None
            };
            let mut logits = vec![0.0; vocab];
            let mut recurrent = vec![0.0; n];
            be.read(BufId::Logits, (row as u64) * (vocab as u64), &mut logits);
            if n != 0 {
                be.read(BufId::RecurSnap, (slot as u64) * (n as u64), &mut recurrent);
            }
            be.end()
                .map_err(|rc| format!("target M9 witness readback rc={rc}"))?;
            save(
                cfg,
                "device-verify",
                pos,
                &logits,
                &recurrent,
                serde_json::json!({
                    "route":"actual device greedy verification packet","input_tokens":tokens,
                    "block_start":start,"physical_rows":tokens.len(),"logit_row":row,
                    "packet_u32":packet,"actual_selected_id":actual_selected,
                    "actual_selected_source":"accepted-input identity or correction from existing packet; null beyond consumed prefix",
                    "recurrent_source":"existing RecurSnap prefix","recurrent_slot":slot,
                    "recurrent_elements":n,"vocab":vocab
                }),
            )?;
        }
        Ok(())
    }
}

fn checkpoint_words(note: &[u8]) -> Vec<f32> {
    note.chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}

fn rollback<A: Architecture>(
    w: &mut Workflow<A>,
    start: usize,
    n: u32,
    history: WindowHistory,
    note: Vec<u8>,
    note_at: usize,
    original: &str,
) -> String {
    let be = crate::gpu_support::be();
    let pending = be.end().err();
    be.begin();
    if n != 0 {
        be.copy_range(BufId::Recur, 0, BufId::RecurSnap, n, n);
        if !note.is_empty() {
            be.write(BufId::RecurSnap, 0, &checkpoint_words(&note));
        }
    }
    let restore = be.end().err();
    w.state.kv_rt.filled = start;
    w.state.window_history = history;
    w.state.recur_ckpt = note;
    w.state.recur_ckpt_at = note_at;
    w.state.recur_ckpt_on_device = false;
    w.state.recur_snap = None;
    w.state.verification_prefix_tokens = 0;
    w.state.output_demand = OutputDemand::LastToken;
    format!(
        "{original}; verification rollback drain={pending:?}, restore={restore:?}{}",
        if restore.is_some() {
            "; device state recovery is unconfirmed"
        } else {
            ""
        }
    )
}

/// Materialize a deferred v2 checkpoint before the verifier reuses RecurSnap.
fn recurrent_note<A: Architecture>(w: &mut Workflow<A>) -> (usize, Vec<u8>) {
    let (at, note) = KvPoolMember::kv_recurrent_note(w)
        .unwrap_or((w.state.recur_ckpt_at, Vec::new()));
    w.state.recur_ckpt.clone_from(&note);
    w.state.recur_ckpt_on_device = false;
    (at, note)
}

/// A drained verifier owns the current state exclusively. Canonicalize its live
/// recurrent plane before the existing packed-prefix transaction uses buffer offset 0.
fn prepare_recurrent_plane<A: Architecture>(w: &mut Workflow<A>) -> Result<(), String> {
    let n = w.plan.recurrent_elems();
    let plane = w.state.recur_plane;
    if plane != w.state.recur_plane_next {
        return Err("verification recurrent cursor is not retired".into());
    }
    if n != 0 && plane != 0 {
        let off = n
            .checked_mul(plane)
            .ok_or("verification recurrent plane overflow")?;
        let be = crate::gpu_support::be();
        be.begin();
        be.copy_range(BufId::Recur, 0, BufId::Recur, off, n);
        be.end()
            .map_err(|rc| format!("verification recurrent plane copy rc={rc}"))?;
    }
    w.state.recur_plane = 0;
    w.state.recur_plane_next = 0;
    Ok(())
}

pub(crate) fn verify_workflow<A: Architecture>(
    w: &mut Workflow<A>,
    tokens: &[u32],
    start: usize,
) -> Result<GreedyVerification, String> {
    verify_workflow_limited(w, tokens, start, tokens.len())
}

pub(crate) fn verify_workflow_limited<A: Architecture>(
    w: &mut Workflow<A>,
    tokens: &[u32],
    start: usize,
    max_consumed: usize,
) -> Result<GreedyVerification, String> {
    if max_consumed == 0 || max_consumed > tokens.len() {
        return Err("invalid verification commit limit".into());
    }
    validate(w, tokens, start)?;
    let b = tokens.len();
    let end = start + b; // checked by validate_all_logits
    let cell = crate::prefill_batch().max(1);
    if !A::DEVICE_PREFIX_VERIFICATION
        || !A::DEVICE_ALL_LOGITS
        || w.state.host_forward
        || b == 1
        || b > cell - start % cell
    {
        return verify_sequential(w, &tokens[..max_consumed], start);
    }
    if end > u32::MAX as usize {
        return Err("verification position exceeds device u32 range".into());
    }
    let ring_batch = if w.state.kv_ring_batch == 0 {
        cell
    } else {
        w.state.kv_ring_batch
    };
    let geom = crate::kv::state_geometry(&w.plan, ring_batch);
    if geom.iter().any(|g| matches!(g.kind, imparo_kv::StateKind::Window { window, ring } if ring < window)) {
        return Err("verification window ring is smaller than its window".into());
    }
    let slack = imparo_kv::state::window_slack(&geom);
    // The transaction rolls back to start, never to an older cursor prefix.
    // Narrow old advertised coverage only after proving the real start survives.
    let Some(history) = w.state.window_history.for_verification(start, end, slack)
    else {
        return verify_sequential(w, &tokens[..max_consumed], start);
    };
    let n = w.plan.recurrent_elems();
    let slots = b
        .checked_add(1)
        .ok_or("verification snapshot slot overflow")?;
    let elems = slots
        .checked_mul(n as usize)
        .ok_or("verification snapshot size overflow")?;
    if elems > u32::MAX as usize {
        return Err("verification snapshot exceeds device u32 range".into());
    }
    let bytes = (elems as u64)
        .checked_mul(4)
        .ok_or("verification snapshot byte overflow")?;
    if !w.state.recur_ckpt.is_empty() && w.state.recur_ckpt.len() != n as usize * 4 {
        return Err("verification recurrent checkpoint has an invalid length".into());
    }
    let ticket = history.clone().begin(start)?; // validate without changing live history
    let (note_at, note) = recurrent_note(w);
    w.ensure_gpu_ready()?;
    w.kv_fit(end)?;
    prepare_recurrent_plane(w)?;
    let be = crate::gpu_support::be();
    let grow = n != 0 && w.state.gpu_recur_snap_slots < slots;
    if grow {
        be.alloc(BufId::RecurSnap, bytes)
            .map_err(|rc| format!("verification snapshot allocation rc={rc}"))?;
        w.state.gpu_recur_snap_slots = slots;
    }
    be.begin();
    if n != 0 {
        if grow && !note.is_empty() {
            be.write(BufId::RecurSnap, 0, &checkpoint_words(&note));
        }
        be.copy_range(BufId::RecurSnap, n, BufId::Recur, 0, n);
    }
    be.end()
        .map_err(|rc| format!("verification initial state copy rc={rc}"))?;
    w.state.window_history = history.clone();
    w.state.verification_prefix_tokens = b;
    let mut logits = Vec::new();
    // The existing device packet selects the whole physical block. Only a
    // bounded terminal block reads its logits; normal blocks keep GPU return.
    let device_return = max_consumed == b && cfg!(feature = "cuda-speculative")
        && A::DEVICE_GREEDY_VERIFICATION
        && be.supports_greedy_verification()
        && std::env::var("IMPARO_LAB_DEVICE_GREEDY_VERIFY").as_deref() == Ok("1")
        // The witness promises complete logits, never a compressed return packet.
        && std::env::var_os("IMPARO_VERIFY_LOGITS_DUMP_DIR").is_none();
    #[cfg(feature = "cuda-owner-lab")]
    let quant_scope =
        unsafe { imparo_cuda::owner_lab::VerificationM1Quant::configured() };
    let forward = if device_return {
        w.forward_with_output_demand(
            tokens,
            start,
            &mut logits,
            None,
            OutputDemand::GreedyVerification,
            None,
        )
    } else {
        w.forward_all_logits_into(tokens, start, &mut logits)
    };
    #[cfg(feature = "cuda-owner-lab")]
    drop(quant_scope);
    w.state.verification_prefix_tokens = 0;
    w.state.output_demand = OutputDemand::LastToken;
    let result = forward.and_then(|()| {
        be.end().map_err(|rc| format!("verification output completion rc={rc}"))?;
        if device_return {
            if logits.len() != 3 { return Err("invalid device verification packet length".into()); }
            let status = logits[0].to_bits();
            let consumed = logits[1].to_bits() as usize;
            let next_token = logits[2].to_bits();
            if status != 0 || consumed == 0 || consumed > b
                || next_token >= w.plan.config.vocab_size {
                return Err(format!("invalid device verification result status={status} consumed={consumed} next={next_token}"));
            }
            #[cfg(feature = "cuda-owner-lab")]
            target_witness::device_packet(w, tokens, start, &logits)?;
            Ok(GreedyVerification { consumed, next_token })
        } else {
            #[cfg(feature = "cuda-owner-lab")]
            dump_verification_logits(&logits, tokens, start, w.plan.config.vocab_size as usize)?;
            {
                let vocab = w.plan.config.vocab_size as usize;
                if logits.len() != b * vocab { return Err("invalid physical verification logits".into()); }
                select(&logits[..max_consumed * vocab], &tokens[..max_consumed], vocab)
            }
        }
    });
    let accepted = match result {
        Ok(r) => r,
        Err(e) => return Err(rollback(w, start, n, history, note, note_at, &e)),
    };
    if !device_return && accepted.consumed < b && n != 0 {
        be.begin();
        let offset = ((accepted.consumed + 1) * n as usize) as u32;
        be.copy_range(BufId::Recur, 0, BufId::RecurSnap, offset, n);
        if let Err(rc) = be.end() {
            return Err(rollback(
                w,
                start,
                n,
                history,
                note,
                note_at,
                &format!("verification prefix restore rc={rc}"),
            ));
        }
    }
    let committed = start + accepted.consumed;
    w.state.window_history = history;
    w.state
        .window_history
        .commit(ticket, start, committed, slack);
    w.state.kv_rt.filled = committed;
    w.state.recur_snap = None;
    if w.state.recur_ckpt_at > committed {
        w.state.recur_ckpt = note;
        w.state.recur_ckpt_at = note_at;
        w.state.recur_ckpt_on_device = false;
    }
    #[cfg(feature = "cuda-owner-lab")]
    if !device_return
        && std::env::var("IMPARO_DEVICE_GREEDY_VERIFY_TRACE").as_deref() == Ok("1")
    {
        static LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[workflow-greedy-verify] route=batch rows={b} start={start} return=host_logits consumed={}",
                accepted.consumed
            );
        }
    }
    Ok(accepted)
}

/// Experimental tree adapter reuses the ordinary workflow, history ticket and
/// recurrent rollback slot. CUDA owns packed-row state until path publication.
#[cfg(feature = "cuda-speculative")]
fn validate_tree_parents(parents: &[i32]) -> Result<(), String> {
    if parents.len() != 16 {
        return Err("tree parent shape".into());
    }
    let mut depths = [0usize; 16];
    for i in 0..16 {
        if (i == 0 && parents[i] != -1)
            || (i > 0 && (parents[i] < 0 || parents[i] as usize >= i))
        {
            return Err("tree parent order".into());
        }
        if i > 0 {
            depths[i] = depths[parents[i] as usize] + 1;
        }
        if depths[i] > 8 {
            return Err("tree depth exceeds trained draft".into());
        }
    }
    Ok(())
}
#[cfg(feature = "cuda-speculative")]
pub(crate) fn prepare_workflow_tree<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
) -> Result<bool, String> {
    if w.plan.config.architecture != "lfm2"
        || w.state.host_forward
        || !A::DEVICE_ALL_LOGITS
        || w.plan.config.n_heads != 32
        || w.plan.config.n_kv_heads != 8
        || tree.tokens.len() != 16
        || tree.parents.len() != 16
    {
        return Err("tree admission capability/shape".into());
    }
    validate_tree_parents(&tree.parents)?;
    validate(w, &tree.tokens, start)?;
    let end = start
        .checked_add(16)
        .ok_or("tree admission position overflow")?;
    let cell = crate::prefill_batch().max(1);
    if end > u32::MAX as usize || 16 > cell - start % cell {
        return Err("tree admission forward cell".into());
    }
    if w.plan.layers.iter().any(|l| {
        l.attention.is_attention()
            && (!matches!(l.attention, crate::Attention::Full { .. })
                || l.attention.head_dim() != 64)
    }) {
        return Err("tree admission attention geometry".into());
    }
    let recurrent = w.plan.recurrent_elems();
    if recurrent == 0 {
        return Err("tree admission recurrent state".into());
    }
    // Do not grow or write target KV just to attempt a tree. A declined tree
    // keeps the generated chain, whose ordinary verifier owns any later growth.
    if end > w.state.kv_rt.slots {
        return Ok(false);
    }
    w.ensure_gpu_ready()?;
    unsafe { imparo_cuda::tree::prepare(start as u32, &tree.parents, recurrent) }
}

#[cfg(feature = "cuda-speculative")]
pub(crate) fn verify_workflow_tree<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
    limit: usize,
    stops: &[u32],
) -> Result<crate::speculative::TreeVerification, String> {
    let tokens = &tree.tokens;
    let parents = &tree.parents;
    let b = tokens.len();
    if w.plan.config.architecture != "lfm2"
        || w.state.host_forward
        || !A::DEVICE_ALL_LOGITS
        || b != 16
        || parents.len() != b
        || limit == 0
        || limit > 9
    {
        return Err("tree target capability/shape".into());
    }
    validate_tree_parents(parents)?;
    validate(w, tokens, start)?;
    let end = start + b;
    let cell = crate::prefill_batch().max(1);
    if b > cell - start % cell || end > u32::MAX as usize {
        return Err("tree crosses supported forward cell".into());
    }
    let ring_batch = if w.state.kv_ring_batch == 0 {
        cell
    } else {
        w.state.kv_ring_batch
    };
    let geom = crate::kv::state_geometry(&w.plan, ring_batch);
    if geom
        .iter()
        .any(|g| matches!(g.kind, imparo_kv::StateKind::Window { .. }))
    {
        return Err("tree window adapter not qualified".into());
    }
    let slack = imparo_kv::state::window_slack(&geom);
    let history = w
        .state
        .window_history
        .for_verification(start, end, slack)
        .ok_or("tree prefix coverage")?;
    let ticket = history.clone().begin(start)?;
    let n = w.plan.recurrent_elems();
    let (note_at, note) = recurrent_note(w);
    if n == 0 || (!note.is_empty() && note.len() != n as usize * 4) {
        return Err("tree recurrent checkpoint contract".into());
    }
    w.ensure_gpu_ready()?;
    w.kv_fit(end)?;
    prepare_recurrent_plane(w)?;
    let be = crate::gpu_support::be();
    if w.state.gpu_recur_snap_slots < 2 {
        be.alloc(BufId::RecurSnap, u64::from(n) * 8)
            .map_err(|rc| format!("tree rollback allocation rc={rc}"))?;
        w.state.gpu_recur_snap_slots = 2;
    }
    be.begin();
    if !note.is_empty() {
        be.write(BufId::RecurSnap, 0, &checkpoint_words(&note));
    }
    be.copy_range(BufId::RecurSnap, n, BufId::Recur, 0, n);
    be.end()
        .map_err(|rc| format!("tree initial state copy rc={rc}"))?;
    unsafe {
        imparo_cuda::tree::begin(start as u32, parents, n)?;
    }
    w.state.window_history = history.clone();
    let mut rows = Vec::new();
    let result = w
        .forward_with_output_demand(
            tokens,
            start,
            &mut rows,
            None,
            OutputDemand::RowArgmax,
            None,
        )
        .and_then(|()| {
            be.end()
                .map_err(|rc| format!("tree output completion rc={rc}"))?;
            if rows.len() != b {
                return Err("tree row argmax packet length".into());
            }
            let picks: Vec<u32> = rows.iter().map(|v| v.to_bits()).collect();
            if picks.iter().any(|&v| v >= w.plan.config.vocab_size) {
                return Err("tree argmax outside vocabulary".into());
            }
            let mut path = vec![0i32];
            let next = loop {
                let node = *path.last().unwrap() as usize;
                let next = picks[node];
                if path.len() == limit || stops.contains(&next) {
                    break next;
                }
                if let Some(child) = (node + 1..b)
                    .find(|&j| parents[j] == node as i32 && tokens[j] == next)
                {
                    path.push(child as i32);
                } else {
                    break next;
                }
            };
            unsafe {
                imparo_cuda::tree::commit(&path)?;
            }
            // Any ordinary in-batch checkpoint was calculated for packed rows; retain
            // the previous valid boundary, never publish that linear snapshot.
            be.begin();
            if !note.is_empty() {
                be.write(BufId::RecurSnap, 0, &checkpoint_words(&note));
            }
            be.end()
                .map_err(|rc| format!("tree checkpoint restore rc={rc}"))?;
            Ok(crate::speculative::TreeVerification {
                path,
                next_token: next,
            })
        });
    w.state.output_demand = OutputDemand::LastToken;
    let accepted = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = be.end();
            let _ = unsafe { imparo_cuda::tree::end() };
            return Err(rollback(w, start, n, history, note, note_at, &e));
        }
    };
    let committed = start + accepted.path.len();
    w.state.window_history = history;
    w.state
        .window_history
        .commit(ticket, start, committed, slack);
    w.state.kv_rt.filled = committed;
    w.state.recur_snap = None;
    w.state.recur_ckpt = note;
    w.state.recur_ckpt_at = note_at;
    w.state.recur_ckpt_on_device = false;
    if std::env::var("IMPARO_LAB_DRAFT_ACCEPTANCE_TRACE").as_deref() == Ok("1") {
        eprintln!(
            "[tree-verify] start={start} rows=16 accepted={} alternate={} path={:?}",
            accepted.path.len(),
            accepted.path.iter().any(|&i| i >= 9),
            accepted.path
        );
    }
    Ok(accepted)
}
