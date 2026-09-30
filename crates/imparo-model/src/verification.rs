//! Greedy verification on the existing workflow and device state owner.
use crate::kv::{KvPoolMember, KvType};
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

/// `select` over device picks: `picks[i]` holds row i's argmax as u32 bits. The device pick
/// cannot check that the logits are finite; `select` can.
fn select_picks(
    picks: &[f32],
    tokens: &[u32],
    vocab: u32,
) -> Result<GreedyVerification, String> {
    if picks.len() != tokens.len() {
        return Err("verification row picks have an invalid length".into());
    }
    for (i, pick) in picks.iter().enumerate() {
        let next = pick.to_bits();
        if next >= vocab {
            return Err(format!(
                "verification row {i} picked {next}, outside the vocabulary"
            ));
        }
        if i + 1 == tokens.len() || next != tokens[i + 1] {
            return Ok(GreedyVerification {
                consumed: i + 1,
                next_token: next,
            });
        }
    }
    Err("verification produced no row picks".into())
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
    use super::{Architecture, BufId, Workflow};
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
            let actual_selected = match (row + 1).cmp(&consumed) {
                std::cmp::Ordering::Less => Some(tokens[row + 1]),
                std::cmp::Ordering::Equal => Some(packet[2]),
                std::cmp::Ordering::Greater => None,
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

/// IMPARO_VERIFY_LOG=1 names the path each block verification took and what it
/// committed. Read once; off, it costs one branch per verification.
fn verify_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_VERIFY_LOG").as_deref() == Ok("1"))
}

/// IMPARO_TREE_FLOAT_Q=1: a tree verify's row-layout attention keeps Q in float instead of
/// rounding it to half. Off by default. The causal entry keeps rounding, so with it on a chain
/// laid out as a tree is no longer bit-equal to the causal forward.
pub(crate) fn tree_float_q() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_TREE_FLOAT_Q").as_deref() == Ok("1"))
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
        if verify_log() {
            eprintln!(
                "[verify] path=sequential b={b} start={start} device_flags={} host={} crosses_cell={}",
                A::DEVICE_PREFIX_VERIFICATION && A::DEVICE_ALL_LOGITS,
                w.state.host_forward,
                b > cell - start % cell
            );
        }
        return verify_sequential(w, &tokens[..max_consumed], start);
    }
    if end > u32::MAX as usize {
        return Err("verification position exceeds device u32 range".into());
    }
    let geom = crate::kv::state_geometry(&w.plan, ring_batch(w));
    if geom.iter().any(|g| matches!(g.kind, imparo_kv::StateKind::Window { window, ring } if ring < window)) {
        return Err("verification window ring is smaller than its window".into());
    }
    let slack = imparo_kv::state::window_slack(&geom);
    // The transaction rolls back to start, never to an older cursor prefix.
    // Narrow old advertised coverage only after proving the real start survives.
    let Some(history) = w.state.window_history.for_verification(start, end, slack)
    else {
        if verify_log() {
            eprintln!(
                "[verify] path=sequential b={b} start={start} reason=window_history"
            );
        }
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
        && (crate::e4b_retained_decode_policy_enabled()
            || crate::lfm_retained_domain() != 0
            || std::env::var("IMPARO_LAB_DEVICE_GREEDY_VERIFY").as_deref() == Ok("1"))
        // The witness promises complete logits, never a compressed return packet.
        && std::env::var_os("IMPARO_VERIFY_LOGITS_DUMP_DIR").is_none();
    #[cfg(feature = "cuda-owner-lab")]
    let quant_scope =
        unsafe { imparo_cuda::owner_lab::VerificationM1Quant::configured() };
    // One pick per row on the device when the backend serves it: the host reads b indices,
    // not b rows of logits. A logits dump needs the logits themselves.
    let row_picks = !device_return
        && be.supports_argmax_rows()
        && std::env::var_os("IMPARO_VERIFY_LOGITS_DUMP_DIR").is_none();
    let forward = if device_return {
        w.forward_with_output_demand(
            tokens,
            start,
            &mut logits,
            None,
            OutputDemand::GreedyVerification,
            None,
        )
    } else if row_picks {
        w.forward_with_output_demand(
            tokens,
            start,
            &mut logits,
            None,
            OutputDemand::RowArgmax,
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
        } else if row_picks {
            if logits.len() != b {
                return Err("invalid verification row pick count".into());
            }
            select_picks(&logits[..max_consumed], &tokens[..max_consumed], w.plan.config.vocab_size)
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
    if verify_log() {
        eprintln!(
            "[verify] path=device b={b} start={start} out={} consumed={} next={}",
            if device_return {
                "packet"
            } else if row_picks {
                "row_picks"
            } else {
                "logits"
            },
            accepted.consumed,
            accepted.next_token
        );
    }
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
    // THE GRID BOUNDARY THIS BLOCK CROSSED, if any: its state becomes the recurrent note, as a
    // one-token decode step's does (`arm_recurrent_snapshot`). The forward already wrote it: slot
    // j + 1 of RecurSnap holds the state after the block's first j tokens, and the block's last
    // token leaves it live. Read now, before the next verify reuses the slots.
    let grid = imparo_kv::grid_tokens();
    let boundary = committed / grid * grid;
    if !device_return && n != 0 && boundary > start {
        let below = boundary - start;
        let slot = if below == b {
            (BufId::Recur, 0)
        } else {
            (
                BufId::RecurSnap,
                u32::try_from(below + 1).map_err(|_| "verification slot overflow")?,
            )
        };
        w.state.recur_ckpt = crate::kv::read_recurrent(n as usize, slot.0, slot.1);
        w.state.recur_ckpt_at = boundary;
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

/// `BufId::RowLayout` words for a tree whose node i has parent `parents[i]` (-1 for the
/// root, which continues the committed prefix): each node sits at `start` plus its depth,
/// sees its ancestors and itself, and carries its first eight ancestors for the short
/// convolution.
///
/// # Errors
/// A tree of no nodes or of more than a layout holds, or parents out of order: the root first
/// and every other parent before its child.
pub fn tree_row_layout(start: usize, parents: &[i32]) -> Result<Vec<u32>, String> {
    use imparo_backend::{ROW_LAYOUT_ANCESTORS, ROW_LAYOUT_MAX_ROWS, ROW_LAYOUT_WORDS};
    let n = parents.len();
    if n == 0 || n > ROW_LAYOUT_MAX_ROWS {
        return Err(format!(
            "a tree of {n} nodes; a row layout holds 1 to {ROW_LAYOUT_MAX_ROWS}"
        ));
    }
    let mut depth = vec![0_u32; n];
    let mut seen = vec![0_u64; n];
    let mut up = vec![[0_u32; ROW_LAYOUT_ANCESTORS]; n];
    let mut words = vec![0_u32; n * ROW_LAYOUT_WORDS];
    for (i, &raw_parent) in parents.iter().enumerate() {
        let parent = usize::try_from(raw_parent).ok();
        if (i == 0) != parent.is_none() || parent.is_some_and(|p| p >= i) {
            return Err(
                "tree parents: the root first, every other parent before its child"
                    .into(),
            );
        }
        seen[i] = 1_u64 << i;
        if let Some(p) = parent {
            depth[i] = depth[p] + 1;
            seen[i] |= seen[p];
            let parent_up = up[p];
            up[i][0] = p as u32;
            up[i][1..].copy_from_slice(&parent_up[..ROW_LAYOUT_ANCESTORS - 1]);
        }
        let position = u32::try_from(start + depth[i] as usize)
            .map_err(|_| "tree position exceeds u32")?;
        let row = &mut words[i * ROW_LAYOUT_WORDS..(i + 1) * ROW_LAYOUT_WORDS];
        row[0] = position;
        row[1] = depth[i];
        row[2] = seen[i] as u32;
        row[3] = (seen[i] >> 32) as u32;
        row[4..4 + ROW_LAYOUT_ANCESTORS].copy_from_slice(&up[i]);
    }
    Ok(words)
}

/// Where a tree verify keeps each node's input to every convolution window: one row of
/// `row_elems` values per node in `BufId::RowInputs`, the windows side by side in layer order.
pub(crate) struct RowInputLayout {
    windows: Vec<RowWindow>,
    /// Values per node.
    pub(crate) row_elems: u32,
    /// Recurrent state values the windows cover, over every layer.
    state_elems: u32,
}

/// One convolution window: its layer, where its state starts in `BufId::Recur`, its shape, and
/// where its inputs start in a node's row.
struct RowWindow {
    layer: usize,
    state_off: u32,
    width: u32,
    history: u32,
    input_off: u32,
}

impl RowInputLayout {
    /// The layout of `windows` over `plan`. Refuses a window outside a recurrent layer or larger
    /// than the layer's rolling state.
    pub(crate) fn of(
        plan: &crate::ModelPlan,
        windows: &[crate::ConvWindow],
    ) -> Result<Self, String> {
        let regions = plan.recurrent_layout();
        let mut out = Vec::with_capacity(windows.len());
        let (mut row_elems, mut state_elems) = (0_u32, 0_u32);
        for win in windows {
            let layer = win.layer as usize;
            let &(state_off, _, r_elems, _) = regions.get(layer).ok_or_else(|| {
                format!("a convolution window on layer {layer}, past the plan")
            })?;
            let elems = win
                .width
                .checked_mul(win.history)
                .filter(|&e| e != 0 && e <= r_elems)
                .ok_or_else(|| {
                    format!("layer {layer}'s convolution window does not fit its rolling state")
                })?;
            out.push(RowWindow {
                layer,
                state_off,
                width: win.width,
                history: win.history,
                input_off: row_elems,
            });
            row_elems = row_elems
                .checked_add(win.width)
                .ok_or("row inputs overflow u32")?;
            state_elems = state_elems
                .checked_add(elems)
                .ok_or("convolution windows overflow u32")?;
        }
        Ok(Self {
            windows: out,
            row_elems,
            state_elems,
        })
    }

    /// Where layer `layer`'s inputs start in a node's row, if the layer has a window.
    pub(crate) fn input_off(&self, layer: usize) -> Option<u32> {
        self.windows
            .iter()
            .find(|w| w.layer == layer)
            .map(|w| w.input_off)
    }
}

/// Where slot `s` (oldest first) of a rebuilt window comes from after an accepted path: the value
/// `history - 1 - s` steps up from the path's last node -- a path node's kept input while the path
/// reaches that far, the old window's slot beyond it. The same value the tree convolution read
/// for that tap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowSource {
    /// The input kept for the node at this index of the path.
    PathNode(usize),
    /// This slot of the window before the verify.
    OldSlot(usize),
}

fn window_sources(path_len: usize, history: usize) -> Vec<WindowSource> {
    let depth = path_len.saturating_sub(1);
    (0..history)
        .map(|s| {
            let back = history - 1 - s;
            if back <= depth {
                WindowSource::PathNode(depth - back)
            } else {
                WindowSource::OldSlot(history + depth - back)
            }
        })
        .collect()
}

/// Rebuilds every convolution window of the live recurrent state (plane 0) for the accepted path,
/// from the nodes' kept inputs and the window before the verify. Host copies over shared buffers:
/// call with the device idle.
///
/// `below_boundary`: when the path crossed a grid boundary, how many of its nodes lie below it.
/// The whole recurrent state at that boundary is then built the same way from those nodes and
/// returned (a tree runs only where the windows are all of the state). The checkpoint at the
/// turn's end needs it -- the boundary a later request resumes on -- and nothing stands on that
/// position again once the path has run past it.
fn commit_conv_windows(
    layout: &RowInputLayout,
    path: &[i32],
    below_boundary: Option<usize>,
    state_elems: usize,
) -> Result<Option<Vec<u8>>, String> {
    let be = crate::gpu_support::be();
    let rows = path
        .iter()
        .map(|&node| u64::try_from(node).map_err(|_| format!("tree path node {node}")))
        .collect::<Result<Vec<u64>, String>>()?;
    let mut at_boundary = below_boundary.map(|_| vec![0.0_f32; state_elems]);
    for win in &layout.windows {
        let (width, history) = (win.width as usize, win.history as usize);
        let mut old = vec![0.0_f32; width * history];
        be.read(BufId::Recur, u64::from(win.state_off), &mut old);
        // The window after the path's first `len` nodes.
        let window_after = |len: usize, window: &mut [f32]| {
            for (slot, source) in window_sources(len, history).into_iter().enumerate() {
                let dst = &mut window[slot * width..(slot + 1) * width];
                match source {
                    WindowSource::PathNode(i) => be.read(
                        BufId::RowInputs,
                        rows[i] * u64::from(layout.row_elems)
                            + u64::from(win.input_off),
                        dst,
                    ),
                    WindowSource::OldSlot(s) => {
                        dst.copy_from_slice(&old[s * width..(s + 1) * width]);
                    }
                }
            }
        };
        if let (Some(k), Some(state)) = (below_boundary, at_boundary.as_mut()) {
            let off = win.state_off as usize;
            let slot = state
                .get_mut(off..off + width * history)
                .ok_or("a convolution window lies past the recurrent state")?;
            window_after(k, slot);
        }
        let mut window = vec![0.0_f32; width * history];
        window_after(rows.len(), &mut window);
        be.write(BufId::Recur, u64::from(win.state_off), &window);
    }
    Ok(at_boundary.map(|state| state.iter().flat_map(|v| v.to_ne_bytes()).collect()))
}

/// The ring batch a verify's state geometry is built with: the configured one, or the prefill
/// cell.
fn ring_batch<A: Architecture>(w: &Workflow<A>) -> usize {
    if w.state.kv_ring_batch == 0 {
        crate::prefill_batch().max(1)
    } else {
        w.state.kv_ring_batch
    }
}

/// Whether this model and backend run a draft tree through a row layout: a device forward that
/// reads the layout, one pick per row, an f16 cache, no windowed layer, and a row layout at every
/// attention head dim. Brings the device up; changes no conversation state.
fn tree_rows_served<A: Architecture>(w: &mut Workflow<A>) -> Result<bool, String> {
    if !A::ROW_LAYOUT_FORWARD
        || !A::DEVICE_ALL_LOGITS
        || w.state.host_forward
        || KvType::k() != KvType::F16
        || KvType::v() != KvType::F16
    {
        return Ok(false);
    }
    w.ensure_gpu_ready()?;
    let be = crate::gpu_support::be();
    if !be.supports_argmax_rows() {
        return Ok(false);
    }
    let windowed = crate::kv::state_geometry(&w.plan, ring_batch(w))
        .iter()
        .any(|g| matches!(g.kind, imparo_kv::StateKind::Window { .. }));
    // A commit rebuilds convolution windows only, so a tree runs only where they are the whole
    // recurrent state.
    let windows_cover = RowInputLayout::of(&w.plan, &A::conv_windows(&w.plan))
        .is_ok_and(|layout| layout.state_elems == w.plan.recurrent_elems());
    Ok(!windowed
        && windows_cover
        && w.plan.layers.iter().all(|l| {
            !l.attention.is_attention()
                || be.supports_row_layout(l.attention.head_dim())
        }))
}

/// Everything a row-layout forward over `b` nodes at `start` needs before it runs: the recurrent
/// checkpoint note taken out (returned, to be put back), the cache grown to `start + b`, the live
/// recurrent state on plane 0, room for every node's convolution-window inputs (their layout is
/// returned), and the layout words uploaded.
fn stage_tree_forward<A: Architecture>(
    w: &mut Workflow<A>,
    words: &[u32],
    start: usize,
    b: usize,
) -> Result<(usize, Vec<u8>, RowInputLayout), String> {
    let row_inputs = RowInputLayout::of(&w.plan, &A::conv_windows(&w.plan))?;
    let (note_at, note) = recurrent_note(w);
    w.kv_fit(start + b)?;
    prepare_recurrent_plane(w)?;
    let be = crate::gpu_support::be();
    if row_inputs.row_elems != 0 {
        // Sized for the widest layout, so a tree of any size reuses the one buffer.
        be.alloc(
            BufId::RowInputs,
            imparo_backend::ROW_LAYOUT_MAX_ROWS as u64
                * u64::from(row_inputs.row_elems)
                * 4,
        )
        .map_err(|rc| format!("row inputs allocation rc={rc}"))?;
    }
    be.alloc(
        BufId::RowLayout,
        (imparo_backend::ROW_LAYOUT_MAX_ROWS * imparo_backend::ROW_LAYOUT_WORDS * 4)
            as u64,
    )
    .map_err(|rc| format!("row layout allocation rc={rc}"))?;
    be.begin();
    be.write_u32(BufId::RowLayout, 0, words);
    be.end()
        .map_err(|rc| format!("tree layout upload rc={rc}"))?;
    Ok((note_at, note, row_inputs))
}

/// The target state after a row-layout forward: the fill index at `filled`, the window history
/// and the recurrent checkpoint note as given, no snapshot pending, the causal output demand. It
/// never touches the live recurrent state: the forward keeps every node's window inputs aside,
/// and a commit rebuilds the accepted path's windows before this runs.
fn settle_tree_state<A: Architecture>(
    w: &mut Workflow<A>,
    filled: usize,
    history: WindowHistory,
    note: Vec<u8>,
    note_at: usize,
) {
    w.state.row_layout = 0;
    w.state.output_demand = OutputDemand::LastToken;
    w.state.kv_rt.filled = filled;
    w.state.window_history = history;
    w.state.recur_ckpt = note;
    w.state.recur_ckpt_at = note_at;
    w.state.recur_ckpt_on_device = false;
    w.state.recur_snap = None;
}

/// Every node's logits from one row-layout forward over `tree` at `start`, then the target state
/// as it was: fill index, window history and the recurrent checkpoint note. A probe of the tree
/// forward, not the verify transaction.
pub(crate) fn forward_workflow_tree_logits<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let b = tree.tokens.len();
    if tree.parents.len() != b {
        return Err("tree tokens and parents differ in length".into());
    }
    validate(w, &tree.tokens, start)?;
    let cell = crate::prefill_batch().max(1);
    if !A::ROW_LAYOUT_FORWARD
        || !A::DEVICE_ALL_LOGITS
        || w.state.host_forward
        || b > cell - start % cell
    {
        return Err(
            "a tree forward runs as one device batch inside one prefill cell, on a forward that reads the row layout"
                .into(),
        );
    }
    let words = tree_row_layout(start, &tree.parents)?;
    w.ensure_gpu_ready()?;
    let be = crate::gpu_support::be();
    if w.plan.layers.iter().any(|l| {
        l.attention.is_attention() && !be.supports_row_layout(l.attention.head_dim())
    }) {
        return Err(
            "the backend serves no row layout at this model's attention head dims"
                .into(),
        );
    }
    let history = w.state.window_history.clone();
    let (note_at, note, _) = stage_tree_forward(w, &words, start, b)?;
    w.state.row_layout = b as u32;
    let result = w
        .forward_with_output_demand(
            &tree.tokens,
            start,
            out,
            None,
            OutputDemand::AllTokens,
            None,
        )
        .and_then(|()| {
            be.end()
                .map_err(|rc| format!("tree forward completion rc={rc}"))
        });
    if result.is_err() {
        let _ = be.end();
    }
    settle_tree_state(w, start, history, note, note_at);
    result?;
    let vocab = w.plan.config.vocab_size as usize;
    if out.len() != b * vocab {
        return Err(format!(
            "tree forward returned {} logits for {b} nodes",
            out.len()
        ));
    }
    Ok(())
}

/// The greedy walk down a verified tree: from the root, the child whose token is the target's
/// pick at its parent, until the pick has no such child, is a stop token, or the path holds
/// `limit` nodes. Returns the path (node indices, root first) and the pick after its last node,
/// which the path does not consume.
pub(crate) fn tree_accepted_path(
    picks: &[u32],
    tokens: &[u32],
    parents: &[i32],
    limit: usize,
    stops: &[u32],
) -> (Vec<i32>, u32) {
    let mut path = vec![0_i32];
    loop {
        let parent = path[path.len() - 1] as usize;
        let next = picks[parent];
        if path.len() >= limit || stops.contains(&next) {
            return (path, next);
        }
        let child = (parent + 1..tokens.len())
            .find(|&j| parents[j] == parent as i32 && tokens[j] == next);
        match child {
            Some(j) => path.push(j as i32),
            None => return (path, next),
        }
    }
}

/// Tree admission for the row-layout verify: a well-formed tree that fits here. False keeps the
/// chain: the tree crosses the prefill cell or passes the cache's capacity.
fn prepare_workflow_tree_rows<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
) -> Result<bool, String> {
    let b = tree.tokens.len();
    if tree.parents.len() != b {
        return Err("tree tokens and parents differ in length".into());
    }
    tree_row_layout(start, &tree.parents)?;
    let cell = crate::prefill_batch().max(1);
    let end = start.checked_add(b).ok_or("tree position overflow")?;
    if b > cell - start % cell || end > w.state.kv_rt.capacity {
        return Ok(false);
    }
    validate(w, &tree.tokens, start)?;
    Ok(true)
}

/// A tree verify through a row layout, then its commit:
///
/// ```text
/// forward  node t roped at start + depth(t), its K/V stored at start + t, its input to every
///          convolution window kept in RowInputs row t; one pick per node
/// walk     tree_accepted_path over the picks
/// commit   the K/V of path[i] moved from start + path[i] to start + i, where the two differ
///          every convolution window rebuilt from the path's kept inputs and the old window
///          fill index = start + path length
/// ```
///
/// A failure before the commit leaves the target as it was.
fn verify_workflow_tree_rows<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
    limit: usize,
    stops: &[u32],
) -> Result<crate::speculative::TreeVerification, String> {
    // A path never holds more nodes than the tree, so a limit above the tree's size caps nothing.
    let b = tree.tokens.len();
    if tree.parents.len() != b || limit == 0 {
        return Err(
            "tree verify: tokens and parents differ in length, or the limit is 0"
                .into(),
        );
    }
    validate(w, &tree.tokens, start)?;
    let end = start + b;
    let cell = crate::prefill_batch().max(1);
    if b > cell - start % cell || end > u32::MAX as usize {
        return Err("a tree verify runs inside one prefill cell".into());
    }
    let words = tree_row_layout(start, &tree.parents)?;
    let geom = crate::kv::state_geometry(&w.plan, ring_batch(w));
    let slack = imparo_kv::state::window_slack(&geom);
    let history = w
        .state
        .window_history
        .for_verification(start, end, slack)
        .ok_or("tree verify prefix coverage")?;
    let ticket = history.clone().begin(start)?;
    let (note_at, note, row_inputs) = stage_tree_forward(w, &words, start, b)?;
    let be = crate::gpu_support::be();
    w.state.window_history = history.clone();
    w.state.row_layout = b as u32;
    let vocab = w.plan.config.vocab_size;
    let mut rows = Vec::new();
    // THE TARGET DUMP (a data probe): every row's logits come back instead of its argmax, the picks
    // are taken on the host, and each committed position's top candidates are written below.
    let dump = target_dump();
    let walk = w
        .forward_with_output_demand(
            &tree.tokens,
            start,
            &mut rows,
            None,
            if dump.is_some() {
                OutputDemand::AllTokens
            } else {
                OutputDemand::RowArgmax
            },
            None,
        )
        .and_then(|()| {
            be.end()
                .map_err(|rc| format!("tree verify completion rc={rc}"))
        })
        .and_then(|()| {
            let picks: Vec<u32> = if dump.is_some() {
                let v = vocab as usize;
                if rows.len() != b * v {
                    return Err(format!(
                        "tree verify returned {} logits for {b} nodes",
                        rows.len()
                    ));
                }
                rows.chunks(v)
                    .map(|row| {
                        let mut best = 0;
                        for (i, x) in row.iter().enumerate() {
                            if *x > row[best] {
                                best = i;
                            }
                        }
                        u32::try_from(best).unwrap_or(u32::MAX)
                    })
                    .collect()
            } else {
                if rows.len() != b {
                    return Err(format!(
                        "tree verify returned {} picks for {b} nodes",
                        rows.len()
                    ));
                }
                rows.iter().map(|v| v.to_bits()).collect()
            };
            if picks.iter().any(|&pick| pick >= vocab) {
                return Err("tree verify pick outside the vocabulary".into());
            }
            Ok(tree_accepted_path(
                &picks,
                &tree.tokens,
                &tree.parents,
                limit,
                stops,
            ))
        });
    let (path, next_token) = match walk {
        Ok(accepted) => accepted,
        Err(e) => {
            let _ = be.end();
            settle_tree_state(w, start, history, note, note_at);
            return Err(e);
        }
    };
    if let Some(file) = dump {
        let file = file.as_ref().map_err(Clone::clone)?;
        write_target_dump(file, start, &path, &rows, vocab as usize)?;
    }
    // Each accepted node's rows move down to its depth. A node's depth never exceeds its batch
    // row, so no copy overwrites the source of a later one.
    let (from, to): (Vec<u32>, Vec<u32>) = path
        .iter()
        .enumerate()
        .filter(|&(depth, &node)| node as usize != depth)
        .map(|(depth, &node)| ((start + node as usize) as u32, (start + depth) as u32))
        .unzip();
    if !from.is_empty() {
        for g in geom
            .iter()
            .filter(|g| matches!(g.kind, imparo_kv::StateKind::Full))
        {
            if !be.kv_move_rows(
                g.layer,
                g.k_stride as u64,
                g.v_stride as u64,
                &from,
                &to,
            ) {
                settle_tree_state(w, start, history, note, note_at);
                return Err(format!(
                    "tree verify: the backend did not move layer {}'s accepted rows",
                    g.layer
                ));
            }
        }
    }
    // THE GRID BOUNDARY THIS PATH CROSSED, if any: its state becomes the recurrent note, as a
    // one-token decode step's does when it crosses one (`arm_recurrent_snapshot`). Without it a
    // speculative turn kept the note of its prompt, and the checkpoint at the turn's end was
    // dropped for want of a recurrent state.
    let committed = start + path.len();
    let grid = imparo_kv::grid_tokens();
    let boundary = committed / grid * grid;
    let below_boundary = (boundary > start).then(|| boundary - start);
    let crossed = match commit_conv_windows(
        &row_inputs,
        &path,
        below_boundary,
        w.plan.recurrent_elems() as usize,
    ) {
        Ok(state) => state,
        Err(e) => {
            settle_tree_state(w, start, history, note, note_at);
            return Err(format!(
                "tree verify recurrent commit: {e}; the KV rows moved, so device state recovery is unconfirmed"
            ));
        }
    };
    if verify_log() {
        eprintln!(
            "[verify] path=tree b={b} start={start} out=row_picks consumed={} moved_rows={} next={next_token} nodes={path:?} snapshot={}",
            path.len(),
            from.len(),
            if crossed.is_some() { boundary } else { 0 }
        );
    }
    let (note_at, note) = match crossed {
        Some(state) => (boundary, state),
        None => (note_at, note),
    };
    settle_tree_state(w, committed, history, note, note_at);
    w.state
        .window_history
        .commit(ticket, start, committed, slack);
    Ok(crate::speculative::TreeVerification { path, next_token })
}

/// Candidates per committed position in the target dump.
const TARGET_DUMP_TOP: usize = 32;

/// The file `IMPARO_DSPARK_TARGET_DUMP=PATH` names: for an offline replay that trains the drafter
/// toward the target's distribution instead of the committed token alone. A path that cannot be
/// opened fails every verify rather than leaving a run without its records.
type TargetDump = Result<std::sync::Mutex<std::fs::File>, String>;

fn target_dump() -> Option<&'static TargetDump> {
    static FILE: std::sync::OnceLock<Option<TargetDump>> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::var("IMPARO_DSPARK_TARGET_DUMP").ok()?;
        Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map(std::sync::Mutex::new)
                .map_err(|e| format!("IMPARO_DSPARK_TARGET_DUMP={path}: {e}")),
        )
    })
    .as_ref()
}

/// One record per verified tree: the target's distribution at every committed position.
///
/// ```text
///   u32 x 3                    start, path length n, candidates k
///   per path node d (0..n)     u32 position start + d + 1 (the position its row predicts),
///                              u32 the target's pick there, u32 x k ids, f32 x k logits (largest first)
/// ```
///
/// Row `path[d]` is the node at depth `d` of the accepted path, so its logits are the target's
/// prediction for the committed token at `start + d + 1`; rows off the path are conditioned on
/// tokens that were not committed and are not written.
fn write_target_dump(
    file: &std::sync::Mutex<std::fs::File>,
    start: usize,
    path: &[i32],
    logits: &[f32],
    vocab: usize,
) -> Result<(), String> {
    use std::io::Write;
    let word = |v: usize| u32::try_from(v).unwrap_or(u32::MAX).to_le_bytes();
    let mut out: Vec<u8> =
        Vec::with_capacity(12 + path.len() * (8 + 8 * TARGET_DUMP_TOP));
    out.extend_from_slice(&word(start));
    out.extend_from_slice(&word(path.len()));
    out.extend_from_slice(&word(TARGET_DUMP_TOP));
    for (d, &node) in path.iter().enumerate() {
        let node =
            usize::try_from(node).map_err(|_| "target dump: a negative path node")?;
        let row = logits
            .get(node * vocab..(node + 1) * vocab)
            .ok_or("target dump: a path node past the logits")?;
        let mut order: Vec<usize> = (0..vocab).collect();
        let by_value = |a: &usize, b: &usize| {
            row[*b]
                .partial_cmp(&row[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        };
        order.select_nth_unstable_by(TARGET_DUMP_TOP - 1, by_value);
        order.truncate(TARGET_DUMP_TOP);
        order.sort_unstable_by(by_value);
        out.extend_from_slice(&word(start + d + 1));
        out.extend_from_slice(&word(order[0]));
        for &id in &order {
            out.extend_from_slice(&word(id));
        }
        for &id in &order {
            out.extend_from_slice(&row[id].to_le_bytes());
        }
    }
    file.lock()
        .map_err(|_| "target dump: the file's lock is poisoned".to_string())?
        .write_all(&out)
        .map_err(|e| format!("target dump: {e}"))
}

/// Tree admission: through a row layout where this model and backend run one, else the CUDA
/// lab's tree where that feature is built, else none (the chain).
pub(crate) fn prepare_greedy_tree<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
) -> Result<bool, String> {
    if tree_rows_served(w)? {
        return prepare_workflow_tree_rows(w, tree, start);
    }
    #[cfg(feature = "cuda-speculative")]
    {
        prepare_workflow_tree(w, tree, start)
    }
    #[cfg(not(feature = "cuda-speculative"))]
    {
        Ok(false)
    }
}

/// The tree verify on the route `prepare_greedy_tree` admitted.
pub(crate) fn verify_greedy_tree<A: Architecture>(
    w: &mut Workflow<A>,
    tree: &crate::speculative::DraftTree,
    start: usize,
    limit: usize,
    stops: &[u32],
) -> Result<crate::speculative::TreeVerification, String> {
    if tree_rows_served(w)? {
        return verify_workflow_tree_rows(w, tree, start, limit, stops);
    }
    #[cfg(feature = "cuda-speculative")]
    {
        verify_workflow_tree(w, tree, start, limit, stops)
    }
    #[cfg(not(feature = "cuda-speculative"))]
    {
        Err("tree verification unsupported by this model adapter".into())
    }
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
    {
        return Err("tree target capability/shape".into());
    }
    // The cursor now supplies an output budget, which may exceed the tree depth.
    // Parent validation bounds every native path to nine nodes; a larger budget
    // cannot create a longer path and must not reject an otherwise valid tree.
    validate_tree_parents(parents)?;
    validate(w, tokens, start)?;
    let end = start + b;
    let cell = crate::prefill_batch().max(1);
    if b > cell - start % cell || end > u32::MAX as usize {
        return Err("tree crosses supported forward cell".into());
    }
    let geom = crate::kv::state_geometry(&w.plan, ring_batch(w));
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
            let (path, next) =
                tree_accepted_path(&picks, tokens, parents, limit, stops);
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

#[cfg(test)]
mod tests {
    use super::{WindowSource, tree_accepted_path, tree_row_layout, window_sources};

    #[test]
    fn a_rebuilt_window_takes_path_inputs_then_old_slots() {
        use WindowSource::{OldSlot, PathNode};
        assert_eq!(window_sources(1, 2), vec![OldSlot(1), PathNode(0)]);
        assert_eq!(window_sources(6, 2), vec![PathNode(4), PathNode(5)]);
        assert_eq!(
            window_sources(1, 3),
            vec![OldSlot(1), OldSlot(2), PathNode(0)]
        );
        assert_eq!(
            window_sources(2, 3),
            vec![OldSlot(2), PathNode(0), PathNode(1)]
        );
    }

    #[test]
    fn a_rebuilt_window_equals_the_path_shifted_into_the_old_window() {
        for history in 1..=4 {
            for path_len in 1..=6 {
                // Old slot s holds 100 + s; the path's node i holds i.
                let mut chain: Vec<usize> = (0..history).map(|s| 100 + s).collect();
                chain.extend(0..path_len);
                let rebuilt: Vec<usize> = window_sources(path_len, history)
                    .into_iter()
                    .map(|source| match source {
                        WindowSource::PathNode(i) => i,
                        WindowSource::OldSlot(s) => 100 + s,
                    })
                    .collect();
                assert_eq!(
                    rebuilt,
                    &chain[chain.len() - history..],
                    "history {history}, path of {path_len}"
                );
            }
        }
    }

    /// A chain 0-1-2-3 and a branch 4-5 off node 1.
    const PARENTS: [i32; 6] = [-1, 0, 1, 2, 1, 4];
    const TOKENS: [u32; 6] = [10, 11, 12, 13, 20, 21];

    #[test]
    fn the_walk_follows_the_picks_into_a_branch() {
        let picks = [11, 20, 99, 99, 21, 7];
        assert_eq!(
            tree_accepted_path(&picks, &TOKENS, &PARENTS, 8, &[]),
            (vec![0, 1, 4, 5], 7)
        );
    }

    #[test]
    fn the_walk_ends_at_the_limit_a_stop_token_or_a_leaf() {
        let picks = [11, 12, 13, 14, 21, 7];
        assert_eq!(
            tree_accepted_path(&picks, &TOKENS, &PARENTS, 2, &[]),
            (vec![0, 1], 12)
        );
        assert_eq!(
            tree_accepted_path(&picks, &TOKENS, &PARENTS, 8, &[12]),
            (vec![0, 1], 12)
        );
        assert_eq!(
            tree_accepted_path(&picks, &TOKENS, &PARENTS, 8, &[]),
            (vec![0, 1, 2, 3], 14)
        );
        assert_eq!(
            tree_accepted_path(&[5; 6], &TOKENS, &PARENTS, 8, &[]),
            (vec![0], 5)
        );
    }

    #[cfg(feature = "cuda-speculative")]
    #[test]
    fn native_frontier_accepts_a_budget_above_its_depth() {
        let parents = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6];
        super::validate_tree_parents(&parents).unwrap();
        let tokens: Vec<u32> = (0..16).collect();
        let picks: Vec<u32> = (1..17).collect();
        assert_eq!(
            tree_accepted_path(&picks, &tokens, &parents, 16, &[]),
            ((0..9).collect(), 9)
        );
        assert_eq!(
            tree_accepted_path(&picks, &tokens, &parents, 3, &[]),
            (vec![0, 1, 2], 3)
        );
        let mut invalid = parents;
        invalid[9] = 8;
        assert!(super::validate_tree_parents(&invalid).is_err());
    }

    #[test]
    fn the_layout_holds_positions_visibility_and_ancestors() {
        let words = tree_row_layout(100, &PARENTS).unwrap();
        let row = |t: usize| &words[t * 12..(t + 1) * 12];
        // Node 5: path 0, 1, 4, 5 -- depth 3, sees those four rows, ancestors 4, 1, 0.
        assert_eq!(row(5)[0], 103);
        assert_eq!(row(5)[1], 3);
        assert_eq!(row(5)[2], 0b11_0011);
        assert_eq!(row(5)[3], 0);
        assert_eq!(&row(5)[4..7], &[4, 1, 0]);
        // Node 3: the chain's fourth row.
        assert_eq!(row(3)[0], 103);
        assert_eq!(row(3)[2], 0b1111);
        assert!(tree_row_layout(0, &[0, -1]).is_err());
        assert!(tree_row_layout(0, &[-1, 1]).is_err());
    }
}
