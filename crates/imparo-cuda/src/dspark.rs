//! Static DSpark operator provider on existing CUDA execution owners.
//! Calls require exclusive serialized backend access and a live shared weight mapping.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Config {
    pub embedding: u64,
    pub fc: u64,
    pub enc_norm: u64,
    pub out_norm: u64,
    pub markov1: u64,
    pub markov2: u64,
    pub confidence: u64,
    pub confidence_bias: u64,
    pub hidden: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rank: u32,
    pub block_size: u32,
    pub mask_token: u32,
    pub kv_capacity: u32,
    pub batch_capacity: u32,
    pub target_hidden: u32,
    pub eps: f32,
    pub rope_theta: f32,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Layer {
    pub attn_norm: u64,
    pub q: u64,
    pub k: u64,
    pub v: u64,
    pub o: u64,
    pub qn: u64,
    pub kn: u64,
    pub ffn_norm: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}
const _: () =
    assert!(std::mem::size_of::<Config>() == 120 && std::mem::size_of::<Layer>() == 88);
/// One expanded parent's complete local top-4 before beam pruning.
/// `logp` is normalized over the vocabulary, not the four returned choices.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct FrontierGroup {
    pub parent: u32,
    pub tokens: [u32; 4],
    pub logp: [f32; 4],
    pub confidence: f32,
}
const _: () = assert!(std::mem::size_of::<FrontierGroup>() == 40);
unsafe extern "C" {
    fn imparo_cuda_dspark_attach(
        c: *const Config,
        w: *const Layer,
        n: u32,
        t: *const u32,
        nt: u32,
    ) -> i32;
    fn imparo_cuda_dspark_suspend(ticket: *mut u64, history: *mut u32) -> i32;
    fn imparo_cuda_dspark_resume(
        ticket: u64,
        start: u32,
        c: *const Config,
        w: *const Layer,
        n: u32,
        t: *const u32,
        nt: u32,
    ) -> i32;
    fn imparo_cuda_dspark_reset() -> i32;
    fn imparo_cuda_dspark_capture_mode(enabled: u32) -> i32;
    fn imparo_cuda_dspark_capture_layer(
        layer: u32,
        start: u32,
        n: u32,
        src: u32,
    ) -> i32;
    fn imparo_cuda_dspark_append(start: u32, n: u32) -> i32;
    fn imparo_cuda_dspark_generate(
        start: u32,
        anchor: u32,
        ids: *mut u32,
        conf: *mut f32,
    ) -> i32;
    fn imparo_cuda_dspark_detach() -> i32;
    fn imparo_cuda_dspark_generate_with_accept_features(
        start: u32,
        anchor: u32,
        ids: *mut u32,
        conf: *mut f32,
    ) -> i32;
    fn imparo_cuda_dspark_frontier_groups(
        start: u32,
        anchor: u32,
        groups: *mut FrontierGroup,
    ) -> i32;
}
fn checked(rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("DSpark native rc={rc}; stop this continuation"))
    }
}
/// # Safety
/// Validate every descriptor span/type/shape against the live shared weights; keep
/// weights and exclusive target execution ownership until successful detach.
/// Fresh attach may destroy a parked Session bound to this target; it cannot take
/// a live Session. Invalidate any old parked ticket when replacing the Session.
pub unsafe fn attach(
    c: &Config,
    layers: &[Layer],
    targets: &[u32],
) -> Result<(), String> {
    let n = u32::try_from(layers.len()).map_err(|_| "too many draft layers")?;
    let nt = u32::try_from(targets.len()).map_err(|_| "too many target features")?;
    unsafe {
        checked(imparo_cuda_dspark_attach(
            c,
            layers.as_ptr(),
            n,
            targets.as_ptr(),
            nt,
        ))
    }
}
/// Park the current Session without releasing its draft owner or committed KV.
/// Returns a single-use ticket and the actual committed history length.
/// # Safety
/// Own exclusive target/backend access at a closed boundary. Keep target owner,
/// shared weights, and native backend alive until resume or successful detach;
/// parking does not transfer their ownership. Disable the model's layer-output
/// subscription before allowing any unrelated target forward. A final owner must
/// explicitly detach before releasing target/weights, including on cache eviction.
pub unsafe fn suspend() -> Result<(u64, u32), String> {
    let mut ticket = 0u64;
    let mut history = 0u32;
    unsafe {
        checked(imparo_cuda_dspark_suspend(
            &raw mut ticket,
            &raw mut history,
        ))?;
    }
    Ok((ticket, history))
}
/// Resume a parked Session at an already committed, matching target prefix.
/// A rejected ticket/descriptor leaves the existing Session unchanged.
/// # Safety
/// Own exclusive access to the same target and live shared weights at a closed
/// boundary. Prove exact token-prefix identity through start and restore its
/// matching target KV/recurrent checkpoint before this call; native validation
/// cannot infer token identity from KV. Use the original validated descriptors and
/// unchanged inference configuration. Never advance beyond the returned history.
/// Re-establish the model's exclusive output subscription and explicitly enable
/// capture after success. Do not call reset: retained history would be discarded.
pub unsafe fn resume(
    ticket: u64,
    start: u32,
    c: &Config,
    layers: &[Layer],
    targets: &[u32],
) -> Result<(), String> {
    let n = u32::try_from(layers.len()).map_err(|_| "too many draft layers")?;
    let nt = u32::try_from(targets.len()).map_err(|_| "too many target features")?;
    unsafe {
        checked(imparo_cuda_dspark_resume(
            ticket,
            start,
            c,
            layers.as_ptr(),
            n,
            targets.as_ptr(),
            nt,
        ))
    }
}
/// # Safety
/// Own exclusive access to the attached target and live weights at a closed boundary.
pub unsafe fn reset() -> Result<(), String> {
    unsafe { checked(imparo_cuda_dspark_reset()) }
}
/// # Safety
/// Same lifetime/serialization requirement as attach; target forward must be closed.
pub unsafe fn capture_mode(enabled: bool) -> Result<(), String> {
    unsafe { checked(imparo_cuda_dspark_capture_mode(u32::from(enabled))) }
}
/// # Safety
/// Called in the target forward with n dense rows of its complete layer residual.
pub unsafe fn capture_layer(
    layer: u32,
    start: u32,
    n: u32,
    src: u32,
) -> Result<(), String> {
    unsafe { checked(imparo_cuda_dspark_capture_layer(layer, start, n, src)) }
}
/// # Safety
/// Most recent target output must contain the accepted n-row prefix at start.
pub unsafe fn append(start: u32, n: u32) -> Result<(), String> {
    unsafe { checked(imparo_cuda_dspark_append(start, n)) }
}
/// # Safety
/// Retain attached target/weights; ids and confidence each have exactly block_size elements.
pub unsafe fn generate(
    start: u32,
    anchor: u32,
    ids: &mut [u32],
    confidence: &mut [f32],
    block_size: usize,
) -> Result<(), String> {
    if ids.len() != block_size || confidence.len() != block_size {
        return Err("draft output capacity mismatch".into());
    }
    unsafe {
        checked(imparo_cuda_dspark_generate(
            start,
            anchor,
            ids.as_mut_ptr(),
            confidence.as_mut_ptr(),
        ))
    }
}
/// # Safety
/// Same owner/lifetime as attach; do not use the detached descriptor or parked
/// ticket afterward. Valid for both live and parked Sessions. On failure retain
/// the owner and shared weight lifetime so teardown can be retried.
pub unsafe fn detach() -> Result<(), String> {
    unsafe { checked(imparo_cuda_dspark_detach()) }
}

unsafe extern "C" {
    fn imparo_cuda_dspark_tree_leaves(start: u32, anchor: u32, ids: *mut u32) -> i32;
    fn imparo_cuda_dspark_compact_features(path: *const i32, count: u32) -> i32;
}
/// # Safety
/// Same live exclusive target/provider, immediately following its draft call.
pub unsafe fn tree_leaves(start: u32, anchor: u32) -> Result<Vec<u32>, String> {
    let mut ids = vec![0; 7];
    unsafe {
        checked(imparo_cuda_dspark_tree_leaves(
            start,
            anchor,
            ids.as_mut_ptr(),
        ))?;
    }
    Ok(ids)
}
/// # Safety
/// Path was verified by target; most recent feature capture has its 2–16 tree rows.
pub unsafe fn compact_features(path: &[i32]) -> Result<(), String> {
    unsafe {
        checked(imparo_cuda_dspark_compact_features(
            path.as_ptr(),
            path.len() as u32,
        ))
    }
}

unsafe extern "C" {
    fn imparo_cuda_dspark_frontier_tree(
        start: u32,
        anchor: u32,
        ids: *mut u32,
        parents: *mut i32,
    ) -> i32;
}
/// # Safety
/// Same exclusive live owner, immediately after successful frontier generation.
pub unsafe fn frontier_tree(
    start: u32,
    anchor: u32,
) -> Result<(Vec<u32>, Vec<i32>), String> {
    let mut ids = vec![0; 16];
    let mut parents = vec![0; 16];
    unsafe {
        checked(imparo_cuda_dspark_frontier_tree(
            start,
            anchor,
            ids.as_mut_ptr(),
            parents.as_mut_ptr(),
        ))?;
    }
    Ok((ids, parents))
}

unsafe extern "C" {
    fn imparo_cuda_dspark_frontier_scores(start: u32, anchor: u32, scores: *mut f32) -> i32;
}
/// Cumulative log probabilities already cached by native frontier generation.
/// # Safety
/// Same exclusive owner and round as `frontier_tree`; no intervening draft.
pub unsafe fn frontier_scores(start: u32, anchor: u32) -> Result<Vec<f32>, String> {
    let mut scores = vec![0.0; 16];
    unsafe { checked(imparo_cuda_dspark_frontier_scores(start, anchor, scores.as_mut_ptr()))?; }
    Ok(scores)
}

unsafe extern "C" {
    fn imparo_cuda_dspark_frontier_candidates(
        start: u32,
        anchor: u32,
        ids: *mut u32,
        parents: *mut i32,
        scores: *mut f32,
    ) -> i32;
}
/// The 33 already-read beam nodes before the default 16-row selection.
/// Includes the anchor at row 0 with parent -1 and score 0. Every other parent
/// precedes its child, depth is at most 8, and scores are cumulative log
/// probabilities, not confidence-head values or target acceptance estimates.
/// No additional GPU work or synchronization is submitted by this accessor.
/// # Safety
/// Same exclusive live owner and round as `frontier_tree`, immediately after
/// successful frontier generation. No intervening generate, reset, append,
/// suspend, resume, or target-state change; each destination has 33 elements.
pub unsafe fn frontier_candidates(
    start: u32,
    anchor: u32,
) -> Result<(Vec<u32>, Vec<i32>, Vec<f32>), String> {
    let mut ids = vec![0; 33];
    let mut parents = vec![0; 33];
    let mut scores = vec![0.0; 33];
    unsafe {
        checked(imparo_cuda_dspark_frontier_candidates(
            start,
            anchor,
            ids.as_mut_ptr(),
            parents.as_mut_ptr(),
            scores.as_mut_ptr(),
        ))?;
    }
    Ok((ids, parents, scores))
}

/// Generate with the same kernels while retaining complete frontier groups and
/// computing their real confidence heads on the host. Additional feature copies
/// occur only when this call actually takes the native frontier route.
/// # Safety
/// Same owner, weight lifetime, and destination capacities as `generate`.
pub unsafe fn generate_with_accept_features(
    start: u32,
    anchor: u32,
    ids: &mut [u32],
    confidence: &mut [f32],
    block_size: usize,
) -> Result<(), String> {
    if ids.len() != block_size || confidence.len() != block_size {
        return Err("draft output capacity mismatch".into());
    }
    unsafe {
        checked(imparo_cuda_dspark_generate_with_accept_features(
            start, anchor, ids.as_mut_ptr(), confidence.as_mut_ptr(),
        ))
    }
}

/// The 29 expanded parents, indexed by parent node 0..28 in `frontier_candidates`.
/// Each group retains all four original local choices, even if beam pruning
/// omitted a child. Confidence is computed from this parent's token and depth.
/// # Safety
/// Same exclusive live owner and round as `frontier_candidates`, immediately
/// after a successful frontier `generate_with_accept_features`. No intervening
/// generation, reset, append, suspend, resume, or target-state change.
pub unsafe fn frontier_groups(
    start: u32,
    anchor: u32,
) -> Result<Vec<FrontierGroup>, String> {
    let mut groups = vec![FrontierGroup::default(); 29];
    unsafe {
        checked(imparo_cuda_dspark_frontier_groups(start, anchor, groups.as_mut_ptr()))?;
    }
    Ok(groups)
}
