//! Static, owner-lab Gemma4 assistant calls; no required dynamic-loader ABI change.
//! The caller retains the target owner, its KV and composite weight mapping.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Config {
    pub target_embedding: u64,
    pub pre_proj: u64,
    pub post_proj: u64,
    pub out_norm: u64,
    pub head: u64,
    pub hidden: u32,
    pub target_hidden: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub batch_capacity: u32,
    pub target_final_layer: u32,
    pub target_embedding_kind: u32,
    pub reserved: u32,
    pub eps: f32,
    pub embedding_scale: f32,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Layer {
    pub attn_norm: u64,
    pub q: u64,
    pub qn: u64,
    pub o: u64,
    pub attn_post_norm: u64,
    pub ffn_norm: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    pub ffn_post_norm: u64,
    pub rope_freqs: u64,
    pub target_kv_layer: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub window: u32,
    pub ring: u32,
    pub had_k: u32,
    pub had_v: u32,
    pub has_rope_freqs: u32,
    pub rope_theta: f32,
    pub out_scale: f32,
}
const _: () =
    assert!(std::mem::size_of::<Config>() == 88 && std::mem::size_of::<Layer>() == 128);
unsafe extern "C" {
    fn imparo_cuda_gemma4_mtp_attach(
        config: *const Config,
        layers: *const Layer,
        count: u32,
    ) -> i32;
    fn imparo_cuda_gemma4_mtp_reset() -> i32;
    fn imparo_cuda_gemma4_mtp_capture_mode(enabled: u32) -> i32;
    fn imparo_cuda_gemma4_mtp_capture_layer(
        layer: u32,
        start: u32,
        rows: u32,
        src: u32,
    ) -> i32;
    fn imparo_cuda_gemma4_mtp_append(start: u32, consumed: u32) -> i32;
    fn imparo_cuda_gemma4_mtp_append_tree(
        start: u32,
        path: *const i32,
        count: u32,
    ) -> i32;
    fn imparo_cuda_gemma4_mtp_generate(start: u32, anchor: u32, ids: *mut u32) -> i32;
    fn imparo_cuda_gemma4_mtp_generate_selected(start: u32, anchor: u32, ids: *mut u32) -> i32;
    fn imparo_cuda_gemma4_mtp_generate_tree4(
        start: u32,
        anchor: u32,
        ids: *mut u32,
    ) -> i32;
    fn imparo_cuda_gemma4_mtp_generate_gated(
        start: u32,
        anchor: u32,
        minimum: f32,
        ids: *mut u32,
        count: *mut u32,
        probability: *mut f32,
    ) -> i32;
    fn imparo_cuda_gemma4_mtp_detach() -> i32;
}
fn checked(rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("Gemma4 MTP native rc={rc}; stop this continuation"))
    }
}
/// # Safety
/// Validate all offsets/types/shapes and actual target KV owner geometry. Retain
/// exclusive same-thread target ownership and live weights through successful detach.
/// Target KV is borrowed read-only; no other provider may be attached concurrently.
pub unsafe fn attach(config: &Config, layers: &[Layer]) -> Result<(), String> {
    let count = u32::try_from(layers.len()).map_err(|_| "MTP layer count overflow")?;
    if crate::knobs::e4b_retained_decode_policy_enabled() {
        crate::mtp_cluster_assets::prepare()?;
    }
    unsafe {
        checked(imparo_cuda_gemma4_mtp_attach(
            config,
            layers.as_ptr(),
            count,
        ))
    }
}
/// # Safety
/// Own the attached target at a closed execution boundary. Invalidates hidden history.
pub unsafe fn reset() -> Result<(), String> {
    unsafe { checked(imparo_cuda_gemma4_mtp_reset()) }
}
/// # Safety
/// Retain attached target and weights, at a closed boundary. Capture must not allow
/// replay of a graph that omits the normalized-hidden callback or retains stale destinations.
pub unsafe fn capture_mode(enabled: bool) -> Result<(), String> {
    unsafe { checked(imparo_cuda_gemma4_mtp_capture_mode(u32::from(enabled))) }
}
/// # Safety
/// src contains the dense final target X tail after output RMSNorm, beginning
/// at the actual absolute start after pruning. LastToken normalizes only the last
/// physical row; its commit must select that row. AllTokens normalizes every row.
pub unsafe fn capture_layer(
    layer: u32,
    start: u32,
    rows: u32,
    src: u32,
) -> Result<(), String> {
    unsafe {
        checked(imparo_cuda_gemma4_mtp_capture_layer(
            layer, start, rows, src,
        ))
    }
}
/// # Safety
/// Target has committed this accepted prefix. The native callback selects absolute
/// row start+consumed-1, never the last attempted row of a rejected batch. State-only
/// prefill chunks may advance coverage without hidden, but cannot authorize generate.
pub unsafe fn append(start: u32, consumed: u32) -> Result<(), String> {
    unsafe { checked(imparo_cuda_gemma4_mtp_append(start, consumed)) }
}
fn tree_commit_count(start: u32, path: &[i32]) -> Result<u32, String> {
    let count = u32::try_from(path.len()).map_err(|_| "MTP tree path overflow")?;
    if path.first() != Some(&0)
        || path.windows(2).any(|pair| pair[0] >= pair[1])
        || start.checked_add(count).is_none()
        || path
            .last()
            .and_then(|&node| u32::try_from(node).ok())
            .and_then(|node| start.checked_add(node))
            .is_none()
    {
        return Err("MTP tree commit path invalid".into());
    }
    Ok(count)
}
/// # Safety
/// Target has verified and committed this root-first accepted path. Every index names a
/// normalized row in the current capture at `start`, and parent/child edges were checked
/// by the target verifier. Native selects FEATURE[path.last()] while advancing hidden
/// history by path.len(); missing/stale captured rows are errors, never an empty hidden.
pub unsafe fn append_tree(start: u32, path: &[i32]) -> Result<(), String> {
    let count = tree_commit_count(start, path)?;
    unsafe { checked(imparo_cuda_gemma4_mtp_append_tree(start, path.as_ptr(), count)) }
}
/// # Safety
/// Target state and committed post-output-norm hidden both end at start. The two predictions
/// use the same target position, with h_p paired with target embedding x_(p+1).
pub unsafe fn generate(
    start: u32,
    anchor: u32,
    ids: &mut [u32; 2],
) -> Result<(), String> {
    unsafe {
        checked(imparo_cuda_gemma4_mtp_generate(
            start,
            anchor,
            ids.as_mut_ptr(),
        ))
    }
}
/// # Safety
/// Same boundary as generate, in the default-off short E4B selector domain.
/// Returns [selected first, conditional second, original first, alternate first].
/// The ordinary target verifier must validate both proposals before any commit.
pub unsafe fn generate_selected(start: u32, anchor: u32, ids: &mut [u32; 4]) -> Result<(), String> {
    unsafe { checked(imparo_cuda_gemma4_mtp_generate_selected(start, anchor, ids.as_mut_ptr())) }
}
/// # Safety
/// Same committed target/hidden boundary as generate, in the admitted short E4B tree4
/// geometry. Returns the original two-step chain and a second candidate from the
/// selected first/second step (the explicit delayed-tree lab flag). Both reuse
/// that step's dense/cluster logits, without another assistant forward.
pub unsafe fn generate_tree4(
    start: u32,
    anchor: u32,
    ids: &mut [u32; 3],
) -> Result<(), String> {
    unsafe {
        checked(imparo_cuda_gemma4_mtp_generate_tree4(
            start,
            anchor,
            ids.as_mut_ptr(),
        ))
    }
}
/// # Safety
/// Same target/hidden boundary as generate. Zero proposals changes no committed
/// state and requires an observed target M1 commit before the next attempt.
pub unsafe fn generate_gated(
    start: u32,
    anchor: u32,
    minimum: f32,
    ids: &mut [u32; 2],
) -> Result<(u32, f32), String> {
    let mut count = 0;
    let mut probability = 0.0;
    unsafe {
        checked(imparo_cuda_gemma4_mtp_generate_gated(
            start,
            anchor,
            minimum,
            ids.as_mut_ptr(),
            &raw mut count,
            &raw mut probability,
        ))?;
    }
    if !matches!(count, 0 | 2)
        || !probability.is_finite()
        || probability <= 0.0
        || probability > 1.0
    {
        return Err("MTP invalid confidence result".into());
    }
    Ok((count, probability))
}
/// # Safety
/// Keep the attached target and weights alive through successful teardown. Native
/// must invalidate any graph destinations before releasing its assistant buffers.
pub unsafe fn detach() -> Result<(), String> {
    unsafe { checked(imparo_cuda_gemma4_mtp_detach()) }
}

#[cfg(test)]
mod tests {
    use super::tree_commit_count;

    #[test]
    fn accepted_tree_path_counts_tokens_instead_of_storage_rows() {
        assert_eq!(tree_commit_count(1022, &[0, 3]).unwrap(), 2);
        assert_eq!(tree_commit_count(1022, &[0]).unwrap(), 1);
        assert_eq!(tree_commit_count(1022, &[0, 1, 2]).unwrap(), 3);
    }

    #[test]
    fn malformed_or_overflowing_tree_paths_are_rejected_before_native() {
        for path in [&[][..], &[1], &[-1], &[0, -1], &[0, 0], &[0, 3, 1]] {
            assert!(tree_commit_count(512, path).is_err(), "{path:?}");
        }
        assert!(tree_commit_count(u32::MAX, &[0]).is_err());
        assert!(tree_commit_count(u32::MAX - 2, &[0, 3]).is_err());
        assert_eq!(tree_commit_count(u32::MAX - 2, &[0, 1]).unwrap(), 2);
    }
}
