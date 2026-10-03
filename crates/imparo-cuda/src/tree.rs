//! Explicit laboratory tree transaction on the currently owned CUDA target.

/// One actual KV owner from the model's StateGeometry. `ring` is a mask (slots
/// minus one), or zero for full attention; strides are bytes in the native codec.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KvRowGeometry {
    pub layer: u32,
    pub ring: u32,
    pub k_stride: u64,
    pub v_stride: u64,
}

/// An E4B KV owner derived from StateGeometry and its layer's attention shape.
/// `ring_slots` is the number of slots, unlike `KvRowGeometry::ring`'s mask.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct E4bTreeGeometry {
    pub layer: u32,
    pub head_dim: u32,
    pub window: u32,
    pub ring_slots: u32,
    pub k_stride: u64,
    pub v_stride: u64,
}

unsafe extern "C" {
    fn imparo_cuda_tree_prepare_e4b(
        start: u32,
        parents: *const i32,
        nodes: u32,
        layout: *const u32,
        layout_words: u32,
        geometry: *const E4bTreeGeometry,
        layers: u32,
        admitted: *mut u32,
    ) -> i32;
    fn imparo_cuda_tree_begin_e4b(
        start: u32,
        parents: *const i32,
        nodes: u32,
        layout: *const u32,
        layout_words: u32,
        geometry: *const E4bTreeGeometry,
        layers: u32,
    ) -> i32;
    fn imparo_cuda_tree_commit_e4b(path: *const i32, count: u32) -> i32;
    fn imparo_cuda_tree_commit_kv_rows(
        geometry: *const KvRowGeometry,
        layers: u32,
        from: *const u32,
        to: *const u32,
        count: u32,
    ) -> i32;
    fn imparo_cuda_tree_prepare(
        start: u32,
        parents: *const i32,
        nodes: u32,
        recurrent: u32,
        admitted: *mut u32,
    ) -> i32;
    fn imparo_cuda_tree_begin(
        start: u32,
        parents: *const i32,
        nodes: u32,
        recurrent: u32,
    ) -> i32;
    fn imparo_cuda_tree_commit(
        path: *const i32,
        count: u32,
        kv_width: u32,
        recur_buf: u32,
    ) -> i32;
    fn imparo_cuda_tree_end() -> i32;
}
fn check(rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("tree transaction rc={rc}"))
    }
}

fn e4b_layout_args(
    parents: &[i32],
    row_layout: &[u32],
    geometry: &[E4bTreeGeometry],
) -> Result<u32, String> {
    if !matches!(parents, [-1, 0, 1, 0] | [-1, 0, 1, 1])
        || row_layout.len() != 4 * imparo_backend::ROW_LAYOUT_WORDS
        || geometry.is_empty()
    {
        return Err("E4B tree topology/layout/owners".into());
    }
    u32::try_from(geometry.len()).map_err(|_| "E4B tree owners overflow".into())
}

/// # Safety
/// Exclusive closed CUDA target owner. Descriptors name every actual Own KV
/// layer. Native admission is limited to the explicit tree=4 laboratory route,
/// retained short-domain SM86/Q4, and start positions 512 through 765. False
/// declines before target-state writes; driver/allocation errors remain errors.
pub unsafe fn prepare_e4b(
    start: u32,
    parents: &[i32],
    row_layout: &[u32],
    geometry: &[E4bTreeGeometry],
) -> Result<bool, String> {
    let layers = e4b_layout_args(parents, row_layout, geometry)?;
    let mut admitted = 0;
    unsafe {
        check(imparo_cuda_tree_prepare_e4b(
            start, parents.as_ptr(), parents.len() as u32,
            row_layout.as_ptr(), row_layout.len() as u32,
            geometry.as_ptr(), layers, &raw mut admitted,
        ))?;
    }
    match admitted {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err("invalid E4B tree admission result".into()),
    }
}

/// # Safety
/// The same exclusive target owner and geometry admitted by prepare_e4b; no
/// competing forward until commit_e4b/end. Uploads public RowLayout and retains
/// the existing parents/depths scratch ABI for the native position operations.
pub unsafe fn begin_e4b(
    start: u32,
    parents: &[i32],
    row_layout: &[u32],
    geometry: &[E4bTreeGeometry],
) -> Result<(), String> {
    let layers = e4b_layout_args(parents, row_layout, geometry)?;
    unsafe {
        check(imparo_cuda_tree_begin_e4b(
            start, parents.as_ptr(), parents.len() as u32,
            row_layout.as_ptr(), row_layout.len() as u32,
            geometry.as_ptr(), layers,
        ))
    }
}

/// # Safety
/// Same live E4B transaction after all node rows have completed. Native code
/// publishes only its stored Own geometry; caller owns hidden state/history and
/// cursor publication. A failed publication must not resume the poisoned owner.
pub unsafe fn commit_e4b(path: &[i32]) -> Result<(), String> {
    if path.is_empty() || path.len() > 3 {
        return Err("E4B tree path length".into());
    }
    unsafe { check(imparo_cuda_tree_commit_e4b(path.as_ptr(), path.len() as u32)) }
}

/// # Safety
/// Exclusive closed target owner after successful verification; each descriptor
/// names a distinct real KV owner with its actual strides and ring mask. All
/// source rows are initialized. This moves opaque KV only; the caller owns tree,
/// hidden/recurrent state, history and cursor publication. Native preflights all
/// addresses before writes. A driver error poisons the owner and is not retryable.
pub unsafe fn commit_kv_rows(
    geometry: &[KvRowGeometry],
    from: &[u32],
    to: &[u32],
) -> Result<(), String> {
    if from.len() != to.len() || from.len() > 64 || geometry.is_empty() {
        return Err("tree KV publication geometry/row count".into());
    }
    let layers = u32::try_from(geometry.len()).map_err(|_| "tree KV owners overflow")?;
    unsafe {
        check(imparo_cuda_tree_commit_kv_rows(
            geometry.as_ptr(), layers, from.as_ptr(), to.as_ptr(), from.len() as u32,
        ))
    }
}
/// # Safety
/// Exclusive closed target owner with its actual KV allocations. False means
/// capacity/budget unavailable before any target-state writes; errors stay fatal.
pub unsafe fn prepare(
    start: u32,
    parents: &[i32],
    recurrent: u32,
) -> Result<bool, String> {
    let mut admitted = 0;
    unsafe {
        check(imparo_cuda_tree_prepare(
            start,
            parents.as_ptr(),
            parents.len() as u32,
            recurrent,
            &raw mut admitted,
        ))?;
    }
    match admitted {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err("invalid tree admission result".into()),
    }
}
/// # Safety
/// Exclusive current target owner; no other forward until commit/end. Full Q8
/// D64 LFM2 only. Caller validates model capacity, logical topology and rollback.
pub unsafe fn begin(start: u32, parents: &[i32], recurrent: u32) -> Result<(), String> {
    unsafe {
        check(imparo_cuda_tree_begin(
            start,
            parents.as_ptr(),
            parents.len() as u32,
            recurrent,
        ))
    }
}
/// # Safety
/// Same live transaction; all target rows and recurrent snapshots completed.
/// This publishes the selected path. Caller must update host cursor/history.
pub unsafe fn commit(path: &[i32]) -> Result<(), String> {
    unsafe {
        check(imparo_cuda_tree_commit(
            path.as_ptr(),
            path.len() as u32,
            512,
            25,
        ))
    }
}
/// # Safety
/// Discard only after draining work; this does not restore target state/KV.
pub unsafe fn end() -> Result<(), String> {
    unsafe { check(imparo_cuda_tree_end()) }
}

unsafe extern "C" {
    fn imparo_cuda_argmax_rows(src: u32, dst: u32, width: u32, rows: u32);
}
/// # Safety
/// Current owner has width*rows logits and rows output words; submitted on its stream.
pub unsafe fn argmax_rows(src: u32, dst: u32, width: u32, rows: u32) {
    unsafe {
        imparo_cuda_argmax_rows(src, dst, width, rows);
    }
}

unsafe extern "C" {
    fn imparo_cuda_tree_graph_prepare(action: *mut u32) -> i32;
    fn imparo_cuda_tree_graph_finish() -> i32;
    fn imparo_cuda_tree_graph_abort() -> i32;
}

/// Submission chosen by the current target owner's native tree graph contract.
pub enum TreeGraphSubmission {
    Ordinary,
    Capture(TreeGraphCapture),
    Replayed,
}

/// Cancels an unfinished capture before the caller's error reaches tree rollback.
/// The guard remains on the serialized owner thread until finish or unwinding.
#[must_use]
pub struct TreeGraphCapture {
    armed: bool,
    _owner_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

impl TreeGraphCapture {
    /// Finish capture and submit its first execution, before the ordinary end/read.
    pub fn finish(mut self) -> Result<(), String> {
        unsafe {
            check(imparo_cuda_tree_graph_finish())?;
        }
        self.armed = false;
        Ok(())
    }
}

impl Drop for TreeGraphCapture {
    fn drop(&mut self) {
        if self.armed {
            // Cleanup must not replace the error that caused this early return.
            let rc = unsafe { imparo_cuda_tree_graph_abort() };
            if rc != 0 {
                eprintln!("[tree-graph] capture cleanup failed rc={rc}");
            }
        }
    }
}

/// # Safety
/// Own the live static CUDA target/tree exclusively on this thread. Its forward
/// is open and Tokens has been uploaded; embedding has not been submitted. Keep
/// that owner until the returned capture guard is finished or dropped. Native
/// admission validates the actual tree, storage, feature owner and geometry.
pub unsafe fn graph_prepare() -> Result<TreeGraphSubmission, String> {
    // Also clean up a partial native prepare or an unrecognized action.
    let mut capture = TreeGraphCapture {
        armed: true,
        _owner_thread: std::marker::PhantomData,
    };
    let mut action = 0;
    unsafe {
        check(imparo_cuda_tree_graph_prepare(&raw mut action))?;
    }
    match action {
        0 => {
            capture.armed = false;
            Ok(TreeGraphSubmission::Ordinary)
        }
        1 => Ok(TreeGraphSubmission::Capture(capture)),
        2 => {
            capture.armed = false;
            Ok(TreeGraphSubmission::Replayed)
        }
        _ => Err(format!("invalid tree graph action {action}")),
    }
}
