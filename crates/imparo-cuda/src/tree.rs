//! Explicit laboratory tree transaction on the currently owned CUDA target.
unsafe extern "C" {
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
