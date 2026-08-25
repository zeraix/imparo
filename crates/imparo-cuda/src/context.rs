//! Device context: CUDA device + stream(s) + the weight blob + the BufId slot map.
//! The CUDA analogue of the Metal backend's process-global state. Fill with cudarc
//! or raw FFI -- the choice is the CUDA developer's; nothing outside this crate
//! depends on it.

pub struct CudaContext {
    // device handle, stream, weight blob device pointer, slot table:
    // slots: [DevicePtr; BufId::COUNT], kv_k/kv_v per layer, arena, ...
}

impl CudaContext {
    /// One context per process, created on first use (the Metal backend's shape).
    pub fn get() -> &'static Self {
        todo!("cuda: device init, weight upload, slot table")
    }
}
