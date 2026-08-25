//! Memory-mapped GGUF weight access: the mapping, the tensor index, and the
//! weight-type table every backend dispatches on.
//!
//! CONTAINER ONLY -- no compute. The CPU matmuls/dequant that read these mappings
//! live in imparo-cpu (`ops`); GPU backends receive `base_ptr()/byte_len()` via
//! `init_weights`. This module lived in imparo-cpu for a while, which made the "CPU
//! backend" crate the loader every backend depended on; it is container machinery
//! and this crate's README-stated job ("mmap, metadata spans, tensor index, quant
//! kinds"), so it lives here.
//!
//! Weights are mapped, never copied. A tensor is read where it lies in the file; the
//! only copies are the small f32 buffers a kernel writes into.

use std::collections::BTreeMap;
use std::path::Path;

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q8_0: u32 = 8;

/// Q4_0 geometry: 32 values per block, an f16 scale + 16 packed bytes.
pub const QK4_0: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Q8_0 geometry: 32 values per block, an f16 scale + 32 SIGNED bytes. No packing and
/// no -8 bias, unlike Q4_0 -- a byte is the value.
pub const QK8_0: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// The weight-type -> kernel table, as a type (task #17). Every matmul weight must map
/// to a variant; an unmapped ggml type is REJECTED AT LOAD with the tensor's name --
/// never silently misread by a kernel built for another layout. Adding a weight quant
/// means: a variant here, one dequant/stage implementation per backend, and a table
/// entry in each backend's dispatch -- no call-site changes.
///
/// The discriminants are the wire values the backends receive (imparo_metal_matmat's
/// `wkind`), so keep them stable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum WeightKind {
    F32 = 0,
    Q4_0 = 1,
    Q8_0 = 2,
}

/// Names of every type `weight_kind` accepts, for the error message that rejects the
/// others. Kept beside the match because it had already gone stale once: it still read
/// "F32, Q4_0" after Q8_0 landed, in the message a user sees when a file is refused.
pub const SUPPORTED_WEIGHT_TYPES: &str = "F32, Q4_0, Q8_0";

/// Maps a GGUF ggml type id to the kernel table, or None for anything unsupported.
#[must_use]
pub fn weight_kind(ggml_type: u32) -> Option<WeightKind> {
    match ggml_type {
        GGML_F32 => Some(WeightKind::F32),
        GGML_Q4_0 => Some(WeightKind::Q4_0),
        GGML_Q8_0 => Some(WeightKind::Q8_0),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub offset: usize,
    pub bytes: usize,
    pub ggml_type: u32,
    /// ne[0] is the fastest-varying dimension. For a projection weight this is the INPUT
    /// width and ne[1] the output width, so `mul_mat(W, x)` yields ne[1] values.
    pub ne: [u64; 4],
    pub n_dims: u32,
}

impl Tensor {
    #[must_use]
    pub fn ne0(&self) -> usize {
        self.ne[0] as usize
    }
    #[must_use]
    pub fn ne1(&self) -> usize {
        self.ne[1] as usize
    }
    #[must_use]
    pub fn elements(&self) -> usize {
        self.ne.iter().take(self.n_dims as usize).product::<u64>() as usize
    }
}

/// Maps an open file read-only, returning the base pointer. One implementation per
/// OS family, used by `Mapping` and `Weights` alike -- the unix mmap used to be
/// duplicated at both sites and the windows branch DID NOT EXIST, so the "portable"
/// crates could never have compiled on windows (caught preparing the first windows CI).
fn map_file(
    file: &std::fs::File,
    len: usize,
) -> Result<*const u8, Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        // SAFETY: fd is valid and open for reading; PROT_READ/MAP_PRIVATE is sound.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        Ok(p.cast::<u8>().cast_const())
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        // Minimal kernel32 FFI (repo rule: no binding-library dependencies).
        type Handle = *mut core::ffi::c_void;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CreateFileMappingW(
                file: Handle,
                attrs: *mut core::ffi::c_void,
                protect: u32,
                size_hi: u32,
                size_lo: u32,
                name: *const u16,
            ) -> Handle;
            fn MapViewOfFile(
                mapping: Handle,
                access: u32,
                off_hi: u32,
                off_lo: u32,
                len: usize,
            ) -> *mut core::ffi::c_void;
            fn CloseHandle(h: Handle) -> i32;
        }
        const PAGE_READONLY: u32 = 0x02;
        const FILE_MAP_READ: u32 = 0x04;
        let _ = len;
        // SAFETY: the handle is a valid open file; the mapping object is closed right
        // after the view is created (the view keeps the mapping alive on windows).
        unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle().cast(),
                std::ptr::null_mut(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return Err("CreateFileMapping failed".into());
            }
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            CloseHandle(mapping);
            if view.is_null() {
                return Err("MapViewOfFile failed".into());
            }
            Ok(view.cast::<u8>().cast_const())
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, len);
        Err("no file-mapping implementation for this OS".into())
    }
}

fn unmap_file(base: *const u8, len: usize) {
    #[cfg(unix)]
    // SAFETY: base/len came from a successful map_file and are unmapped exactly once.
    unsafe {
        libc::munmap(base.cast_mut().cast::<libc::c_void>(), len);
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn UnmapViewOfFile(base: *const core::ffi::c_void) -> i32;
        }
        let _ = len;
        // SAFETY: base came from MapViewOfFile and is unmapped exactly once.
        unsafe {
            UnmapViewOfFile(base.cast());
        }
    }
    #[cfg(not(any(unix, windows)))]
    let _ = (base, len);
}

/// A read-only file mapping.
///
/// File-backed pages are not charged to the process footprint: a mapped copy of the
/// weights costs disk and nothing measurable.
pub struct Mapping {
    base: *const u8,
    len: usize,
}

// SAFETY: the mapping is read-only and immutable for its whole lifetime.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Maps a file read-only.
    ///
    /// # Errors
    /// Returns an error when the file cannot be opened or mapped.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err("empty mapping".into());
        }
        Ok(Self {
            base: map_file(&file, len)?,
            len,
        })
    }

    #[must_use]
    pub fn base(&self) -> *const u8 {
        self.base
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unmap_file(self.base, self.len);
    }
}

pub struct Weights {
    base: *const u8,
    len: usize,
    pub tensors: BTreeMap<String, Tensor>,
    /// GPU dispatch is opt-in per run so the CPU path stays available as the oracle.
    gpu: bool,
}

// SAFETY: the mapping is read-only and lives for the object's lifetime; no interior mutation.
unsafe impl Send for Weights {}
unsafe impl Sync for Weights {}

impl Weights {
    /// Maps a GGUF file read-only and indexes its tensors by name.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened, mapped, or parsed.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let document = crate::read(path)?;
        Self::open_with(&document, path)
    }

    /// Maps the file using an already-parsed document.
    ///
    /// Parsing a 4 GB GGUF header costs ~700 ms, and the server had three callers doing it
    /// independently.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or mapped.
    pub fn open_with(
        document: &crate::Document,
        path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let base = map_file(&file, len)?;
        let mut tensors = BTreeMap::new();
        for t in &document.tensors {
            let mut ne = [1_u64; 4];
            for (i, d) in t.dimensions.iter().take(4).enumerate() {
                ne[i] = *d;
            }
            tensors.insert(
                t.name.clone(),
                Tensor {
                    offset: t.absolute_offset as usize,
                    bytes: t.byte_size as usize,
                    ggml_type: t.ggml_type,
                    ne,
                    n_dims: t.dimensions.len() as u32,
                },
            );
        }
        // Sharing/uploading the mapping to a GPU backend is the COMPOSITION ROOT's job
        // (imparo-model::backend + the bins/server), so this crate knows no backend
        // exists. The caller flips `gpu` via mark_gpu() after a backend's init_weights
        // succeeds.
        Ok(Self {
            base,
            len,
            tensors,
            gpu: false,
        })
    }

    /// CPU-visible base of the weight mapping, for a backend's init_weights.
    #[must_use]
    pub fn base_ptr(&self) -> *const u8 {
        self.base
    }
    /// Byte length of the mapping.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.len as u64
    }
    /// The composition root calls this once a GPU backend has taken the weights.
    pub fn mark_gpu(&mut self) {
        self.gpu = true;
    }

    /// True when a GPU backend holds these weights and the GPU path may run.
    #[must_use]
    pub fn gpu_enabled(&self) -> bool {
        self.gpu
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// A tensor's bytes, without copying.
    ///
    /// # Panics
    /// Panics when the tensor lies outside the mapping.
    #[must_use]
    pub fn raw(&self, t: &Tensor) -> &[u8] {
        assert!(
            t.offset + t.bytes <= self.len,
            "tensor {t:?} outside mapping"
        );
        // SAFETY: bounds checked above; mapping is read-only for the object's lifetime.
        unsafe { std::slice::from_raw_parts(self.base.add(t.offset), t.bytes) }
    }

    /// Reads an F32 tensor as a slice without copying.
    ///
    /// # Panics
    ///
    /// Panics when the tensor is not F32.
    #[must_use]
    // GGUF tensor data starts at the header's declared alignment (32 bytes), so
    // the f32 view is aligned by the container's own contract (cast_ptr_alignment).
    #[allow(clippy::cast_ptr_alignment)]
    pub fn f32s(&self, t: &Tensor) -> &[f32] {
        assert_eq!(t.ggml_type, GGML_F32, "expected F32 tensor");
        let bytes = self.raw(t);
        // SAFETY: GGUF guarantees 4-byte alignment for F32 tensor data via its alignment field.
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4)
        }
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        unmap_file(self.base, self.len);
    }
}

/// Round to IEEE binary16 and back, the precision a GPU KV cache carries.
///
/// Round to nearest, ties to even -- a truncating shift biases every value the same way,
/// which would make any comparison built on this meaningless.
#[must_use]
pub fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

#[must_use]
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007F_FFFF;
    let exp = ((x >> 23) & 0xFF) as i32;
    if exp == 0xFF {
        // inf or nan
        return sign | 0x7C00 | (u16::from(mant != 0) * 0x0200);
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    } // overflows to infinity
    if e <= 0 {
        if e < -10 {
            return sign;
        } // underflows to zero
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut out = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let mut m = mant >> 13;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    let mut ee = e as u32;
    if m == 0x400 {
        m = 0;
        ee += 1;
    }
    if ee >= 0x1F {
        return sign | 0x7C00;
    }
    sign | ((ee as u16) << 10) | m as u16
}

#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: renormalise
            let mut e = -1_i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            // 114 + e, not 113 + e. A subnormal is mant * 2^-24; after normalising with s
            // left shifts, value = (1+f) * 2^(-14-s), so the f32 exponent field is 113 - s,
            // and e = -1 - s makes that 114 + e. The old form returned exactly half.
            sign | (((127 - 15 + e + 2) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}
