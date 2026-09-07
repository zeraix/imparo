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
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q8_0: u32 = 8;
/// IMPARO-PRIVATE type id: Q8_0 values in the tile-major layout `imparo-repack` writes
/// (see docs/q8-tile-major-weights.md). GGUF's own ids stop far below 1000. Same bytes per
/// 32 elements as Q8_0 (34), so a converted tensor keeps its size and offset; only the
/// order of the bytes inside the tensor differs.
pub const GGML_Q8_0_TM: u32 = 1000;
/// IMPARO-PRIVATE: Q4_0 values in the same tile-major layout (18-byte blocks split into a
/// 16-byte payload and a 2-byte scale). No kernel reads it yet; the rule exists so the
/// converter, the load-time transform and the readers share one table (`TM_RULES`).
pub const GGML_Q4_0_TM: u32 = 1001;

/// Q4_0 geometry: 32 values per block, an f16 scale + 16 packed bytes.
pub const QK4_0: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Q8_0 geometry: 32 values per block, an f16 scale + 32 SIGNED bytes. No packing and
/// no -8 bias, unlike Q4_0 -- a byte is the value.
pub const QK8_0: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Q8_0_TM: the 8-row x 32-element unit, 272 bytes = the unit's eight half scales
/// (16 bytes, row order) followed by its eight 32-byte payload rows `[row 0..8][k 0..32]`.
/// Units are row-tile-major with K blocks adjacent. The scales sit INSIDE their unit
/// (changed 2026-09-04): kept in one array after the whole payload, the decode GEMV
/// streamed two regions per tensor and read 0.7..1.2% slower than row-major; one stream
/// per tensor removes that. The payload starts 16 bytes into the unit, so the prefill tile
/// loads stay 16-byte aligned; the tensor's size is the row-major size. Every address a
/// reader needs comes from `TmRule` so that no kernel, converter or oracle carries its own
/// copy of the rule; these two functions are the Q8 rule's addresses under the old names.
pub const Q8_0_TM_UNIT_ROWS: usize = 8;
/// Bytes per unit: scales + payload.
pub const Q8_0_TM_UNIT_BYTES: usize = Q8_0_TM_UNIT_ROWS * (2 + QK8_0);

/// Byte offset of `row`'s 32 int8 values for K block `block` inside a Q8_0_TM tensor whose
/// rows have `blocks` K blocks.
#[must_use]
pub const fn q8_0_tm_payload_offset(row: usize, block: usize, blocks: usize) -> usize {
    TM_RULES[0].payload_offset(row, block, blocks)
}

/// Byte offset of `row`'s half scale for K block `block`. `n_out` is unused since the
/// scales moved into their unit; kept so callers did not change.
#[must_use]
pub const fn q8_0_tm_scale_offset(
    row: usize,
    block: usize,
    blocks: usize,
    n_out: usize,
) -> usize {
    TM_RULES[0].scale_offset(row, block, blocks, n_out)
}

/// A Q8_0 tensor converts to Q8_0_TM only when its rows fill whole units and its row
/// width fills whole blocks; anything else keeps the row-major layout.
#[must_use]
pub const fn q8_0_tm_convertible(n_in: usize, n_out: usize) -> bool {
    n_in % QK8_0 == 0 && n_out % Q8_0_TM_UNIT_ROWS == 0
}

/// THE REPACK RULE TABLE. One tile-major rule per weight kind: eight row-major blocks of
/// `scale (2 bytes) + payload` (one row tile, one K block) become one unit of
/// `[8 scales][8 payload rows]`; units are row-tile-major with K blocks adjacent. Q8_0 has
/// 32 payload bytes per block (272-byte units), Q4_0 16 (144-byte units). The converter
/// (`imparo-repack`), the load-time transform (`Backend::transform_weights`) and the
/// verify step all take their addresses from here, so no consumer carries its own copy of
/// the layout; the Metal shader's `q8_tm_payload` / `q8_tm_scale` mirror these formulas
/// and are pinned bit-identical by the gates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TmRule {
    /// The row-major ggml type this rule converts from.
    pub from: u32,
    /// The tile-major type it produces.
    pub to: u32,
    pub from_name: &'static str,
    pub to_name: &'static str,
    /// Elements per block (32 for both Q4_0 and Q8_0).
    pub block_elems: usize,
    /// Payload bytes per block after the 2-byte scale is split off.
    pub payload_bytes: usize,
    /// Rows per unit.
    pub unit_rows: usize,
    /// Backends with kernels that read `to`. A rule with no readers is a layout the
    /// converter can write only on request (`--write-unread`) and the load-time transform
    /// never applies; a file carrying such a kind is refused at load by a backend that
    /// does not serve it, never misread.
    pub readers: &'static [&'static str],
}

impl TmRule {
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        2 + self.payload_bytes
    }
    /// Rows must fill whole units and the row width whole blocks.
    #[must_use]
    pub const fn convertible(&self, n_in: usize, n_out: usize) -> bool {
        n_in % self.block_elems == 0 && n_out % self.unit_rows == 0
    }
    /// Bytes per unit: the unit's scales, then its payload rows.
    #[must_use]
    pub const fn unit_bytes(&self) -> usize {
        self.unit_rows * (2 + self.payload_bytes)
    }
    const fn unit_start(&self, row: usize, block: usize, blocks: usize) -> usize {
        ((row / self.unit_rows) * blocks + block) * self.unit_bytes()
    }
    /// Byte offset of `row`'s payload for K block `block` (`blocks` K blocks per row):
    /// after the unit's scales.
    #[must_use]
    pub const fn payload_offset(
        &self,
        row: usize,
        block: usize,
        blocks: usize,
    ) -> usize {
        self.unit_start(row, block, blocks)
            + self.unit_rows * 2
            + (row % self.unit_rows) * self.payload_bytes
    }
    /// Byte offset of `row`'s half scale for K block `block`: at the head of its unit.
    /// `_n_out` is no longer needed by the layout; kept so the signature is stable.
    #[must_use]
    pub const fn scale_offset(
        &self,
        row: usize,
        block: usize,
        blocks: usize,
        _n_out: usize,
    ) -> usize {
        self.unit_start(row, block, blocks) + (row % self.unit_rows) * 2
    }
    /// Row-major bytes of one whole tensor -> tile-major bytes, same length.
    #[must_use]
    pub fn convert(&self, src: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
        let blocks = n_in / self.block_elems;
        let bb = self.block_bytes();
        let mut out = vec![0_u8; src.len()];
        for r in 0..n_out {
            for b in 0..blocks {
                let blk = &src[(r * blocks + b) * bb..][..bb];
                let p = self.payload_offset(r, b, blocks);
                out[p..p + self.payload_bytes].copy_from_slice(&blk[2..]);
                let s = self.scale_offset(r, b, blocks, n_out);
                out[s..s + 2].copy_from_slice(&blk[..2]);
            }
        }
        out
    }
    /// Every payload byte and every scale of `tm`, read back at the rule's address, must
    /// equal the row-major block's bytes. Byte equality, no float arithmetic.
    ///
    /// # Errors
    /// The first row/block that differs, by name.
    pub fn verify(
        &self,
        src: &[u8],
        tm: &[u8],
        n_in: usize,
        n_out: usize,
        name: &str,
    ) -> Result<(), String> {
        let blocks = n_in / self.block_elems;
        let bb = self.block_bytes();
        if tm.len() != src.len() {
            return Err(format!(
                "{name}: {} tile-major bytes for {} row-major",
                tm.len(),
                src.len()
            ));
        }
        for r in 0..n_out {
            for b in 0..blocks {
                let blk = &src[(r * blocks + b) * bb..][..bb];
                let p = self.payload_offset(r, b, blocks);
                if tm[p..p + self.payload_bytes] != blk[2..] {
                    return Err(format!("{name}: values differ at row {r} block {b}"));
                }
                let s = self.scale_offset(r, b, blocks, n_out);
                if tm[s..s + 2] != blk[..2] {
                    return Err(format!("{name}: scale differs at row {r} block {b}"));
                }
            }
        }
        Ok(())
    }
}

pub const TM_RULES: [TmRule; 2] = [
    TmRule {
        from: GGML_Q8_0,
        to: GGML_Q8_0_TM,
        from_name: "Q8_0",
        to_name: "Q8_0_TM",
        block_elems: QK8_0,
        payload_bytes: QK8_0,
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_Q4_0,
        to: GGML_Q4_0_TM,
        from_name: "Q4_0",
        to_name: "Q4_0_TM",
        block_elems: QK4_0,
        payload_bytes: QK4_0 / 2,
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &[],
    },
];

/// The rule that converts FROM this row-major type, if any.
#[must_use]
pub fn tm_rule_for(ggml_type: u32) -> Option<&'static TmRule> {
    TM_RULES.iter().find(|r| r.from == ggml_type)
}

/// The rule that produced this tile-major type, if any.
#[must_use]
pub fn tm_rule_to(ggml_type: u32) -> Option<&'static TmRule> {
    TM_RULES.iter().find(|r| r.to == ggml_type)
}

/// Tensors read by ROW keep the row-major layout: the embedding gather and the per-layer
/// token-embedding table (gemma4's PLE, staged by rows on the host) want a row's bytes
/// contiguous, and they never go through the GEMM. The role is the tensor's name, the
/// same on every model this engine loads.
pub const ROW_MAJOR_ROLES: &[&str] =
    &["token_embd.weight", "per_layer_token_embd.weight"];

/// Why a tensor keeps its layout, or the rule that converts it. The single decision the
/// converter and the load-time transform both make; `ne` is the tensor's dimensions
/// (ne[0] = row width, ne[1] = rows).
#[must_use]
pub fn tm_applies(
    name: &str,
    ggml_type: u32,
    ne: &[u64],
) -> Result<&'static TmRule, &'static str> {
    let Some(rule) = tm_rule_for(ggml_type) else {
        return Err("no tile-major rule for this type");
    };
    if ne.len() != 2 && !(ne.len() > 2 && ne[2..].iter().all(|&d| d == 1)) {
        return Err("not 2-D");
    }
    if ROW_MAJOR_ROLES.contains(&name) {
        return Err("read by row (embedding)");
    }
    if !rule.convertible(ne[0] as usize, ne[1] as usize) {
        return Err("rows not a multiple of 8 or width not a multiple of 32");
    }
    Ok(rule)
}

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
#[allow(non_camel_case_types)] // the variants are the GGUF type names, and the TM suffix is the layout
pub enum WeightKind {
    F32 = 0,
    Q4_0 = 1,
    Q8_0 = 2,
    /// Q8_0 values, tile-major (`GGML_Q8_0_TM`); written by `imparo-repack` or produced
    /// at load by the fast-tier transform.
    Q8_0_TM = 3,
    /// Q4_0 values, tile-major (`GGML_Q4_0_TM`). The rule exists; no backend reads it yet.
    Q4_0_TM = 4,
}

/// Names of every type `weight_kind` accepts, for the error message that rejects the
/// others. Kept beside the match because it had already gone stale once: it still read
/// "F32, Q4_0" after Q8_0 landed, in the message a user sees when a file is refused.
pub const SUPPORTED_WEIGHT_TYPES: &str =
    "F32, Q4_0, Q8_0, Q8_0_TM, Q4_0_TM (imparo-repack)";

/// Maps a GGUF ggml type id to the kernel table, or None for anything unsupported.
#[must_use]
pub fn weight_kind(ggml_type: u32) -> Option<WeightKind> {
    match ggml_type {
        GGML_F32 => Some(WeightKind::F32),
        GGML_Q4_0 => Some(WeightKind::Q4_0),
        GGML_Q8_0 => Some(WeightKind::Q8_0),
        GGML_Q8_0_TM => Some(WeightKind::Q8_0_TM),
        GGML_Q4_0_TM => Some(WeightKind::Q4_0_TM),
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
    source_path: PathBuf,
    full_file_sha256: OnceLock<[u8; 32]>,
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
            source_path: path.to_path_buf(),
            full_file_sha256: OnceLock::new(),
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
    /// Path originally used to open this mapping.
    ///
    /// This is provenance for diagnostics and discovery, not a request to reopen the
    /// pathname: identity is computed from the mapping already held by `Weights`.
    #[must_use]
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    /// SHA-256 of every byte in the held GGUF mapping.
    ///
    /// The first call touches the complete model and caches the result. Callers should
    /// therefore invoke it only after discovering a candidate tuned configuration that
    /// needs validation; ordinary startup performs no model-sized hashing pass.
    #[must_use]
    pub fn full_file_sha256(&self) -> [u8; 32] {
        *self.full_file_sha256.get_or_init(|| {
            // SAFETY: `base..base+len` is the immutable read-only mapping owned by self.
            let bytes = unsafe { std::slice::from_raw_parts(self.base, self.len) };
            Sha256::digest(bytes).into()
        })
    }

    /// Lowercase hexadecimal form of [`Self::full_file_sha256`].
    #[must_use]
    pub fn full_file_sha256_hex(&self) -> String {
        self.full_file_sha256().iter().fold(
            String::with_capacity(64),
            |mut out, byte| {
                use std::fmt::Write as _;
                let _ = write!(out, "{byte:02x}");
                out
            },
        )
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

#[cfg(test)]
mod identity_tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn minimal_gguf(marker: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(32, marker);
        bytes
    }

    fn temp_model(bytes: &[u8]) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "imparo-weights-identity-{}-{id}.gguf",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn full_identity_is_lazy_cached_and_uses_the_held_mapping() {
        let bytes = minimal_gguf(0xA5);
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        let path = temp_model(&bytes);
        {
            let weights = Weights::open(&path).unwrap();
            assert_eq!(weights.source_path(), path);
            assert!(weights.full_file_sha256.get().is_none());
            assert_eq!(weights.full_file_sha256(), expected);
            assert_eq!(weights.full_file_sha256.get(), Some(&expected));
            assert_eq!(weights.full_file_sha256_hex().len(), 64);
            assert_eq!(weights.full_file_sha256(), expected);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn identity_covers_bytes_outside_the_parsed_directory() {
        let first_path = temp_model(&minimal_gguf(0x11));
        let second_path = temp_model(&minimal_gguf(0x22));
        let first = Weights::open(&first_path).unwrap();
        let second = Weights::open(&second_path).unwrap();
        assert_ne!(first.full_file_sha256(), second.full_file_sha256());
        drop((first, second));
        fs::remove_file(first_path).unwrap();
        fs::remove_file(second_path).unwrap();
    }
}

#[cfg(test)]
mod tm_rule_tests {
    use super::*;

    /// Every rule round-trips a synthetic tensor: convert, verify byte-for-byte, and the
    /// two address functions cover the whole tensor exactly once.
    #[test]
    fn rules_round_trip_and_cover() {
        for rule in &TM_RULES {
            let (n_in, n_out) = (rule.block_elems * 3, rule.unit_rows * 2);
            let blocks = n_in / rule.block_elems;
            let bb = rule.block_bytes();
            let src: Vec<u8> = (0..n_out * blocks * bb)
                .map(|i| (i * 7 + 3) as u8)
                .collect();
            let tm = rule.convert(&src, n_in, n_out);
            rule.verify(&src, &tm, n_in, n_out, rule.to_name).unwrap();
            let mut hit = vec![0_u8; src.len()];
            for r in 0..n_out {
                for b in 0..blocks {
                    let p = rule.payload_offset(r, b, blocks);
                    for i in 0..rule.payload_bytes {
                        hit[p + i] += 1;
                    }
                    let s = rule.scale_offset(r, b, blocks, n_out);
                    hit[s] += 1;
                    hit[s + 1] += 1;
                }
            }
            assert!(
                hit.iter().all(|&h| h == 1),
                "{}: addresses do not tile the tensor",
                rule.to_name
            );
            assert!(rule.convertible(n_in, n_out));
            assert!(!rule.convertible(n_in + 1, n_out));
            assert!(!rule.convertible(n_in, n_out + 1));
        }
    }

    #[test]
    fn q8_rule_matches_the_legacy_functions() {
        let rule = tm_rule_for(GGML_Q8_0).unwrap();
        for (r, b, blocks, n_out) in [(0, 0, 4, 16), (7, 3, 4, 16), (9, 1, 8, 24)] {
            assert_eq!(
                rule.payload_offset(r, b, blocks),
                q8_0_tm_payload_offset(r, b, blocks)
            );
            assert_eq!(
                rule.scale_offset(r, b, blocks, n_out),
                q8_0_tm_scale_offset(r, b, blocks, n_out)
            );
        }
    }

    #[test]
    fn applies_by_role_and_shape() {
        assert!(tm_applies("blk.0.ffn_up.weight", GGML_Q8_0, &[2048, 8192]).is_ok());
        assert_eq!(
            tm_applies("blk.0.ffn_up.weight", GGML_Q4_0, &[2048, 8192])
                .unwrap()
                .to,
            GGML_Q4_0_TM
        );
        assert!(tm_applies("token_embd.weight", GGML_Q8_0, &[2048, 65536]).is_err());
        assert!(tm_applies("blk.0.attn_norm.weight", GGML_F32, &[2048]).is_err());
        assert!(tm_applies("blk.0.x.weight", GGML_Q8_0, &[2048, 12]).is_err());
    }
}
