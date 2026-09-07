#![doc = "Bounded, execution-independent GGUF subset importer for Zeraix."]

pub mod scan;
pub mod weights;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

const MAGIC: [u8; 4] = *b"GGUF";
const MAXIMUM_METADATA_ENTRIES: u64 = 1_000_000;
const MAXIMUM_TENSOR_ENTRIES: u64 = 10_000_000;
const MAXIMUM_STRING_BYTES: u64 = 256 * 1024 * 1024;
const MAXIMUM_DIMENSIONS: u32 = 4;
const ARRAY_PREVIEW_ELEMENTS: u64 = 16;
const MINIMUM_TENSOR_HEADER_BYTES: u64 = 32;
const MINIMUM_METADATA_ENTRY_BYTES: u64 = 12;
/// Maximum resident heap budget for the parsed tensor directory, including a
/// conservative allowance for vector entries, duplicate-name tracking,
/// dimensions, and container nodes before variable name bytes.
pub const MAXIMUM_TENSOR_DIRECTORY_RESIDENT_BYTES: u64 = 64 * 1024 * 1024;
const MINIMUM_TENSOR_ENTRY_RESIDENT_BYTES: u64 = 256;
/// Maximum resident heap budget for materialized metadata keys, scalar string
/// payloads, array previews, and conservative map/container overhead.
pub const MAXIMUM_METADATA_RESIDENT_BYTES: u64 = 64 * 1024 * 1024;
const MINIMUM_METADATA_ENTRY_RESIDENT_BYTES: u64 = 1024;

/// Test-only, thread-scoped coordination for placing a negative disturbance
/// immediately before one physical read. It cannot inject data or success.
#[cfg(test)]
mod test_coordination {
    use std::cell::RefCell;
    use std::io;
    #[cfg(unix)]
    use std::marker::PhantomData;
    #[cfg(unix)]
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    const TIMEOUT: Duration = Duration::from_secs(10);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Phase {
        Armed,
        ReaderLocated,
        Released,
        Failed,
    }

    #[derive(Debug)]
    struct State {
        reads_to_skip: AtomicU64,
        armed: AtomicBool,
        phase: Mutex<Phase>,
        changed: Condvar,
    }

    thread_local! {
        static ACTIVE: RefCell<Option<Arc<State>>> = const { RefCell::new(None) };
    }

    #[cfg(unix)]
    #[derive(Debug)]
    pub(super) struct Scope {
        state: Arc<State>,
        _not_send: PhantomData<Rc<()>>,
    }

    #[cfg(unix)]
    #[derive(Debug)]
    pub(super) struct Controller {
        state: Arc<State>,
    }

    #[cfg(unix)]
    pub(super) fn intercept_after(
        reads_to_skip: u64,
    ) -> io::Result<(Scope, Controller)> {
        let state = Arc::new(State {
            reads_to_skip: AtomicU64::new(reads_to_skip),
            armed: AtomicBool::new(true),
            phase: Mutex::new(Phase::Armed),
            changed: Condvar::new(),
        });
        ACTIVE.with(|active| {
            let mut active = active.borrow_mut();
            if active.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "nested positional-read test window",
                ));
            }
            *active = Some(Arc::clone(&state));
            Ok(())
        })?;
        Ok((
            Scope {
                state: Arc::clone(&state),
                _not_send: PhantomData,
            },
            Controller { state },
        ))
    }

    #[cfg(unix)]
    impl Controller {
        pub(super) fn wait_until_reader_located(&self) -> io::Result<()> {
            let phase = self.state.phase.lock().map_err(|_| {
                io::Error::other("positional-read test window poisoned")
            })?;
            let (phase, timeout) = self
                .state
                .changed
                .wait_timeout_while(phase, TIMEOUT, |phase| *phase == Phase::Armed)
                .map_err(|_| {
                    io::Error::other("positional-read test window poisoned")
                })?;
            if timeout.timed_out() || *phase != Phase::ReaderLocated {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "reader did not enter positional-read test window",
                ));
            }
            Ok(())
        }

        pub(super) fn release_reader(&self) -> io::Result<()> {
            let mut phase = self.state.phase.lock().map_err(|_| {
                io::Error::other("positional-read test window poisoned")
            })?;
            if *phase != Phase::ReaderLocated {
                *phase = Phase::Failed;
                self.state.changed.notify_all();
                return Err(io::Error::other("reader was not located before release"));
            }
            *phase = Phase::Released;
            self.state.changed.notify_all();
            Ok(())
        }
    }

    #[cfg(unix)]
    impl Drop for Scope {
        fn drop(&mut self) {
            ACTIVE.with(|active| {
                let observed = active.borrow_mut().take();
                debug_assert!(
                    observed
                        .as_ref()
                        .is_some_and(|observed| Arc::ptr_eq(observed, &self.state))
                );
            });
            let mut phase = match self.state.phase.lock() {
                Ok(phase) => phase,
                Err(poisoned) => poisoned.into_inner(),
            };
            if matches!(*phase, Phase::Armed | Phase::ReaderLocated) {
                *phase = Phase::Failed;
                self.state.changed.notify_all();
            }
        }
    }

    pub(super) fn before_physical_read() -> io::Result<()> {
        let state = ACTIVE.with(|active| active.borrow().as_ref().map(Arc::clone));
        let Some(state) = state else {
            return Ok(());
        };
        if !state.armed.load(Ordering::SeqCst) {
            return Ok(());
        }
        match state.reads_to_skip.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |remaining| remaining.checked_sub(1),
        ) {
            Ok(_) => return Ok(()),
            Err(0) => {}
            Err(_) => {
                return Err(io::Error::other("invalid positional-read skip counter"));
            }
        }
        if state
            .armed
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(());
        }
        let mut phase = state
            .phase
            .lock()
            .map_err(|_| io::Error::other("positional-read test window poisoned"))?;
        if *phase != Phase::Armed {
            return Err(io::Error::other("invalid positional-read test phase"));
        }
        *phase = Phase::ReaderLocated;
        state.changed.notify_all();
        let (mut phase, timeout) = state
            .changed
            .wait_timeout_while(phase, TIMEOUT, |phase| *phase == Phase::ReaderLocated)
            .map_err(|_| io::Error::other("positional-read test window poisoned"))?;
        if timeout.timed_out() || *phase != Phase::Released {
            *phase = Phase::Failed;
            state.changed.notify_all();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "positional-read test window was not released",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for ValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::UnknownValueType(value)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArraySummary {
    pub element_type: ValueType,
    pub element_count: u64,
    pub preview: Vec<Scalar>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterializedArray {
    pub element_type: ValueType,
    pub values: Vec<Scalar>,
    pub materialized_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Scalar(Scalar),
    Array(ArraySummary),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorLayout {
    pub block_elements: u64,
    pub block_bytes: u64,
    pub name: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub byte_size: u64,
    /// File offset of this tensor's `ggml_type` u32 inside the tensor-info table, so a
    /// converter can retype a tensor in place without re-serialising the header.
    pub type_field_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInventoryEntry {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub storage_type: String,
    pub elements: u64,
    pub block_elements: u64,
    pub block_bytes: u64,
    pub byte_size: u64,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub alignment: u64,
    pub role: String,
    pub layer: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInventory {
    pub entries: Vec<TensorInventoryEntry>,
    pub bytes_by_type: BTreeMap<String, u64>,
    pub bytes_by_role: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_size: u64,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
}

impl Document {
    #[must_use]
    pub fn string_value(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::Scalar(Scalar::String(value))) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn unsigned_value(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(MetadataValue::Scalar(Scalar::Unsigned(value))) => Some(*value),
            Some(MetadataValue::Scalar(Scalar::Signed(value))) => {
                u64::try_from(*value).ok()
            }
            _ => None,
        }
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

/// Produces a complete, deterministic tensor inventory before execution
/// capability admission. Unsupported execution programs may reject the model
/// later, but they cannot hide tensors after the first unfamiliar storage
/// type.
///
/// # Errors
///
/// Returns arithmetic overflow if dimensions or aggregate byte counts cannot
/// be represented.
pub fn tensor_inventory(document: &Document) -> Result<TensorInventory, GgufError> {
    let mut entries = Vec::with_capacity(document.tensors.len());
    let mut bytes_by_type = BTreeMap::<String, u64>::new();
    let mut bytes_by_role = BTreeMap::<String, u64>::new();
    for tensor in &document.tensors {
        let layout = tensor_layout(tensor.ggml_type)?;
        let elements = tensor
            .dimensions
            .iter()
            .try_fold(1_u64, |total, value| total.checked_mul(*value))
            .ok_or(GgufError::ArithmeticOverflow)?;
        let (role, layer) = tensor_role_and_layer(&tensor.name);
        let type_total = bytes_by_type.entry(layout.name.into()).or_default();
        *type_total = type_total
            .checked_add(tensor.byte_size)
            .ok_or(GgufError::ArithmeticOverflow)?;
        let role_total = bytes_by_role.entry(role.clone()).or_default();
        *role_total = role_total
            .checked_add(tensor.byte_size)
            .ok_or(GgufError::ArithmeticOverflow)?;
        entries.push(TensorInventoryEntry {
            name: tensor.name.clone(),
            dimensions: tensor.dimensions.clone(),
            ggml_type: tensor.ggml_type,
            storage_type: layout.name.into(),
            elements,
            block_elements: layout.block_elements,
            block_bytes: layout.block_bytes,
            byte_size: tensor.byte_size,
            relative_offset: tensor.relative_offset,
            absolute_offset: tensor.absolute_offset,
            alignment: document.alignment,
            role,
            layer,
        });
    }
    Ok(TensorInventory {
        entries,
        bytes_by_type,
        bytes_by_role,
    })
}

fn tensor_role_and_layer(name: &str) -> (String, Option<u32>) {
    if name == "token_embd.weight" {
        return ("token-embedding".into(), None);
    }
    if name == "output.weight" {
        return ("output-head".into(), None);
    }
    if name == "output_norm.weight" {
        return ("output-norm".into(), None);
    }
    let mut parts = name.split('.');
    let layer = match (parts.next(), parts.next()) {
        (Some("blk"), Some(value)) => value.parse::<u32>().ok(),
        _ => None,
    };
    let role = if name.contains(".attn_q_norm.") {
        "attention-query-norm"
    } else if name.contains(".attn_k_norm.") {
        "attention-key-norm"
    } else if name.contains(".attn_q.") {
        "attention-query"
    } else if name.contains(".attn_k.") {
        "attention-key"
    } else if name.contains(".attn_v.") {
        "attention-value"
    } else if name.contains(".attn_output.") {
        "attention-output"
    } else if name.contains(".attn_norm.") {
        "attention-norm"
    } else if name.contains(".ffn_gate_inp.") {
        "moe-router"
    } else if name.contains(".ffn_gate_exps.") {
        "moe-expert-gate"
    } else if name.contains(".ffn_up_exps.") {
        "moe-expert-up"
    } else if name.contains(".ffn_down_exps.") {
        "moe-expert-down"
    } else if name.contains(".ffn_gate.") {
        "ffn-gate"
    } else if name.contains(".ffn_up.") {
        "ffn-up"
    } else if name.contains(".ffn_down.") {
        "ffn-down"
    } else if name.contains(".ffn_norm.") {
        "ffn-norm"
    } else {
        "unclassified"
    };
    (role.into(), layer)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GgufError {
    Io(String),
    UnsafePath,
    SourceChangedDuringSnapshot,
    InvalidMagic,
    UnsupportedVersion(u32),
    TooManyMetadataEntries(u64),
    TooManyTensorEntries(u64),
    StringTooLarge(u64),
    UnknownValueType(u32),
    NestedArray,
    InvalidBoolean(u8),
    InvalidUtf8,
    DuplicateMetadata(String),
    DuplicateTensor(String),
    InvalidDimensionCount(u32),
    ZeroDimension,
    InvalidAlignment(u64),
    UnknownTensorType(u32),
    TensorBlockMismatch(String),
    TensorOffsetMisaligned(String),
    TensorOutOfBounds(String),
    TensorOverlap,
    MissingMetadata(String),
    MetadataNotArray(String),
    ArrayTooLarge(u64),
    MaterializedDataTooLarge(u64),
    ResourceLimit,
    ArithmeticOverflow,
}

impl Display for GgufError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Private logical reader over the snapshot's already-open file object.
///
/// Unix positional reads do not use or change the shared open-file-description
/// cursor. Windows `seek_read` uses the explicit offset to determine returned
/// bytes, although the operating system may still update the handle cursor.
/// This adapter's checked Rust cursor is the only parse cursor.
#[derive(Debug)]
struct HeldSnapshotPositionalReader<'file> {
    file: &'file File,
    logical_cursor: u64,
    length: u64,
}

impl Read for HeldSnapshotPositionalReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        #[cfg(test)]
        test_coordination::before_physical_read()?;
        #[cfg(unix)]
        let read = {
            use std::os::unix::fs::FileExt as _;
            self.file.read_at(buffer, self.logical_cursor)?
        };
        #[cfg(windows)]
        let read = {
            use std::os::windows::fs::FileExt as _;
            self.file.seek_read(buffer, self.logical_cursor)?
        };
        #[cfg(not(any(unix, windows)))]
        let read = {
            let _ = buffer;
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "held snapshot positional reads require Unix or Windows",
            ));
        };
        self.logical_cursor = self
            .logical_cursor
            .checked_add(u64::try_from(read).map_err(|_| {
                std::io::Error::other("held snapshot read length overflow")
            })?)
            .ok_or_else(|| {
                std::io::Error::other("held snapshot logical cursor overflow")
            })?;
        Ok(read)
    }
}

impl Seek for HeldSnapshotPositionalReader<'_> {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let target = match position {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::Current(delta) => {
                i128::from(self.logical_cursor) + i128::from(delta)
            }
            SeekFrom::End(delta) => i128::from(self.length) + i128::from(delta),
        };
        if !(0..=i128::from(u64::MAX)).contains(&target) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "held snapshot seek outside u64 domain",
            ));
        }
        self.logical_cursor = u64::try_from(target).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "held snapshot seek conversion",
            )
        })?;
        Ok(self.logical_cursor)
    }
}

fn snapshot_read_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        const FILE_SHARE_READ: u32 = 0x0000_0001;
        const FILE_SHARE_WRITE: u32 = 0x0000_0002;
        const FILE_SHARE_DELETE: u32 = 0x0000_0004;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        // Keep Unix-equivalent snapshot semantics on Windows: other processes
        // may replace or mutate the pathname while this identity-bound handle
        // remains valid. Every public read re-verifies the held handle/path and
        // fails closed on drift; denying sharing would hide those transitions
        // behind ERROR_SHARING_VIOLATION instead of exercising that contract.
        options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE);
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    options
}

/// Stable, opened-handle view of one GGUF object.
///
/// Parsing and subsequent Capsule import use [`Self::read_document`] and
/// [`Self::try_clone_file`], never a fresh pathname open. The source is opened
/// once without following the final link, hashed through that held handle, and
/// checked for size/content-metadata drift. No model-sized temporary copy is
/// created.
#[derive(Debug)]
pub struct VerifiedGgufSnapshot {
    original_path: PathBuf,
    length: u64,
    sha256: String,
    file: File,
    opened_metadata: fs::Metadata,
    logical_read_bytes: u64,
}

impl VerifiedGgufSnapshot {
    /// Opens a direct, non-reparse regular file and binds all subsequent reads
    /// to that one held object. The complete source is hashed through the held
    /// handle, but it is never copied to a temporary model-sized file.
    ///
    /// # Errors
    ///
    /// Rejects unsafe components, links/reparse points, non-regular files,
    /// short reads and observable source drift.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        require_direct_components(path)?;
        let options = snapshot_read_options();
        let mut source = options.open(path)?;
        let before = source.metadata()?;
        if !before.is_file() {
            return Err(GgufError::UnsafePath);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
            if before.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                return Err(GgufError::UnsafePath);
            }
        }
        require_direct_components(path)?;
        let path_probe = options.open(path)?;
        let probe_metadata = path_probe.metadata()?;
        if !same_opened_identity(&source, &before, &path_probe, &probe_metadata)? {
            return Err(GgufError::SourceChangedDuringSnapshot);
        }
        let length = before.len();
        let mut hasher = Sha256::new();
        let mut source_bytes_read = 0_u64;
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            #[cfg(test)]
            test_coordination::before_physical_read()?;
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            source_bytes_read = source_bytes_read
                .checked_add(
                    u64::try_from(read).map_err(|_| GgufError::ArithmeticOverflow)?,
                )
                .ok_or(GgufError::ArithmeticOverflow)?;
        }
        let after = source.metadata()?;
        if source_bytes_read != length || !same_handle_content_metadata(&before, &after)
        {
            return Err(GgufError::SourceChangedDuringSnapshot);
        }
        source.seek(SeekFrom::Start(0))?;
        Ok(Self {
            original_path: path.to_path_buf(),
            length,
            sha256: hasher.finalize().iter().fold(String::new(), |mut hex, b| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{b:02x}");
                hex
            }),
            file: source,
            opened_metadata: after,
            logical_read_bytes: source_bytes_read,
        })
    }

    #[must_use]
    pub fn original_path(&self) -> &Path {
        &self.original_path
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.original_path
    }

    #[must_use]
    pub const fn length(&self) -> u64 {
        self.length
    }

    /// Returns the source bytes read while establishing the content identity.
    #[must_use]
    pub const fn logical_read_bytes(&self) -> u64 {
        self.logical_read_bytes
    }

    /// Maximum transient copy buffer used by the opened-handle ingestion.
    #[must_use]
    pub const fn temporary_bytes(&self) -> u64 {
        1024 * 1024
    }

    /// Returns a raw clone of the already-open source object.
    ///
    /// The clone may share a kernel cursor with the snapshot or other clones.
    /// It is not an independent logical reader and cannot establish positive
    /// source or execution authority through a `seek`-then-`read` sequence.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the operating system cannot duplicate the
    /// handle. The original pathname is never reopened.
    pub fn try_clone_file(&self) -> Result<File, GgufError> {
        self.verify_held_content()?;
        Ok(self.file.try_clone()?)
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Parses through a positional reader over the already-open handle, never
    /// through a raw clone or the caller pathname.
    ///
    /// # Errors
    ///
    /// Returns bounded GGUF parse errors or snapshot length drift.
    pub fn read_document(&self) -> Result<Document, GgufError> {
        self.verify_snapshot_path()?;
        self.verify_held_content()?;
        let reader = HeldSnapshotPositionalReader {
            file: &self.file,
            logical_cursor: 0,
            length: self.length,
        };
        let document = read_from(BufReader::new(reader), self.length)?;
        self.verify_held_content()?;
        self.verify_snapshot_path()?;
        Ok(document)
    }

    /// Reads one exact range from the held object with an explicit offset.
    ///
    /// This is a physical range fact only. It cannot mint admission, a receipt,
    /// or positive execution authority. Content metadata is checked before and
    /// after every successful range materialization.
    ///
    /// # Errors
    ///
    /// Rejects arithmetic/bounds errors, short reads, unsupported platforms,
    /// or observable held-content drift.
    pub fn read_exact_at(
        &self,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<(), GgufError> {
        let length =
            u64::try_from(buffer.len()).map_err(|_| GgufError::ArithmeticOverflow)?;
        let end = offset
            .checked_add(length)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if end > self.length {
            return Err(GgufError::SourceChangedDuringSnapshot);
        }
        self.verify_held_content()?;
        let mut reader = HeldSnapshotPositionalReader {
            file: &self.file,
            logical_cursor: offset,
            length: self.length,
        };
        reader.read_exact(buffer)?;
        self.verify_held_content()?;
        Ok(())
    }

    /// Hashes one exact held-object range with positional reads and bounded
    /// memory. The result is a physical content fact only, not admission or
    /// positive execution authority.
    ///
    /// # Errors
    ///
    /// Rejects arithmetic/bounds errors, short reads, unsupported platforms,
    /// or observable held-content drift.
    pub fn sha256_range(
        &self,
        offset: u64,
        length: u64,
    ) -> Result<[u8; 32], GgufError> {
        let end = offset
            .checked_add(length)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if end > self.length {
            return Err(GgufError::SourceChangedDuringSnapshot);
        }
        self.verify_held_content()?;
        let mut reader = HeldSnapshotPositionalReader {
            file: &self.file,
            logical_cursor: offset,
            length: self.length,
        };
        let mut remaining = length;
        let mut buffer = vec![0_u8; 1024 * 1024];
        let maximum_chunk =
            u64::try_from(buffer.len()).map_err(|_| GgufError::ArithmeticOverflow)?;
        let mut hash = Sha256::new();
        while remaining > 0 {
            let count = usize::try_from(remaining.min(maximum_chunk))
                .map_err(|_| GgufError::ArithmeticOverflow)?;
            reader.read_exact(&mut buffer[..count])?;
            hash.update(&buffer[..count]);
            remaining -=
                u64::try_from(count).map_err(|_| GgufError::ArithmeticOverflow)?;
        }
        self.verify_held_content()?;
        Ok(hash.finalize().into())
    }

    /// Revalidates size and same-handle content metadata without reopening the
    /// caller pathname.
    ///
    /// # Errors
    ///
    /// Rejects in-place truncation or mutation of the held object.
    pub fn verify_held_content(&self) -> Result<(), GgufError> {
        let metadata = self.file.metadata()?;
        if !metadata.is_file()
            || metadata.len() != self.length
            || !same_handle_content_metadata(&self.opened_metadata, &metadata)
        {
            return Err(GgufError::SourceChangedDuringSnapshot);
        }
        Ok(())
    }

    /// Revalidates that the caller pathname still denotes the held object.
    ///
    /// # Errors
    ///
    /// Rejects replacement, truncation, symlink/reparse substitution, or a
    /// path that no longer names the opened snapshot object.
    pub fn verify_snapshot_path(&self) -> Result<(), GgufError> {
        require_direct_components(&self.original_path)?;
        let options = snapshot_read_options();
        let probe = options.open(&self.original_path)?;
        let held = self.file.metadata()?;
        let observed = probe.metadata()?;
        if held.len() != self.length
            || observed.len() != self.length
            || !same_opened_identity(&self.file, &held, &probe, &observed)?
        {
            return Err(GgufError::SourceChangedDuringSnapshot);
        }
        Ok(())
    }

    /// Materializes one bounded metadata array from the held object.
    /// `maximum_materialized_bytes` bounds resident `Scalar` vector storage
    /// plus owned string heap bytes, not merely encoded payload length.
    ///
    /// # Errors
    ///
    /// Returns the same schema and bound errors as [`read_metadata_array`]
    /// without reopening the caller pathname.
    pub fn read_metadata_array(
        &self,
        key: &str,
        maximum_elements: u64,
        maximum_materialized_bytes: u64,
    ) -> Result<MaterializedArray, GgufError> {
        let document = self.read_document()?;
        let summary = match document.metadata.get(key) {
            Some(MetadataValue::Array(summary)) => summary,
            Some(_) => return Err(GgufError::MetadataNotArray(key.into())),
            None => return Err(GgufError::MissingMetadata(key.into())),
        };
        if summary.element_count > maximum_elements {
            return Err(GgufError::ArrayTooLarge(summary.element_count));
        }
        let reader = HeldSnapshotPositionalReader {
            file: &self.file,
            logical_cursor: 0,
            length: self.length,
        };
        let result = (|| {
            let mut parser = Parser::new(BufReader::new(reader), self.length);
            let (_, _, metadata_count) = parser.header()?;
            let mut metadata_resident =
                parser.metadata_resident_baseline(metadata_count)?;
            for _ in 0..metadata_count {
                let observed_key = parser.metadata_string(&mut metadata_resident)?;
                let value_type = ValueType::try_from(parser.u32()?)?;
                if observed_key == key {
                    if value_type != ValueType::Array {
                        return Err(GgufError::MetadataNotArray(key.into()));
                    }
                    return parser.materialized_array(
                        maximum_elements,
                        maximum_materialized_bytes,
                    );
                }
                let _ = parser.metadata_value(value_type, &mut metadata_resident)?;
            }
            Err(GgufError::MissingMetadata(key.into()))
        })()?;
        self.verify_held_content()?;
        Ok(result)
    }
}

#[cfg(unix)]
#[allow(clippy::unnecessary_wraps)]
fn same_opened_identity(
    _source: &File,
    source_metadata: &fs::Metadata,
    _probe: &File,
    probe_metadata: &fs::Metadata,
) -> Result<bool, GgufError> {
    use std::os::unix::fs::MetadataExt;
    Ok(source_metadata.dev() == probe_metadata.dev()
        && source_metadata.ino() == probe_metadata.ino())
}

#[cfg(windows)]
fn same_opened_identity(
    source: &File,
    _source_metadata: &fs::Metadata,
    probe: &File,
    _probe_metadata: &fs::Metadata,
) -> Result<bool, GgufError> {
    // Inlined from the purged zeraix-windows-system (a 21k-line crate serving only
    // this call): two open handles name the same file iff volume serial + file index
    // agree, via GetFileInformationByHandle. Kept dependency-free.
    use std::os::windows::io::AsRawHandle;
    #[repr(C)]
    #[derive(Default)]
    struct Info {
        attrs: u32,
        ct: [u32; 2],
        at: [u32; 2],
        wt: [u32; 2],
        vol: u32,
        size_hi: u32,
        size_lo: u32,
        links: u32,
        idx_hi: u32,
        idx_lo: u32,
    }
    unsafe extern "system" {
        fn GetFileInformationByHandle(h: *mut core::ffi::c_void, out: *mut Info)
        -> i32;
    }
    let ident = |f: &File| -> Result<(u32, u32, u32), GgufError> {
        let mut i = Info::default();
        // SAFETY: a valid open handle and an out-struct of the documented layout.
        if unsafe { GetFileInformationByHandle(f.as_raw_handle().cast(), &raw mut i) }
            == 0
        {
            return Err(GgufError::Io(std::io::Error::last_os_error().to_string()));
        }
        Ok((i.vol, i.idx_hi, i.idx_lo))
    };
    Ok(ident(source)? == ident(probe)?)
}

#[cfg(not(any(unix, windows)))]
fn same_opened_identity(
    _source: &File,
    source_metadata: &fs::Metadata,
    _probe: &File,
    probe_metadata: &fs::Metadata,
) -> Result<bool, GgufError> {
    Ok(source_metadata.len() == probe_metadata.len())
}

#[cfg(unix)]
fn same_handle_content_metadata(before: &fs::Metadata, after: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.len() == after.len()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
}

#[cfg(windows)]
fn same_handle_content_metadata(before: &fs::Metadata, after: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    before.file_size() == after.file_size()
        && before.last_write_time() == after.last_write_time()
}

#[cfg(not(any(unix, windows)))]
fn same_handle_content_metadata(before: &fs::Metadata, after: &fs::Metadata) -> bool {
    before.len() == after.len() && before.modified().ok() == after.modified().ok()
}

fn require_direct_components(path: &Path) -> Result<(), GgufError> {
    let mut current = if path.is_absolute() {
        path.ancestors()
            .last()
            .ok_or(GgufError::UnsafePath)?
            .to_path_buf()
    } else {
        std::env::current_dir()?
    };
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => continue,
            Component::CurDir | Component::ParentDir => {
                return Err(GgufError::UnsafePath);
            }
            Component::Normal(value) => current.push(value),
        }
        let metadata = fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() {
            return Err(GgufError::UnsafePath);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
            if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                return Err(GgufError::UnsafePath);
            }
        }
    }
    Ok(())
}

/// Reads and validates a GGUF directory without reading tensor payloads.
///
/// # Errors
///
/// Returns an error for I/O failure, malformed metadata, unsupported types,
/// unsafe counts, invalid tensor geometry, overlap, or out-of-file spans.
pub fn read(path: impl AsRef<Path>) -> Result<Document, GgufError> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    read_from(BufReader::new(file), file_size)
}

/// Materializes one validated metadata array under explicit memory bounds.
///
/// The GGUF directory is first validated in full, then rescanned to load only
/// the requested array. `maximum_materialized_bytes` bounds resident `Scalar`
/// vector storage plus owned string heap bytes. Tensor payloads are never read.
///
/// # Errors
///
/// Returns an error for any [`read`] validation failure, a missing or non-array
/// key, an excessive element count, or an excessive decoded byte count.
pub fn read_metadata_array(
    path: impl AsRef<Path>,
    key: &str,
    maximum_elements: u64,
    maximum_materialized_bytes: u64,
) -> Result<MaterializedArray, GgufError> {
    let path = path.as_ref();
    let document = read(path)?;
    let summary = match document.metadata.get(key) {
        Some(MetadataValue::Array(summary)) => summary,
        Some(_) => return Err(GgufError::MetadataNotArray(key.into())),
        None => return Err(GgufError::MissingMetadata(key.into())),
    };
    if summary.element_count > maximum_elements {
        return Err(GgufError::ArrayTooLarge(summary.element_count));
    }

    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut parser = Parser::new(BufReader::new(file), file_size);
    let (_, _, metadata_count) = parser.header()?;
    let mut metadata_resident = parser.metadata_resident_baseline(metadata_count)?;
    for _ in 0..metadata_count {
        let observed_key = parser.metadata_string(&mut metadata_resident)?;
        let value_type = ValueType::try_from(parser.u32()?)?;
        if observed_key == key {
            if value_type != ValueType::Array {
                return Err(GgufError::MetadataNotArray(key.into()));
            }
            return parser
                .materialized_array(maximum_elements, maximum_materialized_bytes);
        }
        let _ = parser.metadata_value(value_type, &mut metadata_resident)?;
    }
    Err(GgufError::MissingMetadata(key.into()))
}

/// Parses a GGUF stream with a declared total byte length.
///
/// # Errors
///
/// Returns the same validation failures as [`read`].
pub fn read_from<R: Read + Seek>(
    reader: R,
    file_size: u64,
) -> Result<Document, GgufError> {
    Parser::new(reader, file_size).parse()
}

/// Resolves the storage geometry for a supported GGML tensor type.
///
/// # Errors
///
/// Returns [`GgufError::UnknownTensorType`] when the type is not part of the
/// independently implemented and reviewed layout table.
pub fn tensor_layout(ggml_type: u32) -> Result<TensorLayout, GgufError> {
    let (block_elements, block_bytes, name) = match ggml_type {
        0 => (1, 4, "F32"),
        1 => (1, 2, "F16"),
        2 => (32, 18, "Q4_0"),
        3 => (32, 20, "Q4_1"),
        6 => (32, 22, "Q5_0"),
        7 => (32, 24, "Q5_1"),
        8 => (32, 34, "Q8_0"),
        // imparo-private: Q8_0 bytes in the tile-major order imparo-repack writes; the
        // block arithmetic (and so every size and bounds check) is Q8_0's.
        1000 => (32, 34, "Q8_0_TM"),
        1001 => (32, 18, "Q4_0_TM"),
        9 => (32, 40, "Q8_1"),
        10 => (256, 84, "Q2_K"),
        11 => (256, 110, "Q3_K"),
        12 => (256, 144, "Q4_K"),
        13 => (256, 176, "Q5_K"),
        14 => (256, 210, "Q6_K"),
        15 => (256, 292, "Q8_K"),
        16 => (256, 66, "IQ2_XXS"),
        17 => (256, 74, "IQ2_XS"),
        18 => (256, 98, "IQ3_XXS"),
        19 => (256, 50, "IQ1_S"),
        20 => (32, 18, "IQ4_NL"),
        21 => (256, 110, "IQ3_S"),
        22 => (256, 82, "IQ2_S"),
        23 => (256, 136, "IQ4_XS"),
        24 => (1, 1, "I8"),
        25 => (1, 2, "I16"),
        26 => (1, 4, "I32"),
        27 => (1, 8, "I64"),
        28 => (1, 8, "F64"),
        29 => (256, 56, "IQ1_M"),
        30 => (1, 2, "BF16"),
        34 => (256, 54, "TQ1_0"),
        35 => (256, 66, "TQ2_0"),
        39 => (32, 17, "MXFP4"),
        40 => (64, 36, "NVFP4"),
        41 => (128, 18, "Q1_0"),
        42 => (64, 18, "Q2_0"),
        _ => return Err(GgufError::UnknownTensorType(ggml_type)),
    };
    Ok(TensorLayout {
        block_elements,
        block_bytes,
        name,
    })
}

type TensorHeader = (String, Vec<u64>, u32, u64, u64);

struct Parser<R> {
    reader: R,
    file_size: u64,
}

impl<R: Read + Seek> Parser<R> {
    const fn new(reader: R, file_size: u64) -> Self {
        Self { reader, file_size }
    }

    fn parse(mut self) -> Result<Document, GgufError> {
        let (version, tensor_count, metadata_count) = self.header()?;
        let metadata = self.metadata(metadata_count)?;
        let tensor_headers = self.tensor_headers(tensor_count)?;
        self.finish_document(version, metadata, tensor_headers)
    }

    fn header(&mut self) -> Result<(u32, u64, u64), GgufError> {
        if self.bytes::<4>()? != MAGIC {
            return Err(GgufError::InvalidMagic);
        }
        let version = self.u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = self.u64()?;
        let metadata_count = self.u64()?;
        if metadata_count > MAXIMUM_METADATA_ENTRIES {
            return Err(GgufError::TooManyMetadataEntries(metadata_count));
        }
        if tensor_count > MAXIMUM_TENSOR_ENTRIES {
            return Err(GgufError::TooManyTensorEntries(tensor_count));
        }
        Ok((version, tensor_count, metadata_count))
    }

    fn metadata(
        &mut self,
        metadata_count: u64,
    ) -> Result<BTreeMap<String, MetadataValue>, GgufError> {
        let mut resident_bytes = self.metadata_resident_baseline(metadata_count)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = self.metadata_string(&mut resident_bytes)?;
            if metadata.contains_key(&key) {
                return Err(GgufError::DuplicateMetadata(key));
            }
            let value_type = ValueType::try_from(self.u32()?)?;
            let value = self.metadata_value(value_type, &mut resident_bytes)?;
            metadata.insert(key, value);
        }
        Ok(metadata)
    }

    fn metadata_resident_baseline(
        &mut self,
        metadata_count: u64,
    ) -> Result<u64, GgufError> {
        let maximum_from_file = self.remaining_bytes()? / MINIMUM_METADATA_ENTRY_BYTES;
        if metadata_count > maximum_from_file {
            return Err(GgufError::TooManyMetadataEntries(metadata_count));
        }
        let resident_bytes = metadata_count
            .checked_mul(MINIMUM_METADATA_ENTRY_RESIDENT_BYTES)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if resident_bytes > MAXIMUM_METADATA_RESIDENT_BYTES {
            return Err(GgufError::TooManyMetadataEntries(metadata_count));
        }
        Ok(resident_bytes)
    }

    fn metadata_string(
        &mut self,
        resident_bytes: &mut u64,
    ) -> Result<String, GgufError> {
        let remaining_budget = MAXIMUM_METADATA_RESIDENT_BYTES
            .checked_sub(*resident_bytes)
            .ok_or(GgufError::ResourceLimit)?;
        let value = self.string_with_resident_budget(remaining_budget)?;
        *resident_bytes = resident_bytes
            .checked_add(
                u64::try_from(value.len())
                    .map_err(|_| GgufError::ArithmeticOverflow)?,
            )
            .ok_or(GgufError::ArithmeticOverflow)?;
        Ok(value)
    }

    fn tensor_headers(
        &mut self,
        tensor_count: u64,
    ) -> Result<Vec<TensorHeader>, GgufError> {
        let maximum_from_file = self.remaining_bytes()? / MINIMUM_TENSOR_HEADER_BYTES;
        if tensor_count > maximum_from_file {
            return Err(GgufError::TooManyTensorEntries(tensor_count));
        }
        let baseline_resident_bytes = tensor_count
            .checked_mul(MINIMUM_TENSOR_ENTRY_RESIDENT_BYTES)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if baseline_resident_bytes > MAXIMUM_TENSOR_DIRECTORY_RESIDENT_BYTES {
            return Err(GgufError::TooManyTensorEntries(tensor_count));
        }
        let tensor_capacity =
            usize::try_from(tensor_count).map_err(|_| GgufError::ArithmeticOverflow)?;
        let mut tensor_headers = Vec::new();
        tensor_headers
            .try_reserve_exact(tensor_capacity)
            .map_err(|_| GgufError::ResourceLimit)?;
        let mut names = HashSet::new();
        names
            .try_reserve(tensor_capacity)
            .map_err(|_| GgufError::ResourceLimit)?;
        let mut resident_bytes = baseline_resident_bytes;
        for _ in 0..tensor_count {
            let remaining_name_budget = MAXIMUM_TENSOR_DIRECTORY_RESIDENT_BYTES
                .checked_sub(resident_bytes)
                .ok_or(GgufError::ResourceLimit)?;
            let name = self.string_with_resident_budget(remaining_name_budget / 2)?;
            let name_resident_bytes = u64::try_from(name.len())
                .map_err(|_| GgufError::ArithmeticOverflow)?
                .checked_mul(2)
                .ok_or(GgufError::ArithmeticOverflow)?;
            resident_bytes = resident_bytes
                .checked_add(name_resident_bytes)
                .ok_or(GgufError::ArithmeticOverflow)?;
            if resident_bytes > MAXIMUM_TENSOR_DIRECTORY_RESIDENT_BYTES {
                return Err(GgufError::ResourceLimit);
            }
            let mut tracked_name = String::new();
            tracked_name
                .try_reserve_exact(name.len())
                .map_err(|_| GgufError::ResourceLimit)?;
            tracked_name.push_str(&name);
            if !names.insert(tracked_name) {
                return Err(GgufError::DuplicateTensor(name));
            }
            let dimension_count = self.u32()?;
            if dimension_count == 0 || dimension_count > MAXIMUM_DIMENSIONS {
                return Err(GgufError::InvalidDimensionCount(dimension_count));
            }
            let dimension_capacity = usize::try_from(dimension_count)
                .map_err(|_| GgufError::ArithmeticOverflow)?;
            let mut dimensions = Vec::new();
            dimensions
                .try_reserve_exact(dimension_capacity)
                .map_err(|_| GgufError::ResourceLimit)?;
            for _ in 0..dimension_count {
                let dimension = self.u64()?;
                if dimension == 0 {
                    return Err(GgufError::ZeroDimension);
                }
                dimensions.push(dimension);
            }
            let type_field_offset = self.position()?;
            let ggml_type = self.u32()?;
            let relative_offset = self.u64()?;
            tensor_headers.push((
                name,
                dimensions,
                ggml_type,
                relative_offset,
                type_field_offset,
            ));
        }
        Ok(tensor_headers)
    }

    fn finish_document(
        &mut self,
        version: u32,
        metadata: BTreeMap<String, MetadataValue>,
        tensor_headers: Vec<TensorHeader>,
    ) -> Result<Document, GgufError> {
        let alignment = metadata
            .get("general.alignment")
            .and_then(|value| match value {
                MetadataValue::Scalar(Scalar::Unsigned(value)) => Some(*value),
                MetadataValue::Scalar(Scalar::Signed(value)) => {
                    u64::try_from(*value).ok()
                }
                _ => None,
            })
            .unwrap_or(32);
        let directory_end = self.position()?;
        let data_offset = align_up(directory_end, alignment)?;
        if data_offset > self.file_size {
            return Err(GgufError::TensorOutOfBounds("data section".into()));
        }

        let mut tensors = Vec::with_capacity(tensor_headers.len());
        for (name, dimensions, ggml_type, relative_offset, type_field_offset) in
            tensor_headers
        {
            if relative_offset % alignment != 0 {
                return Err(GgufError::TensorOffsetMisaligned(name));
            }
            let layout = tensor_layout(ggml_type)?;
            let elements = dimensions
                .iter()
                .try_fold(1_u64, |total, dimension| total.checked_mul(*dimension))
                .ok_or(GgufError::ArithmeticOverflow)?;
            if elements % layout.block_elements != 0 {
                return Err(GgufError::TensorBlockMismatch(name));
            }
            let byte_size = elements
                .checked_div(layout.block_elements)
                .and_then(|blocks| blocks.checked_mul(layout.block_bytes))
                .ok_or(GgufError::ArithmeticOverflow)?;
            let absolute_offset = data_offset
                .checked_add(relative_offset)
                .ok_or(GgufError::ArithmeticOverflow)?;
            let end = absolute_offset
                .checked_add(byte_size)
                .ok_or(GgufError::ArithmeticOverflow)?;
            if end > self.file_size {
                return Err(GgufError::TensorOutOfBounds(name));
            }
            tensors.push(TensorInfo {
                name,
                dimensions,
                ggml_type,
                relative_offset,
                absolute_offset,
                byte_size,
                type_field_offset,
            });
        }
        validate_non_overlapping(&tensors)?;
        Ok(Document {
            version,
            alignment,
            data_offset,
            file_size: self.file_size,
            metadata,
            tensors,
        })
    }

    fn metadata_value(
        &mut self,
        value_type: ValueType,
        resident_bytes: &mut u64,
    ) -> Result<MetadataValue, GgufError> {
        if value_type != ValueType::Array {
            let value = if value_type == ValueType::String {
                Scalar::String(self.metadata_string(resident_bytes)?)
            } else {
                self.scalar(value_type)?
            };
            return Ok(MetadataValue::Scalar(value));
        }
        let element_type = ValueType::try_from(self.u32()?)?;
        if element_type == ValueType::Array {
            return Err(GgufError::NestedArray);
        }
        let element_count = self.u64()?;
        self.require_scalar_count_fits_remaining(element_type, element_count)?;
        let preview_count = element_count.min(ARRAY_PREVIEW_ELEMENTS);
        let mut preview = Vec::with_capacity(
            usize::try_from(preview_count)
                .map_err(|_| GgufError::ArithmeticOverflow)?,
        );
        for _ in 0..preview_count {
            let value = if element_type == ValueType::String {
                Scalar::String(self.metadata_string(resident_bytes)?)
            } else {
                self.scalar(element_type)?
            };
            preview.push(value);
        }
        self.skip_scalars(element_type, element_count - preview_count)?;
        Ok(MetadataValue::Array(ArraySummary {
            element_type,
            element_count,
            preview,
        }))
    }

    fn materialized_array(
        &mut self,
        maximum_elements: u64,
        maximum_materialized_bytes: u64,
    ) -> Result<MaterializedArray, GgufError> {
        let element_type = ValueType::try_from(self.u32()?)?;
        if element_type == ValueType::Array {
            return Err(GgufError::NestedArray);
        }
        let element_count = self.u64()?;
        if element_count > maximum_elements {
            return Err(GgufError::ArrayTooLarge(element_count));
        }
        self.require_scalar_count_fits_remaining(element_type, element_count)?;
        let minimum_materialized_bytes = u64::try_from(std::mem::size_of::<Scalar>())
            .map_err(|_| GgufError::ArithmeticOverflow)?;
        let conservative_materialized_bytes = element_count
            .checked_mul(minimum_materialized_bytes)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if conservative_materialized_bytes > maximum_materialized_bytes {
            return Err(GgufError::MaterializedDataTooLarge(
                conservative_materialized_bytes,
            ));
        }
        let capacity = usize::try_from(element_count)
            .map_err(|_| GgufError::ArithmeticOverflow)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(capacity)
            .map_err(|_| GgufError::ResourceLimit)?;
        let mut materialized_bytes = conservative_materialized_bytes;
        for _ in 0..element_count {
            let value = if element_type == ValueType::String {
                let remaining_budget = maximum_materialized_bytes
                    .checked_sub(materialized_bytes)
                    .ok_or(GgufError::MaterializedDataTooLarge(materialized_bytes))?;
                Scalar::String(self.string_with_resident_budget(remaining_budget)?)
            } else {
                self.scalar(element_type)?
            };
            materialized_bytes = materialized_bytes
                .checked_add(scalar_materialized_heap_bytes(&value))
                .ok_or(GgufError::ArithmeticOverflow)?;
            if materialized_bytes > maximum_materialized_bytes {
                return Err(GgufError::MaterializedDataTooLarge(materialized_bytes));
            }
            values.push(value);
        }
        Ok(MaterializedArray {
            element_type,
            values,
            materialized_bytes,
        })
    }

    fn scalar(&mut self, value_type: ValueType) -> Result<Scalar, GgufError> {
        match value_type {
            ValueType::Uint8 => Ok(Scalar::Unsigned(u64::from(self.u8()?))),
            ValueType::Int8 => Ok(Scalar::Signed(i64::from(self.i8()?))),
            ValueType::Uint16 => Ok(Scalar::Unsigned(u64::from(self.u16()?))),
            ValueType::Int16 => Ok(Scalar::Signed(i64::from(self.i16()?))),
            ValueType::Uint32 => Ok(Scalar::Unsigned(u64::from(self.u32()?))),
            ValueType::Int32 => Ok(Scalar::Signed(i64::from(self.i32()?))),
            ValueType::Float32 => Ok(Scalar::Float(f64::from(self.f32()?))),
            ValueType::Bool => match self.u8()? {
                0 => Ok(Scalar::Bool(false)),
                1 => Ok(Scalar::Bool(true)),
                value => Err(GgufError::InvalidBoolean(value)),
            },
            ValueType::String => Ok(Scalar::String(self.string()?)),
            ValueType::Uint64 => Ok(Scalar::Unsigned(self.u64()?)),
            ValueType::Int64 => Ok(Scalar::Signed(self.i64()?)),
            ValueType::Float64 => Ok(Scalar::Float(self.f64()?)),
            ValueType::Array => Err(GgufError::NestedArray),
        }
    }

    fn skip_scalars(
        &mut self,
        value_type: ValueType,
        count: u64,
    ) -> Result<(), GgufError> {
        if value_type == ValueType::String {
            for _ in 0..count {
                let length = self.u64()?;
                self.skip_bounded(length)?;
            }
            return Ok(());
        }
        let scalar_size = match value_type {
            ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => 1,
            ValueType::Uint16 | ValueType::Int16 => 2,
            ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => 4,
            ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => 8,
            ValueType::String | ValueType::Array => 0,
        };
        let bytes = count
            .checked_mul(scalar_size)
            .ok_or(GgufError::ArithmeticOverflow)?;
        self.skip_bounded(bytes)
    }

    fn require_scalar_count_fits_remaining(
        &mut self,
        value_type: ValueType,
        count: u64,
    ) -> Result<(), GgufError> {
        let minimum_encoded_bytes = match value_type {
            ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => 1,
            ValueType::Uint16 | ValueType::Int16 => 2,
            ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => 4,
            ValueType::Uint64
            | ValueType::Int64
            | ValueType::Float64
            | ValueType::String => 8,
            ValueType::Array => return Err(GgufError::NestedArray),
        };
        let maximum_from_file = self.remaining_bytes()? / minimum_encoded_bytes;
        if count > maximum_from_file {
            return Err(GgufError::ArrayTooLarge(count));
        }
        Ok(())
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let length = self.u64()?;
        if length > MAXIMUM_STRING_BYTES {
            return Err(GgufError::StringTooLarge(length));
        }
        let length =
            usize::try_from(length).map_err(|_| GgufError::ArithmeticOverflow)?;
        let mut bytes = vec![0_u8; length];
        self.reader.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8)
    }

    fn string_with_resident_budget(
        &mut self,
        maximum_bytes: u64,
    ) -> Result<String, GgufError> {
        let length = self.u64()?;
        if length > MAXIMUM_STRING_BYTES {
            return Err(GgufError::StringTooLarge(length));
        }
        if length > maximum_bytes || length > self.remaining_bytes()? {
            return Err(GgufError::ResourceLimit);
        }
        let length =
            usize::try_from(length).map_err(|_| GgufError::ArithmeticOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| GgufError::ResourceLimit)?;
        bytes.resize(length, 0);
        self.reader.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8)
    }

    fn skip_bounded(&mut self, bytes: u64) -> Result<(), GgufError> {
        let current = self.position()?;
        let destination = current
            .checked_add(bytes)
            .ok_or(GgufError::ArithmeticOverflow)?;
        if destination > self.file_size {
            return Err(GgufError::Io("truncated GGUF".into()));
        }
        self.reader.seek(SeekFrom::Start(destination))?;
        Ok(())
    }

    fn position(&mut self) -> Result<u64, GgufError> {
        Ok(self.reader.stream_position()?)
    }

    fn remaining_bytes(&mut self) -> Result<u64, GgufError> {
        self.file_size
            .checked_sub(self.position()?)
            .ok_or(GgufError::ArithmeticOverflow)
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut bytes = [0_u8; N];
        self.reader.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.bytes::<1>()?[0])
    }
    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(i8::from_le_bytes(self.bytes()?))
    }
    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }
    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.bytes()?))
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.bytes()?))
    }
    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.bytes()?))
    }
    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.bytes()?))
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment(alignment));
    }
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or(GgufError::ArithmeticOverflow)
}

fn scalar_materialized_heap_bytes(value: &Scalar) -> u64 {
    match value {
        Scalar::Unsigned(_)
        | Scalar::Signed(_)
        | Scalar::Float(_)
        | Scalar::Bool(_) => 0,
        Scalar::String(value) => u64::try_from(value.len()).unwrap_or(u64::MAX),
    }
}

fn validate_non_overlapping(tensors: &[TensorInfo]) -> Result<(), GgufError> {
    let mut spans = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.absolute_offset,
                tensor.absolute_offset + tensor.byte_size,
            )
        })
        .collect::<Vec<_>>();
    spans.sort_unstable();
    if spans.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(GgufError::TensorOverlap);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{Cursor, Read, Seek, SeekFrom, Write};
    use std::time::{SystemTime, UNIX_EPOCH};

    use sha2::{Digest, Sha256};

    use super::{
        GgufError, MAXIMUM_METADATA_RESIDENT_BYTES,
        MAXIMUM_TENSOR_DIRECTORY_RESIDENT_BYTES, MAXIMUM_TENSOR_ENTRIES, Scalar,
        ValueType, VerifiedGgufSnapshot, read_from, tensor_inventory,
    };

    fn push_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&u64::try_from(value.len()).unwrap().to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "qwen2");
        push_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        push_string(&mut bytes, "output.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&[0_u8; 16]);
        bytes
    }

    fn string_array_fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        push_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        push_string(&mut bytes, "token-a");
        push_string(&mut bytes, "token-b");
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes
    }

    fn temporary(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "imparo-gguf-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[cfg(unix)]
    fn with_cursor_redirect<T>(
        snapshot: &VerifiedGgufSnapshot,
        reads_to_skip: u64,
        action: impl FnOnce() -> T,
    ) -> T {
        let mut raw = snapshot.try_clone_file().unwrap();
        let length = snapshot.length();
        let (scope, controller) =
            super::test_coordination::intercept_after(reads_to_skip).unwrap();
        let worker = std::thread::spawn(move || {
            controller.wait_until_reader_located().unwrap();
            raw.seek(SeekFrom::Start(length)).unwrap();
            controller.release_reader().unwrap();
        });
        let result = action();
        worker.join().unwrap();
        drop(scope);
        result
    }

    #[cfg(unix)]
    fn restored_mtime_mutation(
        controller: super::test_coordination::Controller,
        path: std::path::PathBuf,
        offset: u64,
        replacement: u8,
    ) -> std::thread::JoinHandle<std::io::Result<(fs::Metadata, fs::Metadata)>> {
        std::thread::spawn(move || {
            controller.wait_until_reader_located()?;
            let result: std::io::Result<(fs::Metadata, fs::Metadata)> = (|| {
                let before = fs::metadata(&path)?;
                std::thread::sleep(std::time::Duration::from_millis(2));
                let mut writer = fs::OpenOptions::new().write(true).open(&path)?;
                writer.seek(SeekFrom::Start(offset))?;
                writer.write_all(&[replacement])?;
                writer.sync_all()?;
                writer.set_times(
                    fs::FileTimes::new()
                        .set_accessed(before.accessed()?)
                        .set_modified(before.modified()?),
                )?;
                Ok((before, fs::metadata(&path)?))
            })(
            );
            let release = controller.release_reader();
            let metadata = result?;
            release?;
            Ok(metadata)
        })
    }

    #[cfg(unix)]
    fn assert_restored_mtime_exposes_ctime(
        before: &fs::Metadata,
        after: &fs::Metadata,
    ) {
        use std::os::unix::fs::MetadataExt as _;

        assert_eq!((before.dev(), before.ino()), (after.dev(), after.ino()));
        assert_eq!(before.len(), after.len());
        assert_eq!(
            (before.mtime(), before.mtime_nsec()),
            (after.mtime(), after.mtime_nsec())
        );
        assert_ne!(
            (before.ctime(), before.ctime_nsec()),
            (after.ctime(), after.ctime_nsec())
        );
    }

    #[cfg(unix)]
    #[test]
    fn positional_consumers_ignore_shared_cursor_redirect() {
        let root = temporary("positional");
        fs::create_dir(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, string_array_fixture()).unwrap();
        let snapshot = VerifiedGgufSnapshot::open(&source).unwrap();

        let mut known_bad = snapshot.try_clone_file().unwrap();
        let mut disturber = snapshot.try_clone_file().unwrap();
        known_bad.seek(SeekFrom::Start(0)).unwrap();
        disturber.seek(SeekFrom::End(0)).unwrap();
        let mut magic = [0_u8; 4];
        assert!(known_bad.read_exact(&mut magic).is_err());

        let document =
            with_cursor_redirect(&snapshot, 0, || snapshot.read_document()).unwrap();
        assert_eq!(document.version, 3);
        let materialized = with_cursor_redirect(&snapshot, 1, || {
            snapshot.read_metadata_array("tokenizer.ggml.tokens", 2, 64)
        })
        .unwrap();
        assert_eq!(
            materialized.values,
            [
                Scalar::String("token-a".into()),
                Scalar::String("token-b".into())
            ]
        );
        with_cursor_redirect(&snapshot, 0, || snapshot.read_exact_at(0, &mut magic))
            .unwrap();
        assert_eq!(magic, *b"GGUF");
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_hash_and_consumers_reject_restored_mtime_drift() {
        let root = temporary("restored-mtime");
        fs::create_dir(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let source = root.join("model.gguf");
        let original = string_array_fixture();
        let offset = u64::try_from(original.len() - 1).unwrap();
        let replacement = original[original.len() - 1] ^ 1;

        fs::write(&source, &original).unwrap();
        let (scope, controller) = super::test_coordination::intercept_after(0).unwrap();
        let worker =
            restored_mtime_mutation(controller, source.clone(), offset, replacement);
        let result = VerifiedGgufSnapshot::open(&source);
        let (before, after) = worker.join().unwrap().unwrap();
        drop(scope);
        assert_restored_mtime_exposes_ctime(&before, &after);
        assert!(matches!(
            result,
            Err(GgufError::SourceChangedDuringSnapshot)
        ));

        for (name, skip, consumer) in [
            ("document", 0_u64, 0_u8),
            ("metadata", 1_u64, 1_u8),
            ("range", 0_u64, 2_u8),
        ] {
            let path = root.join(format!("{name}.gguf"));
            fs::write(&path, &original).unwrap();
            let snapshot = VerifiedGgufSnapshot::open(&path).unwrap();
            let (scope, controller) =
                super::test_coordination::intercept_after(skip).unwrap();
            let worker =
                restored_mtime_mutation(controller, path.clone(), offset, replacement);
            let mut byte = 0_u8;
            let result = match consumer {
                0 => snapshot.read_document().map(|_| ()),
                1 => snapshot
                    .read_metadata_array("tokenizer.ggml.tokens", 2, 64)
                    .map(|_| ()),
                2 => snapshot.read_exact_at(offset, std::slice::from_mut(&mut byte)),
                _ => unreachable!(),
            };
            let (before, after) = worker.join().unwrap().unwrap();
            drop(scope);
            assert_restored_mtime_exposes_ctime(&before, &after);
            assert_eq!(result, Err(GgufError::SourceChangedDuringSnapshot));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_range_rejects_bounds_overflow_and_short_read() {
        let root = temporary("exact-range-errors");
        fs::create_dir(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let source = root.join("model.gguf");
        let bytes = string_array_fixture();
        fs::write(&source, &bytes).unwrap();
        let snapshot = VerifiedGgufSnapshot::open(&source).unwrap();
        assert_eq!(
            snapshot.read_exact_at(u64::MAX, &mut [0_u8; 2]),
            Err(GgufError::ArithmeticOverflow)
        );
        assert!(matches!(
            snapshot.read_exact_at(snapshot.length(), &mut [0_u8; 1]),
            Err(GgufError::SourceChangedDuringSnapshot)
        ));
        let file = fs::File::open(&source).unwrap();
        let mut reader = super::HeldSnapshotPositionalReader {
            file: &file,
            logical_cursor: snapshot.length() - 1,
            length: snapshot.length() + 2,
        };
        assert_eq!(
            reader.read_exact(&mut [0_u8; 3]).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_a_bounded_tensor_directory() {
        let bytes = fixture();
        let document =
            read_from(Cursor::new(&bytes), u64::try_from(bytes.len()).unwrap())
                .unwrap();
        assert_eq!(document.version, 3);
        assert_eq!(document.string_value("general.architecture"), Some("qwen2"));
        assert_eq!(document.unsigned_value("general.alignment"), Some(32));
        assert_eq!(document.tensors.len(), 1);
        assert_eq!(document.tensors[0].byte_size, 16);
        let inventory = tensor_inventory(&document).unwrap();
        assert_eq!(inventory.entries.len(), document.tensors.len());
        assert_eq!(inventory.entries[0].role, "output-head");
        assert_eq!(inventory.entries[0].elements, 4);
        assert_eq!(inventory.entries[0].block_elements, 1);
        assert_eq!(inventory.bytes_by_type["F32"], 16);
        assert_eq!(inventory.bytes_by_role["output-head"], 16);
    }

    #[test]
    fn routed_expert_inventory_roles_are_architecture_neutral() {
        for (name, expected) in [
            ("blk.7.attn_q_norm.weight", "attention-query-norm"),
            ("blk.7.attn_k_norm.weight", "attention-key-norm"),
            ("blk.7.ffn_gate_inp.weight", "moe-router"),
            ("blk.7.ffn_gate_exps.weight", "moe-expert-gate"),
            ("blk.7.ffn_up_exps.weight", "moe-expert-up"),
            ("blk.7.ffn_down_exps.weight", "moe-expert-down"),
        ] {
            assert_eq!(
                super::tensor_role_and_layer(name),
                (expected.into(), Some(7))
            );
        }
    }

    #[test]
    fn opened_snapshot_keeps_parse_and_identity_on_one_object() {
        let root = std::env::temp_dir()
            .join(format!("imparo-opened-gguf-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let source = root.join("model.gguf");
        let original = fixture();
        fs::write(&source, &original).unwrap();
        let snapshot = VerifiedGgufSnapshot::open(&source).unwrap();
        let replacement = root.join("replacement.gguf");
        let mut changed = original.clone();
        changed[0] = b'X';
        fs::write(&replacement, changed).unwrap();
        fs::rename(&replacement, &source).unwrap();
        assert!(matches!(
            snapshot.read_document(),
            Err(GgufError::SourceChangedDuringSnapshot)
        ));
        assert_eq!(fs::read(snapshot.path()).unwrap()[0], b'X');
        let mut held = snapshot.file.try_clone().unwrap();
        held.seek(SeekFrom::Start(0)).unwrap();
        let mut held_bytes = Vec::new();
        held.read_to_end(&mut held_bytes).unwrap();
        assert_eq!(held_bytes, original);
        assert_eq!(
            snapshot.logical_read_bytes(),
            u64::try_from(original.len()).unwrap()
        );
        assert_eq!(snapshot.temporary_bytes(), 1024 * 1024);
        assert_eq!(
            snapshot.sha256(),
            Sha256::digest(&original)
                .iter()
                .fold(String::new(), |mut hex, b| {
                    use std::fmt::Write as _;
                    let _ = write!(hex, "{b:02x}");
                    hex
                })
        );
        let displaced = snapshot.path().with_extension("displaced");
        fs::rename(snapshot.path(), &displaced).unwrap();
        fs::write(snapshot.path(), &original).unwrap();
        assert!(matches!(
            snapshot.verify_snapshot_path(),
            Err(GgufError::SourceChangedDuringSnapshot)
        ));
        drop(snapshot);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn opened_snapshot_rejects_in_place_mutation_without_full_copy() {
        let root = std::env::temp_dir().join(format!(
            "imparo-opened-gguf-mutation-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let source = root.join("model.gguf");
        let original = fixture();
        fs::write(&source, &original).unwrap();
        let snapshot = VerifiedGgufSnapshot::open(&source).unwrap();
        let mut changed = original;
        changed[0] = b'X';
        fs::write(&source, changed).unwrap();
        assert!(matches!(
            snapshot.read_document(),
            Err(GgufError::SourceChangedDuringSnapshot)
        ));
        assert!(matches!(
            snapshot.try_clone_file(),
            Err(GgufError::SourceChangedDuringSnapshot)
        ));
        eprintln!(
            "GGUF_SINGLE_HANDLE_NO_FULL_COPY=PASS source_bytes={} temporary_bytes={} logical_reads={}",
            snapshot.length(),
            snapshot.temporary_bytes(),
            snapshot.logical_read_bytes()
        );
        drop(snapshot);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn opened_snapshot_rejects_final_and_parent_symlinks() {
        let root = std::env::temp_dir()
            .join(format!("imparo-opened-gguf-links-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("direct")).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let source = root.join("direct/model.gguf");
        fs::write(&source, fixture()).unwrap();
        let file_link = root.join("file-link.gguf");
        std::os::unix::fs::symlink(&source, &file_link).unwrap();
        assert!(matches!(
            VerifiedGgufSnapshot::open(file_link),
            Err(GgufError::UnsafePath | GgufError::Io(_))
        ));
        let parent_link = root.join("parent-link");
        std::os::unix::fs::symlink(root.join("direct"), &parent_link).unwrap();
        assert!(matches!(
            VerifiedGgufSnapshot::open(parent_link.join("model.gguf")),
            Err(GgufError::UnsafePath | GgufError::Io(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_files_fail_closed() {
        let mut bytes = fixture();
        bytes[0] = b'X';
        assert_eq!(
            read_from(Cursor::new(&bytes), u64::try_from(bytes.len()).unwrap()),
            Err(GgufError::InvalidMagic)
        );

        let mut truncated = fixture();
        truncated.pop();
        assert!(matches!(
            read_from(
                Cursor::new(&truncated),
                u64::try_from(truncated.len()).unwrap()
            ),
            Err(GgufError::TensorOutOfBounds(_))
        ));
    }

    #[test]
    fn untrusted_counts_are_rejected_before_capacity_allocation() {
        let mut tensors = Vec::new();
        tensors.extend_from_slice(b"GGUF");
        tensors.extend_from_slice(&3_u32.to_le_bytes());
        tensors.extend_from_slice(&MAXIMUM_TENSOR_ENTRIES.to_le_bytes());
        tensors.extend_from_slice(&0_u64.to_le_bytes());
        assert_eq!(
            read_from(Cursor::new(&tensors), u64::try_from(tensors.len()).unwrap()),
            Err(GgufError::TooManyTensorEntries(MAXIMUM_TENSOR_ENTRIES))
        );

        let mut array = Vec::new();
        array.extend_from_slice(b"GGUF");
        array.extend_from_slice(&3_u32.to_le_bytes());
        array.extend_from_slice(&0_u64.to_le_bytes());
        array.extend_from_slice(&1_u64.to_le_bytes());
        push_string(&mut array, "x");
        array.extend_from_slice(&(ValueType::Array as u32).to_le_bytes());
        array.extend_from_slice(&(ValueType::String as u32).to_le_bytes());
        array.extend_from_slice(&1_000_000_u64.to_le_bytes());
        assert_eq!(
            read_from(Cursor::new(&array), u64::try_from(array.len()).unwrap()),
            Err(GgufError::ArrayTooLarge(1_000_000))
        );
    }

    #[test]
    fn duplicate_metadata_moves_the_incoming_key_without_cloning_the_stored_key() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        for value in [1_u8, 2_u8] {
            push_string(&mut bytes, "duplicate-key");
            bytes.extend_from_slice(&(ValueType::Uint8 as u32).to_le_bytes());
            bytes.push(value);
        }
        let aligned = usize::try_from(
            super::align_up(u64::try_from(bytes.len()).unwrap(), 32).unwrap(),
        )
        .unwrap();
        bytes.resize(aligned, 0);
        assert_eq!(
            read_from(Cursor::new(&bytes), u64::try_from(bytes.len()).unwrap()),
            Err(GgufError::DuplicateMetadata("duplicate-key".into()))
        );
    }

    #[test]
    fn sparse_valid_encoding_cannot_exceed_resident_directory_or_array_budget() {
        let root = temporary("sparse-resource-budget");
        fs::create_dir(&root).unwrap();
        let root = fs::canonicalize(root).unwrap();

        let tensor_path = root.join("tensors.gguf");
        let tensor_count = MAXIMUM_TENSOR_DIRECTORY_RESIDENT_BYTES
            / super::MINIMUM_TENSOR_ENTRY_RESIDENT_BYTES
            + 1;
        let mut tensor_prefix = Vec::new();
        tensor_prefix.extend_from_slice(b"GGUF");
        tensor_prefix.extend_from_slice(&3_u32.to_le_bytes());
        tensor_prefix.extend_from_slice(&tensor_count.to_le_bytes());
        tensor_prefix.extend_from_slice(&0_u64.to_le_bytes());
        let mut tensor_file = fs::File::create(&tensor_path).unwrap();
        tensor_file.write_all(&tensor_prefix).unwrap();
        tensor_file
            .set_len(
                u64::try_from(tensor_prefix.len()).unwrap()
                    + tensor_count * super::MINIMUM_TENSOR_HEADER_BYTES,
            )
            .unwrap();
        assert_eq!(
            super::read(&tensor_path),
            Err(GgufError::TooManyTensorEntries(tensor_count))
        );

        let metadata_path = root.join("metadata.gguf");
        let metadata_count = MAXIMUM_METADATA_RESIDENT_BYTES
            / super::MINIMUM_METADATA_ENTRY_RESIDENT_BYTES
            + 1;
        let mut metadata_prefix = Vec::new();
        metadata_prefix.extend_from_slice(b"GGUF");
        metadata_prefix.extend_from_slice(&3_u32.to_le_bytes());
        metadata_prefix.extend_from_slice(&0_u64.to_le_bytes());
        metadata_prefix.extend_from_slice(&metadata_count.to_le_bytes());
        let mut metadata_file = fs::File::create(&metadata_path).unwrap();
        metadata_file.write_all(&metadata_prefix).unwrap();
        metadata_file
            .set_len(
                u64::try_from(metadata_prefix.len()).unwrap()
                    + metadata_count * super::MINIMUM_METADATA_ENTRY_BYTES,
            )
            .unwrap();
        assert_eq!(
            super::read(&metadata_path),
            Err(GgufError::TooManyMetadataEntries(metadata_count))
        );

        let array_path = root.join("array.gguf");
        let element_count = 100_000_u64;
        let mut array_prefix = Vec::new();
        array_prefix.extend_from_slice(b"GGUF");
        array_prefix.extend_from_slice(&3_u32.to_le_bytes());
        array_prefix.extend_from_slice(&0_u64.to_le_bytes());
        array_prefix.extend_from_slice(&1_u64.to_le_bytes());
        push_string(&mut array_prefix, "x");
        array_prefix.extend_from_slice(&(ValueType::Array as u32).to_le_bytes());
        array_prefix.extend_from_slice(&(ValueType::Bool as u32).to_le_bytes());
        array_prefix.extend_from_slice(&element_count.to_le_bytes());
        let mut array_file = fs::File::create(&array_path).unwrap();
        array_file.write_all(&array_prefix).unwrap();
        let encoded_end = u64::try_from(array_prefix.len()).unwrap() + element_count;
        array_file
            .set_len(super::align_up(encoded_end, 32).unwrap())
            .unwrap();
        let resident_bytes =
            element_count * u64::try_from(std::mem::size_of::<Scalar>()).unwrap();
        assert_eq!(
            super::read_metadata_array(
                &array_path,
                "x",
                element_count,
                resident_bytes - 1,
            ),
            Err(GgufError::MaterializedDataTooLarge(resident_bytes))
        );
        fs::remove_dir_all(root).unwrap();
    }

    /// Known-answer check against FIPS 180-2's published vector, so a sha2 crate
    /// upgrade (0.10 -> 0.11 changed the digest output type and killed LowerHex)
    /// cannot silently change the hex strings the snapshot machinery emits.
    #[test]
    fn sha256_hex_matches_published_vector() {
        let hex = Sha256::digest(b"abc")
            .iter()
            .fold(String::new(), |mut hex, b| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{b:02x}");
                hex
            });
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn every_prefix_truncation_fails_closed_without_panicking() {
        let bytes = fixture();
        for length in 0..bytes.len() {
            let truncated = &bytes[..length];
            assert!(
                read_from(
                    Cursor::new(truncated),
                    u64::try_from(truncated.len()).unwrap()
                )
                .is_err(),
                "unexpectedly accepted prefix length {length}"
            );
        }
    }

    #[test]
    fn scalar_enum_remains_explicit() {
        assert_eq!(Scalar::Unsigned(32), Scalar::Unsigned(32));
    }
}
