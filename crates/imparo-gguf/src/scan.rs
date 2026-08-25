//! One-pass GGUF metadata reader.
//!
//! Shared by the model and tokenizer loaders so a 4 GB file's header is parsed once.
//!
//! Only the header key/values are read; tensor data is never touched. Exists because
//! reading four metadata arrays through a per-call API re-parses the whole file each time.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

/// A read-only mapping of the whole GGUF file.
///
/// Exists so the tokenizer can BORROW its token bytes from the file instead of copying
/// them into an anonymous blob: mapped pages are clean and file-backed, so they leave
/// `phys_footprint` under pressure instead of living in it.
#[cfg(unix)]
pub struct Map {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

/// Windows fallback: no mmap, the file is read to the heap. Loses the clean-pages
/// footprint property (the tokenizer's Owned path economics apply) but keeps the
/// crate compiling and correct off-unix; a MapViewOfFile port is the Windows
/// developer's optimization, not a prerequisite.
#[cfg(not(unix))]
pub struct Map {
    data: Vec<u8>,
}

#[cfg(not(unix))]
impl Map {
    fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            data: std::fs::read(path)?,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
}

// SAFETY: the mapping is PROT_READ and never mutated; sharing &self across threads only
// ever reads it, and unmapping happens exactly once in Drop.
#[cfg(unix)]
unsafe impl Send for Map {}
#[cfg(unix)]
unsafe impl Sync for Map {}

#[cfg(unix)]
impl Map {
    fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len())?;
        if len == 0 {
            return Err("empty file".into());
        }
        use std::os::fd::AsRawFd;
        // SAFETY: fd is open, len is the file's size; result checked below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        Ok(Self { ptr, len })
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: ptr/len come from a successful mmap that lives as long as self.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

#[cfg(unix)]
impl Drop for Map {
    fn drop(&mut self) {
        // SAFETY: ptr/len are the exact mapping from open(); dropped once.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARR: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

const MAX_ELEMENTS: u64 = 1 << 22;

pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Strings(Vec<String>),
    /// A string array as (file offset, byte length) pairs into the file [`Map`], from
    /// [`Metadata::read_mapped`]. The bytes stay in the file; nothing is copied.
    Spans(Vec<(u32, u32)>),
    Ints(Vec<i64>),
    Other,
}

pub struct Metadata {
    map: BTreeMap<String, Value>,
    file: Option<Arc<Map>>,
}

struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, n: usize) -> std::io::Result<Vec<u8>> {
        let mut v = vec![0_u8; n];
        self.inner.read_exact(&mut v)?;
        Ok(v)
    }
    fn u32(&mut self) -> std::io::Result<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> std::io::Result<u64> {
        let b = self.bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn string(&mut self) -> std::io::Result<String> {
        let n = usize::try_from(self.u64()?).unwrap_or(0);
        Ok(String::from_utf8_lossy(&self.bytes(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> std::io::Result<Value> {
        Ok(match t {
            T_U8 => Value::U64(u64::from(self.bytes(1)?[0])),
            T_I8 => Value::I64(i64::from(self.bytes(1)?[0] as i8)),
            T_U16 => {
                let b = self.bytes(2)?;
                Value::U64(u64::from(u16::from_le_bytes([b[0], b[1]])))
            }
            T_I16 => {
                let b = self.bytes(2)?;
                Value::I64(i64::from(i16::from_le_bytes([b[0], b[1]])))
            }
            T_U32 => Value::U64(u64::from(self.u32()?)),
            T_I32 => Value::I64(i64::from(self.u32()? as i32)),
            T_F32 => {
                let b = self.bytes(4)?;
                Value::F64(f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            }
            T_BOOL => Value::Bool(self.bytes(1)?[0] != 0),
            T_U64 => Value::U64(self.u64()?),
            T_I64 => Value::I64(self.u64()? as i64),
            T_F64 => {
                let b = self.bytes(8)?;
                Value::F64(f64::from_le_bytes([
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                ]))
            }
            T_STR => Value::Str(self.string()?),
            _ => return Err(std::io::Error::other(format!("bad scalar type {t}"))),
        })
    }
}

impl Metadata {
    /// Reads every metadata key in one pass.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or is not GGUF.
    pub fn read(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut r = Reader {
            inner: BufReader::with_capacity(1 << 20, File::open(path)?),
        };
        if r.bytes(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        Self::read_body(&mut r, None)
    }

    /// Like [`read`], but maps the file and records string ARRAYS as spans into the map
    /// rather than owned Strings -- the vocabulary is 8.2 MiB of bytes that the tokenizer
    /// can then borrow for the process's life. Falls back is the caller's job: any
    /// failure here is an error, and `read` still works without a mapping.
    pub fn read_mapped(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let map = Arc::new(Map::open(path)?);
        // The cursor borrows the map's bytes; both live to the end of this call, and the
        // spans carry plain offsets, so nothing borrows past the return.
        let bytes: &[u8] = map.bytes();
        let mut r = Reader {
            inner: Cursor::new(bytes),
        };
        if r.bytes(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let mut out = Self::read_body(
            &mut r,
            Some(
                &mut |c: &mut Cursor<&[u8]>, n: u64| -> std::io::Result<(u32, u32)> {
                    let off = c.position();
                    c.seek(SeekFrom::Current(
                        i64::try_from(n).map_err(std::io::Error::other)?,
                    ))?;
                    let off32 = u32::try_from(off).map_err(std::io::Error::other)?;
                    let n32 = u32::try_from(n).map_err(std::io::Error::other)?;
                    Ok((off32, n32))
                },
            ),
        )?;
        out.file = Some(map);
        Ok(out)
    }

    fn read_body<R: Read>(
        r: &mut Reader<R>,
        mut span_of: Option<&mut dyn FnMut(&mut R, u64) -> std::io::Result<(u32, u32)>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let _version = r.u32()?;
        let _tensors = r.u64()?;
        let n_kv = r.u64()?;
        let mut map = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let t = r.u32()?;
            let value = if t == T_ARR {
                let et = r.u32()?;
                let n = r.u64()?;
                if n > MAX_ELEMENTS {
                    return Err(format!("array {key} too large: {n}").into());
                }
                let n = usize::try_from(n).unwrap_or(0);
                match et {
                    T_STR => {
                        if let Some(spanner) = span_of.as_mut() {
                            let mut v = Vec::with_capacity(n);
                            for _ in 0..n {
                                let len = r.u64()?;
                                v.push(spanner(&mut r.inner, len)?);
                            }
                            Value::Spans(v)
                        } else {
                            let mut v = Vec::with_capacity(n);
                            for _ in 0..n {
                                v.push(r.string()?);
                            }
                            Value::Strings(v)
                        }
                    }
                    T_U8 | T_I8 | T_U16 | T_I16 | T_U32 | T_I32 | T_BOOL | T_U64
                    | T_I64 | T_F32 | T_F64 => {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(match r.scalar(et)? {
                                Value::U64(x) => i64::try_from(x).unwrap_or(0),
                                Value::I64(x) => x,
                                Value::Bool(b) => i64::from(b),
                                Value::F64(f) => f as i64,
                                _ => 0,
                            });
                        }
                        Value::Ints(v)
                    }
                    other => {
                        return Err(format!("array {key} elem type {other}").into());
                    }
                }
            } else {
                r.scalar(t)?
            };
            map.insert(key, value);
        }
        Ok(Self { map, file: None })
    }

    /// The file mapping, when this metadata came from [`read_mapped`].
    #[must_use]
    pub fn file_arc(&self) -> Option<Arc<Map>> {
        self.file.clone()
    }

    /// Moves a string array's spans out; `read_mapped` metadata only.
    pub fn take_spans(&mut self, key: &str) -> Option<Vec<(u32, u32)>> {
        match self.map.remove(key) {
            Some(Value::Spans(v)) => Some(v),
            Some(other) => {
                self.map.insert(key.to_string(), other);
                None
            }
            None => None,
        }
    }

    #[must_use]
    pub fn strings(&self, key: &str) -> Option<Vec<String>> {
        match self.map.get(key) {
            Some(Value::Strings(v)) => Some(v.clone()),
            _ => None,
        }
    }

    /// Moves an array out instead of cloning it.
    ///
    /// The vocabulary is 262144 strings and the merge list 514906; cloning both meant two
    /// live copies of roughly 30 MB during tokenizer construction, on top of the maps
    /// built from them.
    pub fn take_strings(&mut self, key: &str) -> Option<Vec<String>> {
        match self.map.remove(key) {
            Some(Value::Strings(v)) => Some(v),
            // Mapped metadata: materialize owned copies (transient callers, e.g. merges).
            Some(Value::Spans(sp)) => {
                let b = self.file.as_ref()?.bytes();
                Some(
                    sp.iter()
                        .map(|&(o, n)| {
                            String::from_utf8_lossy(
                                &b[o as usize..o as usize + n as usize],
                            )
                            .into_owned()
                        })
                        .collect(),
                )
            }
            other => {
                if let Some(o) = other {
                    self.map.insert(key.to_string(), o);
                }
                None
            }
        }
    }

    /// Moves an integer array out instead of cloning it.
    pub fn take_ints(&mut self, key: &str) -> Option<Vec<i64>> {
        match self.map.remove(key) {
            Some(Value::Ints(v)) => Some(v),
            other => {
                if let Some(o) = other {
                    self.map.insert(key.to_string(), o);
                }
                None
            }
        }
    }

    #[must_use]
    pub fn ints(&self, key: &str) -> Option<Vec<i64>> {
        match self.map.get(key) {
            Some(Value::Ints(v)) => Some(v.clone()),
            _ => None,
        }
    }
    #[must_use]
    pub fn u32_at(&self, key: &str) -> Option<u32> {
        match self.map.get(key) {
            Some(Value::U64(v)) => u32::try_from(*v).ok(),
            Some(Value::I64(v)) => u32::try_from(*v).ok(),
            _ => None,
        }
    }
    #[must_use]
    pub fn bool_at(&self, key: &str) -> Option<bool> {
        match self.map.get(key) {
            Some(Value::Bool(b)) => Some(*b),
            _ => None,
        }
    }
    /// A SCALAR string, e.g. `general.architecture` or `tokenizer.ggml.pre`.
    ///
    /// Distinct from [`Self::strings`], which takes a string ARRAY: a key holding one of
    /// them is not the other, and returning None rather than guessing is what lets the
    /// tokenizer's algorithm table fail closed on an unexpected file.
    #[must_use]
    pub fn str_at(&self, key: &str) -> Option<&str> {
        match self.map.get(key) {
            Some(Value::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}
