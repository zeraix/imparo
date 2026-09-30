//! How a load tells that a file is the one a pairing manifest recorded (the Gemma4 MTP
//! drafter's manifest; a DSpark drafter is named by its file and needs none).
//!
//! The manifest records each file's size, SHA-256 and stamp (`stamp`). A load whose stamps
//! match reads nothing; a file whose stamp differs, or a manifest without stamps, is hashed and
//! must match the recorded SHA-256.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use sha2::{Digest, Sha256};

/// What tells a file's content apart without reading it: its length, modification time,
/// device and inode. A write or a touch moves the time, a replace moves the inode. Only an
/// in-place rewrite that leaves the modification time where it was -- set back, or inside the
/// file system's timestamp resolution -- keeps the stamp.
///
/// # Errors
/// When the file cannot be stat'ed, or on an OS where this crate takes no stamp (then the
/// loader hashes the file).
pub fn stamp(file: &File) -> Result<String, String> {
    #[cfg(not(windows))]
    let m = file.metadata().map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(format!(
            "{} {}.{:09} {} {}",
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.dev(),
            m.ino()
        ))
    }
    #[cfg(windows)]
    {
        let (len, modified, volume, index) =
            crate::windows_opened_stamp(file).map_err(|e| e.to_string())?;
        Ok(format!("windows {len} {modified} {volume} {index}"))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = m;
        Err("no file stamp on this OS".into())
    }
}

/// How a load established that a file is the one paired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checked {
    /// Its stamp matched; nothing was read.
    Stamp,
    /// Its SHA-256 matched (the stamp differed or was not recorded).
    Hashed,
}

/// Checks that `path` is the file a manifest recorded: the same length, then the same stamp
/// (nothing read), or else the same SHA-256.
///
/// # Errors
/// When the file cannot be read, or its length or content differs from the record.
pub fn verify(
    path: &Path,
    bytes: u64,
    sha256: &str,
    recorded_stamp: Option<&str>,
) -> Result<Checked, String> {
    let named = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = File::open(path).map_err(named)?;
    let len = f.metadata().map_err(named)?.len();
    if len != bytes {
        return Err(format!(
            "{}: {len} bytes, the pairing recorded {bytes}",
            path.display()
        ));
    }
    if recorded_stamp.is_some_and(|want| stamp(&f).is_ok_and(|have| have == want)) {
        return Ok(Checked::Stamp);
    }
    if sha256_of(&mut f, bytes).map_err(|e| format!("{}: {e}", path.display()))?
        != sha256
    {
        return Err(format!(
            "{}: its SHA-256 differs from the pairing's",
            path.display()
        ));
    }
    Ok(Checked::Hashed)
}

/// Lowercase hex SHA-256 of the first `len` bytes of an open file.
fn sha256_of(f: &mut File, len: u64) -> Result<String, String> {
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut left = len;
    let mut buf = vec![0_u8; 1 << 20];
    while left > 0 {
        let take =
            usize::try_from(left.min(buf.len() as u64)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf[..take]).map_err(|e| e.to_string())?;
        h.update(&buf[..take]);
        left -= take as u64;
    }
    Ok(h.finalize()
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("imparo-pairing-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn a_matching_stamp_reads_nothing_and_a_moved_one_hashes() {
        let d = scratch("verify");
        let t = d.join("t.gguf");
        std::fs::write(&t, vec![7_u8; 4096]).unwrap();
        let stamped = stamp(&File::open(&t).unwrap()).unwrap();
        let sha256 = sha256_of(&mut File::open(&t).unwrap(), 4096).unwrap();
        // A stamp that matches is trusted: the recorded hash is not even compared.
        assert_eq!(
            verify(&t, 4096, "not a hash", Some(&stamped)),
            Ok(Checked::Stamp)
        );
        // No stamp, or one that differs: the content decides.
        assert_eq!(verify(&t, 4096, &sha256, None), Ok(Checked::Hashed));
        assert_eq!(
            verify(&t, 4096, &sha256, Some("0 0.0 0 0")),
            Ok(Checked::Hashed)
        );
        assert!(verify(&t, 4096, "not a hash", Some("0 0.0 0 0")).is_err());
        // Same length, new content, a new stamp: refused. The time is set explicitly: a
        // rewrite inside the file system's timestamp resolution would keep the old one.
        std::fs::write(&t, vec![8_u8; 4096]).unwrap();
        File::options()
            .write(true)
            .open(&t)
            .unwrap()
            .set_modified(
                std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1),
            )
            .unwrap();
        assert!(verify(&t, 4096, &sha256, Some(&stamped)).is_err());
        assert!(verify(&t, 4095, &sha256, None).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
