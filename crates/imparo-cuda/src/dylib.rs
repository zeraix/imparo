//! Minimal platform loader used by the thin CUDA runtime.

#[cfg(target_os = "linux")]
use std::ffi::c_char;
use std::ffi::c_void;
use std::path::Path;

pub(crate) struct Library(*mut c_void);

// A loaded module is process-global OS state. Symbol calls are synchronized by the
// CUDA backend itself, and the handle stays alive in the global Api table.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(module: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

impl Library {
    #[cfg(windows)]
    pub(crate) fn open_system32(name: &str) -> Result<Self, String> {
        // CUDA's driver DLL is a Windows system component. Restricting this lookup to
        // System32 prevents a same-named file in the working directory from being loaded.
        const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;
        let wide: Vec<u16> = name.encode_utf16().chain([0]).collect();
        let handle = unsafe {
            LoadLibraryExW(
                wide.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if handle.is_null() {
            Err(format!(
                "LoadLibraryExW(System32) failed for {name}: {}",
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(Self(handle))
        }
    }

    #[cfg(windows)]
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        use std::os::windows::ffi::OsStrExt as _;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            Err(format!(
                "LoadLibraryW failed for {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ))
        } else {
            Ok(Self(handle))
        }
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        use std::os::unix::ffi::OsStrExt as _;
        let name = std::ffi::CString::new(path.as_os_str().as_bytes())
            .map_err(|_| format!("backend path contains NUL: {}", path.display()))?;
        let handle = unsafe { dlopen(name.as_ptr(), 2) }; // RTLD_NOW
        if handle.is_null() {
            let detail = unsafe {
                let p = dlerror();
                if p.is_null() {
                    "unknown dlopen error".into()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            Err(format!("dlopen failed for {}: {detail}", path.display()))
        } else {
            Ok(Self(handle))
        }
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        Err(format!(
            "dynamic CUDA backends are unsupported on this OS: {}",
            path.display()
        ))
    }

    #[cfg(windows)]
    pub(crate) fn symbol(&self, name: &'static [u8]) -> Result<*mut c_void, String> {
        let p = unsafe { GetProcAddress(self.0, name.as_ptr()) };
        if p.is_null() {
            Err(format!(
                "missing backend symbol {}",
                String::from_utf8_lossy(&name[..name.len() - 1])
            ))
        } else {
            Ok(p)
        }
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn symbol(&self, name: &'static [u8]) -> Result<*mut c_void, String> {
        let p = unsafe { dlsym(self.0, name.as_ptr().cast()) };
        if p.is_null() {
            Err(format!(
                "missing backend symbol {}",
                String::from_utf8_lossy(&name[..name.len() - 1])
            ))
        } else {
            Ok(p)
        }
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    pub(crate) fn symbol(&self, _name: &'static [u8]) -> Result<*mut c_void, String> {
        Err("dynamic CUDA backends are unsupported on this OS".into())
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.0);
        }
        #[cfg(target_os = "linux")]
        unsafe {
            dlclose(self.0);
        }
    }
}
