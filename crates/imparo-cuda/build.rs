//! nvcc build, gated twice: the `cuda` feature must be on AND nvcc must exist.
//! On this repo's Mac dev machine neither holds, so the crate stays a cargo-check
//! citizen; on a CUDA host the kernels compile and link like the Metal backend's
//! build.rs compiles the .mm.

fn main() {
    println!("cargo:rerun-if-changed=native/imparo_cuda.cu");
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }
    // Type-check the feature-gated Rust on a machine without a CUDA toolkit:
    // IMPARO_CUDA_SKIP_NVCC=1 skips the kernel compile (cargo check never links,
    // so the missing symbols cost nothing; an actual build without nvcc still fails).
    if std::env::var("IMPARO_CUDA_SKIP_NVCC").is_ok() {
        println!(
            "cargo:warning=imparo-cuda: nvcc skipped (IMPARO_CUDA_SKIP_NVCC); \
                  check-only build"
        );
        return;
    }
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let out = std::env::var("OUT_DIR").unwrap();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let win = target_os == "windows";
    let obj = format!("{out}/imparo_cuda.{}", if win { "obj" } else { "o" });
    let status = std::process::Command::new(&nvcc)
        .args([
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-c",
            "native/imparo_cuda.cu",
            "-o",
            &obj,
        ])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("nvcc failed with {s}; the cuda feature needs a CUDA toolkit"),
        Err(e) => panic!(
            "nvcc not found ({e}); the cuda feature needs a CUDA toolkit \
                          (set NVCC to the compiler path)"
        ),
    }
    // Archive with nvcc itself (-lib): works on MSVC where `ar` does not exist.
    let lib = if win {
        format!("{out}/imparo_cuda.lib")
    } else {
        format!("{out}/libimparo_cuda.a")
    };
    let arc = std::process::Command::new(&nvcc)
        .args(["-lib", &obj, "-o", &lib])
        .status()
        .expect("nvcc -lib");
    assert!(arc.success(), "nvcc -lib failed");
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=imparo_cuda");
    // The CUDA runtime is linked STATICALLY so a released binary runs without the
    // toolkit installed (the GPU driver is still required to actually use the GPU).
    // CUDA_PATH is set by toolkit installers and the CI action alike.
    if let Ok(cp) = std::env::var("CUDA_PATH").or_else(|_| std::env::var("CUDA_HOME")) {
        if win {
            println!("cargo:rustc-link-search=native={cp}/lib/x64");
        } else {
            println!("cargo:rustc-link-search=native={cp}/lib64");
        }
    }
    if win {
        println!("cargo:rustc-link-lib=static=cudart_static");
    } else {
        println!("cargo:rustc-link-lib=static=cudart_static");
        // cudart_static's own dependencies on linux
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=rt");
    }
}
