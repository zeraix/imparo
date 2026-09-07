//! CUDA build contract. Dynamic-runtime builds generate the ABI contract only;
//! source builds additionally compile the native backend when `cuda-static` is on.
//! Hosts without a CUDA toolkit can still build and test the thin runtime.

mod build_support;

struct Catalog {
    backend_abi: u32,
    sms: Vec<u32>,
}

fn read_catalog() -> Catalog {
    let catalog =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("cuda-sm.json");
    println!("cargo:rerun-if-changed={}", catalog.display());
    let text = std::fs::read_to_string(&catalog).expect("read cuda-sm.json");
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("cuda-sm.json must be valid JSON");
    assert_eq!(
        value["schema"].as_u64(),
        Some(1),
        "unsupported cuda-sm.json schema"
    );
    let backend_abi = value["backend_abi"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .filter(|v| *v > 0)
        .expect("cuda-sm.json backend_abi must be a positive u32");
    let sms: Vec<u32> = value["sms"]
        .as_array()
        .expect("cuda-sm.json sms must be an array")
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v >= 50)
                .expect("cuda-sm.json SM values must be u32 compute capabilities")
        })
        .collect();
    assert!(!sms.is_empty(), "cuda-sm.json sms cannot be empty");
    assert!(
        sms.windows(2).all(|pair| pair[0] < pair[1]),
        "cuda-sm.json sms must be sorted and unique"
    );
    Catalog { backend_abi, sms }
}

fn track_native_sources(directory: &std::path::Path) {
    for entry in
        std::fs::read_dir(directory).expect("read CUDA native source directory")
    {
        let entry = entry.expect("read CUDA native source entry");
        let path = entry.path();
        if path.is_dir() {
            // Standalone probes own their compilation and are not linked into the
            // backend object. Tracking them here would rebuild every engine binary.
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            track_native_sources(&path);
            continue;
        }
        if path.extension().is_some_and(|extension| {
            matches!(extension.to_str(), Some("cu" | "cuh" | "h" | "def"))
        }) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn main() {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    track_native_sources(&manifest_dir.join("native"));
    for name in [
        "NVCC",
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDAARCHS",
        "IMPARO_CUDA_ARCHS",
        "IMPARO_CUDA_PRECISE_MATH",
        "IMPARO_CUDA_SKIP_NVCC",
        "IMPARO_CUDA_EMBED_MANIFEST",
        "IMPARO_CUDA_RELEASE_PUBLIC_KEY",
        "IMPARO_CUDA_PROGRAM_SMOKE",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let out = std::env::var("OUT_DIR").unwrap();
    let catalog = read_catalog();
    let backend_abi = catalog.backend_abi;
    let program_smoke =
        std::env::var("IMPARO_CUDA_PROGRAM_SMOKE").as_deref() == Ok("1");
    println!("cargo:rustc-check-cfg=cfg(imparo_cuda_program_smoke)");
    if program_smoke {
        println!("cargo:rustc-cfg=imparo_cuda_program_smoke");
    }
    let dynamic_runtime = std::env::var("CARGO_FEATURE_CUDA_DYNAMIC").is_ok();
    let rust_math_mode = if dynamic_runtime {
        "fast"
    } else if std::env::var("IMPARO_CUDA_PRECISE_MATH").as_deref() == Ok("1") {
        "precise"
    } else {
        "fast"
    };
    let supported_sms = catalog
        .sms
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    std::fs::write(
        format!("{out}/imparo_cuda_abi.rs"),
        format!(
            "/// Native plugin ABI generated from the repository CUDA catalog.\n\
             pub const CUDA_BACKEND_ABI: u32 = {backend_abi};\n\
             /// Native SM targets generated from the same release catalog.\n\
             pub const CUDA_SUPPORTED_SMS: &[u32] = &[{supported_sms}];\n\
             /// Arithmetic mode compiled into the selected native backend.\n\
             pub const CUDA_MATH_MODE: &str = \"{rust_math_mode}\";\n"
        ),
    )
    .expect("write generated CUDA backend ABI");
    if dynamic_runtime {
        let manifest = match std::env::var("IMPARO_CUDA_EMBED_MANIFEST") {
            Ok(path) => {
                std::fs::read_to_string(path).expect("read IMPARO_CUDA_EMBED_MANIFEST")
            }
            Err(_) => "{\"schema\":1,\"backends\":[]}".to_string(),
        };
        std::fs::write(format!("{out}/imparo_cuda_manifest.json"), manifest)
            .expect("write embedded CUDA manifest");
        let public_key = std::env::var("IMPARO_CUDA_RELEASE_PUBLIC_KEY")
            .unwrap_or_else(|_| "0".repeat(64));
        assert!(
            public_key.len() == 64 && public_key.bytes().all(|b| b.is_ascii_hexdigit()),
            "IMPARO_CUDA_RELEASE_PUBLIC_KEY must be 32-byte hex"
        );
        std::fs::write(format!("{out}/imparo_cuda_public_key.txt"), public_key)
            .expect("write embedded CUDA public key");
        return;
    }
    if std::env::var("CARGO_FEATURE_CUDA_STATIC").is_err() {
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
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let win = target_os == "windows";
    let obj = format!("{out}/imparo_cuda.{}", if win { "obj" } else { "o" });
    let mut args = vec![
        "-O3".to_string(),
        "-std=c++17".to_string(),
        "-lineinfo".to_string(),
        format!("-DIMPARO_CUDA_BACKEND_ABI={backend_abi}"),
        // Contributor/source builds retain the opt-in cuBLAS kernel laboratory.
        // Official per-SM release plugins omit this macro and therefore remain
        // driver-only instead of importing the CUDA Toolkit's cublas DLL.
        "-DIMPARO_CUDA_ENABLE_CUBLAS_LAB=1".to_string(),
    ];
    if program_smoke {
        // Hardware-smoke helpers are compile-time gated, absent from the release
        // export list, and never become part of the stable backend ABI.
        args.push("-DIMPARO_CUDA_PROGRAM_SMOKE=1".to_string());
    }
    if win {
        // Rust's MSVC target uses the dynamic CRT. nvcc defaults to /MT, which makes the
        // archived CUDA object drag in LIBCMT and produces LNK4098 in every final binary.
        args.extend(["-Xcompiler".to_string(), "/MD".to_string()]);
    }
    // llama.cpp's production CUDA build uses fast math. The E4B CUDA/FA gate is bit-exact
    // only with the same device arithmetic, so releases and normal source builds match it.
    // A precise-math build remains available solely for kernel localization.
    if !std::env::var("IMPARO_CUDA_PRECISE_MATH").is_ok_and(|v| v == "1") {
        args.push("--use_fast_math".to_string());
    }
    // This path exists for source builds and local development. Official releases never
    // ship its fat artifact: release CI builds one native plugin per SM and the thin
    // runtime downloads exactly one. Contributors should set IMPARO_CUDA_ARCHS to their
    // device SM (for example 86); otherwise the same supported-SM catalog drives the build.
    let arch_list = std::env::var("IMPARO_CUDA_ARCHS")
        .or_else(|_| std::env::var("CUDAARCHS"))
        .unwrap_or_else(|_| {
            catalog
                .sms
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(";")
        });
    let archs: Vec<&str> = arch_list
        .split([';', ',', ' '])
        .filter(|s| !s.is_empty())
        .collect();
    assert!(
        !archs.is_empty(),
        "IMPARO_CUDA_ARCHS/CUDAARCHS cannot be empty"
    );
    let canonical_archs = archs.join(",");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target = format!("{target_os}-{target_arch}");
    let math_mode = if std::env::var("IMPARO_CUDA_PRECISE_MATH").is_ok_and(|v| v == "1")
    {
        "precise"
    } else {
        "fast"
    };
    let compiler = build_support::compiler_identity(&nvcc)
        .unwrap_or_else(|error| panic!("CUDA compiler identity: {error}"));
    let build_sha256 = build_support::native_build_sha256(
        &manifest_dir,
        backend_abi,
        &target,
        &canonical_archs,
        math_mode,
        &compiler,
    );
    args.push(format!("-DIMPARO_CUDA_BUILD_SHA256=\\\"{build_sha256}\\\""));
    args.push("-Wno-deprecated-gpu-targets".to_string());
    for (i, arch) in archs.iter().enumerate() {
        assert!(
            arch.chars().all(|c| c.is_ascii_digit()),
            "invalid CUDA arch '{arch}'"
        );
        let code = if i + 1 == archs.len() {
            format!("[sm_{arch},compute_{arch}]")
        } else {
            format!("sm_{arch}")
        };
        args.extend([
            "--generate-code".to_string(),
            format!("arch=compute_{arch},code={code}"),
        ]);
    }
    args.extend([
        "-c".to_string(),
        "native/imparo_cuda.cu".to_string(),
        "-o".to_string(),
        obj.clone(),
    ]);
    let status = std::process::Command::new(&nvcc).args(&args).status();
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
    // The SM86 laboratory provider mirrors llama.cpp's large-batch
    // Q4_0->F16->cuBLAS path.  It remains runtime opt-in, but source builds must
    // resolve the provider even when the selector is dormant.
    println!("cargo:rustc-link-lib=dylib=cublas");
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
        // cudart_static is built against LIBCMT while Rust's MSVC target and our nvcc
        // object use the dynamic CRT. This is a local source-build path and therefore
        // expects the CUDA runtime on PATH; official per-SM DLLs link cudart statically.
        println!("cargo:rustc-link-lib=dylib=cudart");
    } else {
        println!("cargo:rustc-link-lib=static=cudart_static");
        // nvcc compiles the native backend as C++, so Rust's final linker must
        // retain the GNU C++ runtime explicitly (`rustc` otherwise uses `cc`
        // with `-nodefaultlibs`).
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // cudart_static's own dependencies on linux
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=rt");
    }
}
