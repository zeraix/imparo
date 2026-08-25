use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/imparo_metal.mm");
    println!("cargo:rerun-if-changed=native/imparo.metal");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let native =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("MANIFEST_DIR"))
            .join("native");
    let src = native.join("imparo_metal.mm");
    // The MSL lives in its own file for real editor support; wrap it back into
    // the raw-string constant the host code compiles at runtime. The delimiter
    // cannot appear in valid MSL, but a corrupted file would truncate the
    // shader source silently -- so refuse it loudly instead.
    let msl =
        std::fs::read_to_string(native.join("imparo.metal")).expect("imparo.metal");
    assert!(
        !msl.contains(")METAL"),
        "imparo.metal contains the raw-string delimiter"
    );
    std::fs::write(
        out.join("imparo_msl.inc"),
        format!(
            r#"const char * kSource = R"METAL(
{msl})METAL";
"#
        ),
    )
    .expect("write imparo_msl.inc");
    let obj = out.join("imparo_metal.o");
    let lib = out.join("libimparo_metal.a");
    let ok = Command::new("xcrun")
        .args([
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++20",
            "-fobjc-arc",
            "-mmacosx-version-min=13.0",
            "-O2",
            "-c",
        ])
        .arg("-I")
        .arg(&out)
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("xcrun");
    assert!(ok.success(), "metal bridge compile failed");
    let ok = Command::new("libtool")
        .args(["-static", "-o"])
        .arg(&lib)
        .arg(&obj)
        .status()
        .expect("libtool");
    assert!(ok.success(), "metal bridge archive failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=imparo_metal");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
}
