
//! Release packaging must expose every symbol the runtime requires. This catches
//! a working source build whose official DLL fails before it reaches any kernel.
use std::collections::BTreeSet;
#[test]
fn mandatory_runtime_and_cobatch_extension_symbols_are_exported() {
    let exports: BTreeSet<_> = include_str!("../native/imparo_cuda.def")
        .lines().map(str::trim).collect();
    let ffi=include_str!("../src/ffi.rs");
    let start=ffi.find("    macro_rules! fields {").unwrap();
    let end=start+ffi[start..].find("    macro_rules! declare_api").unwrap();
    let extension=include_str!("../src/cobatch_api.rs");
    let mut checked=0;
    for source in [&ffi[start..end],extension] {
        for line in source.lines().map(str::trim) {
            if !line.starts_with("imparo_cuda_") {continue;}
            let Some((name,_))=line.split_once('(') else {continue;};
            assert!(exports.contains(name),"required CUDA symbol missing from release exports: {name}");
            checked+=1;
        }
    }
    assert!(checked>100,"did not inspect the mandatory and extension contracts");
}
