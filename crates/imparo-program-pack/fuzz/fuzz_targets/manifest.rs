#![no_main]

use imparo_program_pack::Manifest;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = Manifest::parse(data);
});
