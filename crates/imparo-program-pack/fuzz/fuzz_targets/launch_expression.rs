#![no_main]

use imparo_program_pack::Manifest;
use libfuzzer_sys::fuzz_target;

// Launch expressions are a closed, bounded part of the strict Manifest parser. This
// dedicated target lets its corpus evolve independently from general JSON structure.
fuzz_target!(|data: &[u8]| {
    let _ = Manifest::parse(data);
});
