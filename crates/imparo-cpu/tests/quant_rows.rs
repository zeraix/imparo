//! Every row codec against llama.cpp's own dequantiser.
//!
//! `crates/imparo-cpu/src/quants.rs` is a port, and a port is a claim. The fixture holds
//! block bytes next to the floats `gguf.quants.dequantize` produces from them -- the
//! reference implementation of the same block interiors -- so the claim is checked rather
//! than eyeballed. A wrong nibble pairing or a flipped sign convention reads as plausible
//! numbers, never as a crash, which is exactly what this catches.
//!
//! Two case kinds per type: synthetic blocks of pseudo-random bytes (every field, including
//! the extremes) and, for the types a real model carries, the first row of a real tensor.
//! Regenerate with `dev_harness/gen_quant_fixture.py <out> [model.gguf]`.

use imparo_cpu::quants::row_codec;

const FIXTURE: &[u8] = include_bytes!("data/quant_rows.bin");

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

#[test]
fn every_row_codec_agrees_with_the_reference_dequantiser() {
    assert_eq!(&FIXTURE[0..4], b"IQFX", "fixture magic");
    let cases = u32_at(FIXTURE, 4);
    assert!(cases > 0, "fixture is empty");

    let mut off = 8;
    let mut worst = 0.0_f32;
    let mut seen = Vec::new();
    for case in 0..cases {
        let (kind, n_elems, n_bytes) = (
            u32_at(FIXTURE, off),
            u32_at(FIXTURE, off + 4),
            u32_at(FIXTURE, off + 8),
        );
        off += 12;
        let raw = &FIXTURE[off..off + n_bytes];
        off += n_bytes;
        let expect: Vec<f32> = FIXTURE[off..off + n_elems * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        off += n_elems * 4;

        let codec = row_codec(kind as u32).unwrap_or_else(|| {
            panic!("case {case}: no row codec for ggml type {kind}")
        });
        let mut got = vec![0.0_f32; n_elems];
        codec(raw, &mut got);

        for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
            // Relative, with an absolute floor: the reference accumulates the same terms in
            // numpy float32, so anything past the last bit or two is a real disagreement.
            let err = (g - e).abs() / e.abs().max(1e-3);
            assert!(
                err < 1e-5,
                "case {case} (ggml type {kind}) element {i}: got {g}, reference {e}"
            );
            worst = worst.max(err);
        }
        seen.push(kind);
    }
    seen.sort_unstable();
    seen.dedup();
    println!(
        "{cases} cases over {} types, worst relative error {worst:e}",
        seen.len()
    );
}

/// A type the engine cannot read must say so, not read something else. `row_codec` is the
/// one place that answer lives, so a caller can ask before a load rather than during one.
#[test]
fn an_unreadable_type_has_no_codec() {
    for kind in [34_u32, 35, 39, 40] {
        assert!(
            row_codec(kind).is_none(),
            "ggml type {kind} claims a codec it does not have"
        );
    }
    // Tile-major kinds are read through the TmRule, not as a row-major slice.
    for rule in &imparo_gguf::weights::TM_RULES {
        assert!(
            row_codec(rule.to).is_none(),
            "{} must not be a row codec",
            rule.to_name
        );
    }
}

/// The loader and the codecs must agree on ONE list.
///
/// `imparo_gguf::weights::weight_kind` decides whether a file is accepted; `row_codec`
/// decides whether its bytes can be turned into numbers. If those two lists drift, a file
/// is accepted and then read by nothing (or read by something the loader would refuse),
/// and both failures show up as a wrong answer rather than an error. F32 needs no codec
/// and the tile-major kinds are read through the `TmRule`, so they are the only exceptions.
#[test]
fn the_loader_accepts_exactly_what_a_codec_can_read() {
    use imparo_gguf::weights::{GGML_F32, ggml_type_of_wire, weight_kind};
    for kind in 0..=1030 {
        let exception =
            kind == GGML_F32 || imparo_gguf::weights::tm_rule_to(kind).is_some();
        let accepted = weight_kind(kind).is_some();
        let readable = row_codec(kind).is_some() || exception;
        assert_eq!(
            accepted, readable,
            "ggml type {kind}: loader accepts {accepted}, a codec can read {readable}"
        );
        // And the wire value round-trips: the Backend seam hands a backend the compact
        // kind, and the backend's readers switch back on the ggml type.
        if let Some(k) = weight_kind(kind) {
            assert_eq!(
                ggml_type_of_wire(k as u32),
                kind,
                "wire round trip for type {kind}"
            );
        }
    }
}

/// A tile-major row must read the same numbers as the row-major row it came from.
///
/// The layout MOVES bytes and does not change what they mean, so this is an equality, not
/// a tolerance -- the same codec decodes both. It is the property the Metal decoders will
/// be checked against, so it is pinned here first, on every rule in the family.
#[test]
fn a_tile_major_row_reads_what_the_row_major_row_reads() {
    use imparo_cpu::quants::tm_row;
    use imparo_gguf::weights::TM_RULES;
    for rule in &TM_RULES {
        // Two units of rows and three blocks of width: enough for the unit index, the
        // within-unit row and the block stride all to be non-trivial.
        let (n_in, n_out) = (rule.block_elems * 3, rule.unit_rows * 2);
        let blocks = n_in / rule.block_elems;
        let src: Vec<u8> = (0..n_out * blocks * rule.block_bytes())
            .map(|i| (i * 31 + 17) as u8)
            .collect();
        let tm = rule.convert(&src, n_in, n_out);
        let codec = row_codec(rule.from).expect("every rule's source has a codec");
        let rb = src.len() / n_out;
        let (mut want, mut got) = (vec![0.0_f32; n_in], vec![0.0_f32; n_in]);
        for r in 0..n_out {
            codec(&src[r * rb..(r + 1) * rb], &mut want);
            tm_row(&tm, rule.to, n_in, r, &mut got);
            // NaN is possible from random scale bytes and is fine as long as BOTH sides
            // produce it in the same place -- compare the bits, not the values.
            let w: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
            let g: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(w, g, "{} row {r} differs from row-major", rule.to_name);
        }
    }
}
