//! The prefill GEMM's tile choice is arithmetic, so it is tested as arithmetic.
//!
//! Every row here was measured end to end on LFM2 before it was written down; the test
//! exists so the rule cannot drift away from the measurements without failing.

#![cfg(target_os = "macos")]

/// THE TILE RULE, both regimes, in one test because the seat and the bound are process
/// globals: two tests that move them are two threads moving them, and `cargo test` runs
/// them at the same time. (They did race -- one read the other's seat mid-assertion.)
///
/// ```text
///   n <= bound    the NARROWEST tile that holds n_tok in one token group
///   n >  bound    the SEAT
/// ```
///
/// Below the bound a batch walks the weight matrix once whatever tile it takes, so padded
/// rows are the only term left and the narrowest tile that holds it wins: against the seat's
/// 64x32, verify reads -19.5% at 8 rows on a 64x8 tile, -9.9% at 10 and -12.4% at 16 on a
/// 64x16, and 32 rows -- which pad to 32 either way -- tie.
///
/// Above the bound the walks decide instead, and going NARROWER is what the bound prevents:
/// at 464 rows a 16-token tile walks 29 times against 32's 15, and prefill reads 9422.9
/// against 8676.3 ms.
///
/// Going WIDER above the bound was built, measured and rejected. The rule "the widest tile
/// that pads no worse than the seat" would have answered 64-wide at 448 and 512 and 32-wide
/// at 455 and 480 (a wider token tile halves the weight passes and pays in padding, and the
/// two signs are opposite in the right places). It loses: one binary, four prefill samples
/// interleaved with the arms reversed between rounds, 8444 tokens at the 512 chunk --
///
/// ```text
///   derived widest-no-worse   8334.4  8334.6
///   the seat (64x32)          8287.4  8295.8     +0.5% for the wider tile
/// ```
#[test]
fn the_tile_rule_is_two_regimes_with_one_bound() {
    // Shape 2 is 64x16, shape 3 is 64x32, shape 7 is 64x64, shape 12 is 64x8. All four
    // carry four simdgroups and a 32-deep K chunk, so they are one family and n_in only
    // has to be a multiple of 32 for the K-chunk fallback to stay out of it.
    let seat = 3;
    let (was_shape, was_max) = (
        imparo_metal::st_gemm_shape(),
        imparo_metal::st_gemm_narrow_max(),
    );
    imparo_metal::set_st_gemm_shape(seat);

    // BOUND AT 0 is the seat at every width, narrow or wide.
    imparo_metal::set_st_gemm_narrow_max(0);
    for n_tok in [1, 8, 16, 32, 448, 455, 480, 512] {
        let picked = imparo_metal::q8_pick_shape(n_tok, 2048);
        assert_eq!(
            picked, seat,
            "n_tok={n_tok} left the seat with the bound at 0 (picked {picked})"
        );
    }

    // UNDER THE BOUND the tile is the narrowest that holds the batch in ONE token group.
    imparo_metal::set_st_gemm_narrow_max(16);
    for (n_tok, want_toks) in [(1, 8), (8, 8), (9, 16), (16, 16)] {
        let picked = imparo_metal::q8_pick_shape(n_tok, 2048);
        assert_eq!(
            imparo_metal::q8_shape_tokens(picked),
            want_toks,
            "n_tok={n_tok} picked shape {picked}"
        );
    }
    // One row past it the seat decides again, and 464 -- where the narrow rule would have
    // taken a 16-token tile, 29 walks against 15 -- is back on the seat.
    for n_tok in [17, 32, 448, 464, 512] {
        let picked = imparo_metal::q8_pick_shape(n_tok, 2048);
        assert_eq!(
            picked, seat,
            "n_tok={n_tok} is above the bound and still moved off the seat (picked {picked})"
        );
    }

    // A BOUND WIDER THAN THE FAMILY still answers the seat where no tile is narrower: at
    // 33..64 rows the narrowest tile that holds them IS the 64-wide one, and past 64 the
    // family has nothing wide enough, so the seat stands.
    imparo_metal::set_st_gemm_narrow_max(64);
    assert_eq!(
        imparo_metal::q8_shape_tokens(imparo_metal::q8_pick_shape(33, 2048)),
        64
    );
    assert_eq!(imparo_metal::q8_pick_shape(65, 2048), seat);

    imparo_metal::set_st_gemm_shape(was_shape);
    imparo_metal::set_st_gemm_narrow_max(was_max);
}

/// The mirror's padding constant has to cover every token tile the GEMM table can select.
///
/// `half_activation_mirror_requirements` pads to `MAX_GEMM_TOKEN_TILE`; the kernels walk
/// whole tiles from `ST_GEMM_SHAPES`. If a wider shape is ever added and the constant is
/// not raised with it, the conversion pass writes past the mirror again -- silently, into
/// the arena.
#[test]
fn q8_token_tiles_fit_the_mirror() {
    for shape in 0..imparo_metal::st_gemm_shapes() {
        let toks = imparo_metal::q8_shape_tokens(shape);
        assert!(
            toks as usize <= imparo_backend::MAX_GEMM_TOKEN_TILE,
            "shape {shape} uses a {toks}-token tile, wider than MAX_GEMM_TOKEN_TILE"
        );
    }
}
