//! Memory-mapped GGUF weight access: the mapping, the tensor index, and the
//! weight-type table every backend dispatches on.
//!
//! CONTAINER ONLY -- no compute. The CPU matmuls/dequant that read these mappings
//! live in imparo-cpu (`ops`); GPU backends receive `base_ptr()/byte_len()` via
//! `init_weights`. This module lived in imparo-cpu for a while, which made the "CPU
//! backend" crate the loader every backend depended on; it is container machinery
//! and this crate's README-stated job ("mmap, metadata spans, tensor index, quant
//! kinds"), so it lives here.
//!
//! Weights are mapped, never copied. A tensor is read where it lies in the file; the
//! only copies are the small f32 buffers a kernel writes into.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q8_0: u32 = 8;
/// IMPARO-PRIVATE type id: Q8_0 values in the tile-major layout `imparo-repack` writes
/// (see docs/q8-tile-major-weights.md). GGUF's own ids stop far below 1000. Same bytes per
/// 32 elements as Q8_0 (34), so a converted tensor keeps its size and offset; only the
/// order of the bytes inside the tensor differs.
pub const GGML_Q8_0_TM: u32 = 1000;
/// IMPARO-PRIVATE: Q4_0 values in the same tile-major layout (18-byte blocks split into a
/// 16-byte payload and a 2-byte scale). No kernel reads it yet; the rule exists so the
/// converter, the load-time transform and the readers share one table (`TM_RULES`).
pub const GGML_Q4_0_TM: u32 = 1001;

/// Q4_0 geometry: 32 values per block, an f16 scale + 16 packed bytes.
pub const QK4_0: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Q8_0 geometry: 32 values per block, an f16 scale + 32 SIGNED bytes. No packing and
/// no -8 bias, unlike Q4_0 -- a byte is the value.
pub const QK8_0: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// The rest of the legacy family: an asymmetric scale+min (Q4_1, Q5_1) or a fifth bit in
/// a separate word (Q5_0, Q5_1), still one scale per 32 values.
pub const GGML_Q4_1: u32 = 3;
pub const GGML_Q5_0: u32 = 6;
pub const GGML_Q5_1: u32 = 7;

/// The k-quant family. A 256-element SUPER-BLOCK whose sub-blocks carry their own scale,
/// packed against one super-block scale -- a different SHAPE of quantisation from Q4_0 /
/// Q8_0's one scale per 32 values, which is why the reader needs the block interior and
/// not just the geometry (`tensor_layout` already has the geometry: 256/144, 256/176,
/// 256/210).
pub const GGML_Q2_K: u32 = 10;
pub const GGML_Q3_K: u32 = 11;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q5_K: u32 = 13;
pub const GGML_Q6_K: u32 = 14;

/// The k-quant super-block width: 256 elements, whatever the bit depth.
pub const QK_K: usize = 256;

/// The IQ family. A third SHAPE: the stored bits are an INDEX into a fixed codebook, not a
/// number, so no arithmetic reproduces the value -- the table is part of the format. An
/// unsloth "UD" mix carries these next to the k-quants (Qwen3.8-27B-UD-Q4_K_M is 117
/// IQ4_XS tensors, 7 IQ4_NL, 4 IQ3_S), so reading k-quants alone does not load such a file.
pub const GGML_IQ2_XS: u32 = 17;
pub const GGML_IQ3_XXS: u32 = 18;
pub const GGML_IQ4_NL: u32 = 20;
pub const GGML_IQ3_S: u32 = 21;
pub const GGML_IQ2_S: u32 = 22;
pub const GGML_IQ4_XS: u32 = 23;

/// Q8_0_TM: the 8-row x 32-element unit, 272 bytes = the unit's eight half scales
/// (16 bytes, row order) followed by its eight 32-byte payload rows `[row 0..8][k 0..32]`.
/// Units are row-tile-major with K blocks adjacent. The scales sit INSIDE their unit
/// (changed 2026-09-04): kept in one array after the whole payload, the decode GEMV
/// streamed two regions per tensor and read 0.7..1.2% slower than row-major; one stream
/// per tensor removes that. The payload starts 16 bytes into the unit, so the prefill tile
/// loads stay 16-byte aligned; the tensor's size is the row-major size. Every address a
/// reader needs comes from `TmRule` so that no kernel, converter or oracle carries its own
/// copy of the rule; these two functions are the Q8 rule's addresses under the old names.
pub const Q8_0_TM_UNIT_ROWS: usize = 8;
/// Bytes per unit: scales + payload.
pub const Q8_0_TM_UNIT_BYTES: usize = Q8_0_TM_UNIT_ROWS * (2 + QK8_0);

/// Byte offset of `row`'s 32 int8 values for K block `block` inside a Q8_0_TM tensor whose
/// rows have `blocks` K blocks.
#[must_use]
pub const fn q8_0_tm_payload_offset(row: usize, block: usize, blocks: usize) -> usize {
    TM_RULES[0].payload_offset(row, block, blocks)
}

/// Byte offset of `row`'s half scale for K block `block`. `n_out` is unused since the
/// scales moved into their unit; kept so callers did not change.
#[must_use]
pub const fn q8_0_tm_scale_offset(
    row: usize,
    block: usize,
    blocks: usize,
    n_out: usize,
) -> usize {
    TM_RULES[0].scale_offset(row, block, blocks, n_out)
}

/// A Q8_0 tensor converts to Q8_0_TM only when its rows fill whole units and its row
/// width fills whole blocks; anything else keeps the row-major layout.
#[must_use]
pub const fn q8_0_tm_convertible(n_in: usize, n_out: usize) -> bool {
    n_in % QK8_0 == 0 && n_out % Q8_0_TM_UNIT_ROWS == 0
}

/// THE REPACK RULE TABLE. One tile-major rule per weight kind: eight row-major blocks of
/// `scale (2 bytes) + payload` (one row tile, one K block) become one unit of
/// `[8 scales][8 payload rows]`; units are row-tile-major with K blocks adjacent. Q8_0 has
/// 32 payload bytes per block (272-byte units), Q4_0 16 (144-byte units). The converter
/// (`imparo-repack`), the load-time transform (`Backend::transform_weights`) and the
/// verify step all take their addresses from here, so no consumer carries its own copy of
/// the layout; the Metal shader's `q8_tm_payload` / `q8_tm_scale` mirror these formulas
/// and are pinned bit-identical by the gates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TmRule {
    /// The row-major ggml type this rule converts from.
    pub from: u32,
    /// The tile-major type it produces.
    pub to: u32,
    pub from_name: &'static str,
    pub to_name: &'static str,
    /// Elements per block: 32 for the legacy family, 256 for a k-quant or IQ super-block.
    pub block_elems: usize,
    /// Bytes one row-major block occupies. `tensor_layout(from)` says the same thing; it
    /// is repeated here so every address below is `const`.
    pub block_bytes_src: usize,
    /// WHERE THE SCALES ARE IN THE SOURCE BLOCK, as byte ranges `(offset, len)` in order.
    ///
    /// This is what makes the rule a FAMILY rather than one layout. Q4_0 and Q8_0 open
    /// with a 2-byte scale, so a length was enough. A k-quant does not: Q6_K keeps its
    /// sub-scales at byte 192 and its super-scale at 208, Q3_K at 96 and 108, and Q2_K
    /// splits them across the front and the back. A rule that assumed "the first N bytes"
    /// would silently treat quant bits as a scale -- plausible garbage, not a crash.
    ///
    /// The scale bytes move to the head of the unit (one aligned load per row); the rest
    /// of the block, in source order, is the payload.
    pub scale_spans: &'static [(usize, usize)],
    /// Rows per unit.
    pub unit_rows: usize,
    /// Backends with kernels that read `to`. A rule with no readers is a layout the
    /// converter can write only on request (`--write-unread`) and the load-time transform
    /// never applies; a file carrying such a kind is refused at load by a backend that
    /// does not serve it, never misread.
    pub readers: &'static [&'static str],
}

impl TmRule {
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        self.block_bytes_src
    }
    /// Scale bytes per block, summed over the spans.
    #[must_use]
    pub const fn scale_bytes(&self) -> usize {
        let mut n = 0;
        let mut i = 0;
        while i < self.scale_spans.len() {
            n += self.scale_spans[i].1;
            i += 1;
        }
        n
    }
    /// Payload bytes per block: everything the scale spans do not claim.
    #[must_use]
    pub const fn payload_bytes(&self) -> usize {
        self.block_bytes_src - self.scale_bytes()
    }
    /// Rows must fill whole units and the row width whole blocks.
    #[must_use]
    pub const fn convertible(&self, n_in: usize, n_out: usize) -> bool {
        n_in % self.block_elems == 0 && n_out % self.unit_rows == 0
    }
    /// Bytes per unit: the unit's scales, then its payload rows. Always the row-major
    /// size of the same blocks -- the layout MOVES bytes, it never adds or drops one, so
    /// a converted tensor is byte-for-byte the same length and the file's offsets hold.
    #[must_use]
    pub const fn unit_bytes(&self) -> usize {
        self.unit_rows * self.block_bytes_src
    }
    const fn unit_start(&self, row: usize, block: usize, blocks: usize) -> usize {
        ((row / self.unit_rows) * blocks + block) * self.unit_bytes()
    }
    /// Byte offset of `row`'s payload for K block `block` (`blocks` K blocks per row):
    /// after the unit's scales.
    #[must_use]
    pub const fn payload_offset(
        &self,
        row: usize,
        block: usize,
        blocks: usize,
    ) -> usize {
        self.unit_start(row, block, blocks)
            + self.unit_rows * self.scale_bytes()
            + (row % self.unit_rows) * self.payload_bytes()
    }
    /// Byte offset of `row`'s half scale for K block `block`: at the head of its unit.
    /// `_n_out` is no longer needed by the layout; kept so the signature is stable.
    #[must_use]
    pub const fn scale_offset(
        &self,
        row: usize,
        block: usize,
        blocks: usize,
        _n_out: usize,
    ) -> usize {
        self.unit_start(row, block, blocks) + (row % self.unit_rows) * self.scale_bytes()
    }
    /// The byte ranges the scale spans do NOT claim, in source order -- the payload.
    ///
    /// Derived rather than written: a second list would be a second chance to disagree
    /// with the first, and the disagreement reads as a slightly wrong weight.
    #[must_use]
    pub fn payload_spans(&self) -> Vec<(usize, usize)> {
        let mut spans: Vec<(usize, usize)> = self.scale_spans.to_vec();
        spans.sort_unstable();
        let mut out = Vec::new();
        let mut at = 0_usize;
        for (off, len) in spans {
            assert!(at <= off, "{}: scale spans overlap", self.to_name);
            if off > at {
                out.push((at, off - at));
            }
            at = off + len;
        }
        assert!(
            at <= self.block_bytes_src,
            "{}: a scale span runs past the block",
            self.to_name
        );
        if at < self.block_bytes_src {
            out.push((at, self.block_bytes_src - at));
        }
        out
    }

    /// Row-major bytes of one whole tensor -> tile-major bytes, same length.
    #[must_use]
    pub fn convert(&self, src: &[u8], n_in: usize, n_out: usize) -> Vec<u8> {
        let blocks = n_in / self.block_elems;
        let bb = self.block_bytes();
        let mut out = vec![0_u8; src.len()];
        for r in 0..n_out {
            for b in 0..blocks {
                let blk = &src[(r * blocks + b) * bb..][..bb];
                // Scales first, in span order, at the head of the unit.
                let mut at = self.scale_offset(r, b, blocks, n_out);
                for &(off, len) in self.scale_spans {
                    out[at..at + len].copy_from_slice(&blk[off..off + len]);
                    at += len;
                }
                // Then everything the spans did not claim, in SOURCE order.
                let mut at = self.payload_offset(r, b, blocks);
                for (off, len) in self.payload_spans() {
                    out[at..at + len].copy_from_slice(&blk[off..off + len]);
                    at += len;
                }
            }
        }
        out
    }
    /// Every payload byte and every scale of `tm`, read back at the rule's address, must
    /// equal the row-major block's bytes. Byte equality, no float arithmetic.
    ///
    /// # Errors
    /// The first row/block that differs, by name.
    pub fn verify(
        &self,
        src: &[u8],
        tm: &[u8],
        n_in: usize,
        n_out: usize,
        name: &str,
    ) -> Result<(), String> {
        let blocks = n_in / self.block_elems;
        let bb = self.block_bytes();
        if tm.len() != src.len() {
            return Err(format!(
                "{name}: {} tile-major bytes for {} row-major",
                tm.len(),
                src.len()
            ));
        }
        for r in 0..n_out {
            for b in 0..blocks {
                let blk = &src[(r * blocks + b) * bb..][..bb];
                let mut at = self.scale_offset(r, b, blocks, n_out);
                for &(off, len) in self.scale_spans {
                    if tm[at..at + len] != blk[off..off + len] {
                        return Err(format!("{name}: scale differs at row {r} block {b}"));
                    }
                    at += len;
                }
                let mut at = self.payload_offset(r, b, blocks);
                for (off, len) in self.payload_spans() {
                    if tm[at..at + len] != blk[off..off + len] {
                        return Err(format!("{name}: values differ at row {r} block {b}"));
                    }
                    at += len;
                }
            }
        }
        Ok(())
    }
}

/// The tile-major twin of every block format the engine reads. Ids are imparo-private and
/// follow the source's ggml id (1000 + it) so a reader can recover the source at a glance.
pub const GGML_Q4_1_TM: u32 = 1003;
pub const GGML_Q5_0_TM: u32 = 1006;
pub const GGML_Q5_1_TM: u32 = 1007;
pub const GGML_Q2_K_TM: u32 = 1010;
pub const GGML_Q3_K_TM: u32 = 1011;
pub const GGML_Q4_K_TM: u32 = 1012;
pub const GGML_Q5_K_TM: u32 = 1013;
pub const GGML_Q6_K_TM: u32 = 1014;
pub const GGML_IQ2_XS_TM: u32 = 1017;
pub const GGML_IQ3_XXS_TM: u32 = 1018;
pub const GGML_IQ4_NL_TM: u32 = 1020;
pub const GGML_IQ3_S_TM: u32 = 1021;
pub const GGML_IQ2_S_TM: u32 = 1022;
pub const GGML_IQ4_XS_TM: u32 = 1023;

/// THE LAYOUT FAMILY. One rule per readable block format; every rule is the SAME move --
/// the block's scale bytes to the head of an 8-row unit, its payload after them -- so a
/// reader's address arithmetic is one formula and only the DECODE differs, by family:
///
/// ```text
///   affine      value = q * scale + min          Q4_1 Q5_1 Q2_K Q4_K Q5_K
///   symmetric   value = (q - bias) * scale       Q4_0 Q5_0 Q8_0 Q3_K Q6_K
///   codebook    value = table[q] * scale         IQ4_NL IQ4_XS IQ3_S
/// ```
///
/// Byte count is UNCHANGED by construction (`unit_bytes` is the row-major size of the same
/// blocks), so a converted tensor keeps its file offset and a load-time transform needs no
/// extra memory beyond the target buffer. What changes is the ADDRESS a lane touches: one
/// aligned scale load per row per block instead of a strided read plus a bitfield unpack
/// spread over the block.
///
/// `readers` is the gate. A rule with none is a layout the converter writes only on
/// request and the load-time transform never applies, so a backend that cannot decode a
/// family refuses the file by name instead of misreading it.
pub const TM_RULES: [TmRule; 16] = [
    TmRule {
        from: GGML_Q8_0,
        to: GGML_Q8_0_TM,
        from_name: "Q8_0",
        to_name: "Q8_0_TM",
        block_elems: QK8_0,
        block_bytes_src: Q8_0_BLOCK_BYTES,
        scale_spans: &[(0, 2)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_Q4_0,
        to: GGML_Q4_0_TM,
        from_name: "Q4_0",
        to_name: "Q4_0_TM",
        block_elems: QK4_0,
        block_bytes_src: Q4_0_BLOCK_BYTES,
        scale_spans: &[(0, 2)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &[],
    },
    TmRule {
        from: GGML_Q4_1,
        to: GGML_Q4_1_TM,
        from_name: "Q4_1",
        to_name: "Q4_1_TM",
        block_elems: 32,
        block_bytes_src: 20,
        // d, m.
        scale_spans: &[(0, 4)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &[],
    },
    TmRule {
        from: GGML_Q5_0,
        to: GGML_Q5_0_TM,
        from_name: "Q5_0",
        to_name: "Q5_0_TM",
        block_elems: 32,
        block_bytes_src: 22,
        // d only: qh is a fifth QUANT bit, not a scale, so it stays in the payload.
        scale_spans: &[(0, 2)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &[],
    },
    TmRule {
        from: GGML_Q5_1,
        to: GGML_Q5_1_TM,
        from_name: "Q5_1",
        to_name: "Q5_1_TM",
        block_elems: 32,
        block_bytes_src: 24,
        scale_spans: &[(0, 4)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &[],
    },
    TmRule {
        from: GGML_Q2_K,
        to: GGML_Q2_K_TM,
        from_name: "Q2_K",
        to_name: "Q2_K_TM",
        block_elems: QK_K,
        block_bytes_src: 84,
        // TWO spans: the 16 packed sub-scale bytes open the block and (d, dmin) close it.
        scale_spans: &[(0, 16), (80, 4)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_Q3_K,
        to: GGML_Q3_K_TM,
        from_name: "Q3_K",
        to_name: "Q3_K_TM",
        block_elems: QK_K,
        block_bytes_src: 110,
        // scales[12] then d, at the END of the block -- hmask and qs come first.
        scale_spans: &[(96, 14)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_Q4_K,
        to: GGML_Q4_K_TM,
        from_name: "Q4_K",
        to_name: "Q4_K_TM",
        block_elems: QK_K,
        block_bytes_src: 144,
        // d, dmin, scales[12] -- sixteen bytes, so a row's whole scale header is ONE
        // aligned 16-byte load where row-major spread it over a 144-byte stride.
        scale_spans: &[(0, 16)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_Q5_K,
        to: GGML_Q5_K_TM,
        from_name: "Q5_K",
        to_name: "Q5_K_TM",
        block_elems: QK_K,
        block_bytes_src: 176,
        scale_spans: &[(0, 16)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_Q6_K,
        to: GGML_Q6_K_TM,
        from_name: "Q6_K",
        to_name: "Q6_K_TM",
        block_elems: QK_K,
        block_bytes_src: 210,
        // sc[16] (signed) then d, at byte 192 -- ql and qh come first.
        scale_spans: &[(192, 18)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_IQ4_NL,
        to: GGML_IQ4_NL_TM,
        from_name: "IQ4_NL",
        to_name: "IQ4_NL_TM",
        block_elems: 32,
        block_bytes_src: 18,
        scale_spans: &[(0, 2)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_IQ3_S,
        to: GGML_IQ3_S_TM,
        from_name: "IQ3_S",
        to_name: "IQ3_S_TM",
        block_elems: QK_K,
        block_bytes_src: 110,
        // d at the front, the 4 sub-scale bytes at the back; qs / qh / signs between.
        // The ONLY rule whose scales are two spans. The tile-major layout concatenates
        // them into 6 contiguous bytes, so the Metal brick's single scale pointer works;
        // a ROW-MAJOR IQ3_S row has no such pointer, which is why the kind is served only
        // as IQ3_S_TM and a row-gathered IQ3_S tensor is refused at load rather than read
        // through an address the format cannot express.
        scale_spans: &[(0, 2), (106, 4)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_IQ4_XS,
        to: GGML_IQ4_XS_TM,
        from_name: "IQ4_XS",
        to_name: "IQ4_XS_TM",
        block_elems: QK_K,
        block_bytes_src: 136,
        // d, scales_h, scales_l[4].
        scale_spans: &[(0, 8)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_IQ2_XS,
        to: GGML_IQ2_XS_TM,
        from_name: "IQ2_XS",
        to_name: "IQ2_XS_TM",
        block_elems: QK_K,
        block_bytes_src: 74,
        // d at the front, the 8 sub-scale bytes at the back; the 32 index words between.
        scale_spans: &[(0, 2), (66, 8)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_IQ2_S,
        to: GGML_IQ2_S_TM,
        from_name: "IQ2_S",
        to_name: "IQ2_S_TM",
        block_elems: QK_K,
        block_bytes_src: 82,
        // d, then the two 8-byte trailers -- qh (the index's top bits) and the sub-scales.
        // The payload is the 32 index bytes and the 32 sign bytes, in source order.
        scale_spans: &[(0, 2), (66, 8), (74, 8)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
    TmRule {
        from: GGML_IQ3_XXS,
        to: GGML_IQ3_XXS_TM,
        from_name: "IQ3_XXS",
        to_name: "IQ3_XXS_TM",
        block_elems: QK_K,
        block_bytes_src: 98,
        // d, then the eight scale-and-sign words: the sub-block scale lives in the top
        // nibble of the word that also carries that sub-block's four sign indices, so the
        // signs travel with the scales, not with the payload.
        scale_spans: &[(0, 2), (66, 32)],
        unit_rows: Q8_0_TM_UNIT_ROWS,
        readers: &["metal"],
    },
];

/// The rule that converts FROM this row-major type, if any.
#[must_use]
pub fn tm_rule_for(ggml_type: u32) -> Option<&'static TmRule> {
    TM_RULES.iter().find(|r| r.from == ggml_type)
}

/// The rule that produced this tile-major type, if any.
#[must_use]
pub fn tm_rule_to(ggml_type: u32) -> Option<&'static TmRule> {
    TM_RULES.iter().find(|r| r.to == ggml_type)
}

/// Tensors read by ROW keep the row-major layout: the embedding gather and the per-layer
/// token-embedding table (gemma4's PLE, staged by rows on the host) want a row's bytes
/// contiguous, and they never go through the GEMM. The role is the tensor's name, the
/// same on every model this engine loads.
pub const ROW_MAJOR_ROLES: &[&str] =
    &["token_embd.weight", "per_layer_token_embd.weight"];

/// Why a tensor keeps its layout, or the rule that converts it. The single decision the
/// converter and the load-time transform both make; `ne` is the tensor's dimensions
/// (ne[0] = row width, ne[1] = rows).
#[must_use]
pub fn tm_applies(
    name: &str,
    ggml_type: u32,
    ne: &[u64],
) -> Result<&'static TmRule, &'static str> {
    let Some(rule) = tm_rule_for(ggml_type) else {
        return Err("no tile-major rule for this type");
    };
    if ne.len() != 2 && !(ne.len() > 2 && ne[2..].iter().all(|&d| d == 1)) {
        return Err("not 2-D");
    }
    if ROW_MAJOR_ROLES.contains(&name) {
        return Err("read by row (embedding)");
    }
    if !rule.convertible(ne[0] as usize, ne[1] as usize) {
        return Err("rows not a multiple of 8 or width not a multiple of 32");
    }
    Ok(rule)
}

/// The weight-type -> kernel table, as a type (task #17). Every matmul weight must map
/// to a variant; an unmapped ggml type is REJECTED AT LOAD with the tensor's name --
/// never silently misread by a kernel built for another layout. Adding a weight quant
/// means: a variant here, one dequant/stage implementation per backend, and a table
/// entry in each backend's dispatch -- no call-site changes.
///
/// The discriminants are the wire values the backends receive (imparo_metal_matmat's
/// `wkind`), so keep them stable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
#[allow(non_camel_case_types)] // the variants are the GGUF type names, and the TM suffix is the layout
pub enum WeightKind {
    F32 = 0,
    Q4_0 = 1,
    Q8_0 = 2,
    /// Q8_0 values, tile-major (`GGML_Q8_0_TM`); written by `imparo-repack` or produced
    /// at load by the fast-tier transform.
    Q8_0_TM = 3,
    /// Q4_0 values, tile-major (`GGML_Q4_0_TM`). The rule exists; no backend reads it yet.
    Q4_0_TM = 4,
    /// The k-quants: 256-element super-blocks with per-sub-block scales.
    Q4_K = 5,
    Q5_K = 6,
    Q6_K = 7,
    Q2_K = 8,
    Q3_K = 9,
    /// The rest of the legacy family.
    Q4_1 = 10,
    Q5_0 = 11,
    Q5_1 = 12,
    /// The IQ family: the quant is a codebook index.
    IQ4_NL = 13,
    IQ4_XS = 14,
    IQ3_S = 15,
    /// The tile-major twin of each of the above. A backend is told WHICH LAYOUT it is
    /// getting, because the address arithmetic differs even though the decode does not;
    /// `imparo-repack` also writes these kinds to a file, so the loader must name them.
    Q4_1_TM = 16,
    Q5_0_TM = 17,
    Q5_1_TM = 18,
    Q2_K_TM = 19,
    Q3_K_TM = 20,
    Q4_K_TM = 21,
    Q5_K_TM = 22,
    Q6_K_TM = 23,
    IQ4_NL_TM = 24,
    IQ4_XS_TM = 25,
    IQ3_S_TM = 26,
    /// The sub-3-bit IQ types an unsloth "UD" mix reaches for on the tensors it can
    /// afford to spend least on. Same codebook shape as IQ3_S, fewer bits per index.
    IQ2_XS = 27,
    IQ3_XXS = 28,
    IQ2_S = 29,
    IQ2_XS_TM = 30,
    IQ3_XXS_TM = 31,
    IQ2_S_TM = 32,
}

/// Names of every type `weight_kind` accepts, for the error message that rejects the
/// others -- DERIVED from `weight_kind` and the layout table, never written out. The hand
/// written version had already gone stale once (it still read "F32, Q4_0" after Q8_0
/// landed, in the message a user sees when a file is refused), and a list that can
/// disagree with the code it describes is worse than no list.
#[must_use]
pub fn supported_weight_types() -> String {
    WEIGHT_KINDS
        .iter()
        .filter_map(|(t, _)| crate::tensor_layout(*t).ok().map(|l| l.name))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The one table that pairs a GGUF ggml type with its wire weight kind.
///
/// TWO numbering schemes meet at the `Backend` seam: a file names a ggml type, the trait
/// passes the compact wire value (a kernel table index), and a backend's readers switch
/// back on the ggml type. Both directions are read off this table so there is no second
/// copy to disagree with it -- a backend that kept its own half of the mapping read a Q8
/// matrix as Q4 the moment a type was added, and that is plausible garbage, not a crash.
///
/// Adding a format is one row here plus its codec; nothing else enumerates the list.
const WEIGHT_KINDS: &[(u32, WeightKind)] = &[
    (GGML_F32, WeightKind::F32),
    (GGML_Q4_0, WeightKind::Q4_0),
    (GGML_Q4_1, WeightKind::Q4_1),
    (GGML_Q5_0, WeightKind::Q5_0),
    (GGML_Q5_1, WeightKind::Q5_1),
    (GGML_Q8_0, WeightKind::Q8_0),
    (GGML_Q2_K, WeightKind::Q2_K),
    (GGML_Q3_K, WeightKind::Q3_K),
    (GGML_Q4_K, WeightKind::Q4_K),
    (GGML_Q5_K, WeightKind::Q5_K),
    (GGML_Q6_K, WeightKind::Q6_K),
    (GGML_IQ4_NL, WeightKind::IQ4_NL),
    (GGML_IQ3_S, WeightKind::IQ3_S),
    (GGML_IQ4_XS, WeightKind::IQ4_XS),
    (GGML_IQ2_XS, WeightKind::IQ2_XS),
    (GGML_IQ3_XXS, WeightKind::IQ3_XXS),
    (GGML_IQ2_S, WeightKind::IQ2_S),
    (GGML_Q8_0_TM, WeightKind::Q8_0_TM),
    (GGML_Q4_0_TM, WeightKind::Q4_0_TM),
    (GGML_Q4_1_TM, WeightKind::Q4_1_TM),
    (GGML_Q5_0_TM, WeightKind::Q5_0_TM),
    (GGML_Q5_1_TM, WeightKind::Q5_1_TM),
    (GGML_Q2_K_TM, WeightKind::Q2_K_TM),
    (GGML_Q3_K_TM, WeightKind::Q3_K_TM),
    (GGML_Q4_K_TM, WeightKind::Q4_K_TM),
    (GGML_Q5_K_TM, WeightKind::Q5_K_TM),
    (GGML_Q6_K_TM, WeightKind::Q6_K_TM),
    (GGML_IQ4_NL_TM, WeightKind::IQ4_NL_TM),
    (GGML_IQ4_XS_TM, WeightKind::IQ4_XS_TM),
    (GGML_IQ3_S_TM, WeightKind::IQ3_S_TM),
    (GGML_IQ2_XS_TM, WeightKind::IQ2_XS_TM),
    (GGML_IQ3_XXS_TM, WeightKind::IQ3_XXS_TM),
    (GGML_IQ2_S_TM, WeightKind::IQ2_S_TM),
];

/// Every tile-major rule must have a wire kind, or a backend cannot be told which layout
/// it is being handed. Derived check rather than a written list.
#[cfg(test)]
mod wire_kind_tests {
    use super::*;
    #[test]
    fn every_tm_rule_has_a_wire_kind() {
        for rule in &TM_RULES {
            assert!(
                weight_kind(rule.to).is_some(),
                "{} has no WeightKind",
                rule.to_name
            );
            assert!(
                weight_kind(rule.from).is_some(),
                "{} has no WeightKind",
                rule.from_name
            );
        }
    }
}

/// Maps a GGUF ggml type id to the kernel table, or None for anything unsupported.
#[must_use]
pub fn weight_kind(ggml_type: u32) -> Option<WeightKind> {
    WEIGHT_KINDS
        .iter()
        .find(|(t, _)| *t == ggml_type)
        .map(|(_, k)| *k)
}

/// Every (wire kind, ggml type) pair, for a backend that must be told the mapping rather
/// than carry a copy of it.
#[must_use]
pub fn wire_kind_types() -> Vec<(u32, u32)> {
    WEIGHT_KINDS.iter().map(|(t, k)| (*k as u32, *t)).collect()
}

/// The ggml type a wire weight kind names -- the inverse of `weight_kind`.
///
/// # Panics
/// On a wire value no `WeightKind` has. That is a seam defect (a backend inventing a
/// kind, or a stale binary on one side), and reading on is how a wrong matrix becomes a
/// plausible answer.
#[must_use]
pub fn ggml_type_of_wire(wire: u32) -> u32 {
    WEIGHT_KINDS
        .iter()
        .find(|(_, k)| *k as u32 == wire)
        .map(|(t, _)| *t)
        .unwrap_or_else(|| panic!("unknown weight kind wire value {wire}"))
}

/// The same lookup for a caller that must not die on an unknown value -- a TOOL reading a
/// kind out of a file or a config rather than off the engine's own seam. The panicking form
/// above stays the default: inside the engine an unknown wire kind IS a seam defect, and
/// reading on is how a wrong matrix becomes a plausible answer.
#[must_use]
pub fn ggml_type_of_wire_opt(wire: u32) -> Option<u32> {
    WEIGHT_KINDS
        .iter()
        .find(|(_, k)| *k as u32 == wire)
        .map(|(t, _)| *t)
}

#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub offset: usize,
    pub bytes: usize,
    pub ggml_type: u32,
    /// ne[0] is the fastest-varying dimension. For a projection weight this is the INPUT
    /// width and ne[1] the output width, so `mul_mat(W, x)` yields ne[1] values.
    pub ne: [u64; 4],
    pub n_dims: u32,
}

impl Tensor {
    #[must_use]
    pub fn ne0(&self) -> usize {
        self.ne[0] as usize
    }
    #[must_use]
    pub fn ne1(&self) -> usize {
        self.ne[1] as usize
    }
    #[must_use]
    pub fn elements(&self) -> usize {
        self.ne.iter().take(self.n_dims as usize).product::<u64>() as usize
    }
}

/// Maps an open file read-only, returning the base pointer. One implementation per
/// OS family, used by `Mapping` and `Weights` alike -- the unix mmap used to be
/// duplicated at both sites and the windows branch DID NOT EXIST, so the "portable"
/// crates could never have compiled on windows (caught preparing the first windows CI).
fn map_file(
    file: &std::fs::File,
    len: usize,
) -> Result<*const u8, Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        // SAFETY: fd is valid and open for reading; PROT_READ/MAP_PRIVATE is sound.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        Ok(p.cast::<u8>().cast_const())
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        // Minimal kernel32 FFI (repo rule: no binding-library dependencies).
        type Handle = *mut core::ffi::c_void;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CreateFileMappingW(
                file: Handle,
                attrs: *mut core::ffi::c_void,
                protect: u32,
                size_hi: u32,
                size_lo: u32,
                name: *const u16,
            ) -> Handle;
            fn MapViewOfFile(
                mapping: Handle,
                access: u32,
                off_hi: u32,
                off_lo: u32,
                len: usize,
            ) -> *mut core::ffi::c_void;
            fn CloseHandle(h: Handle) -> i32;
        }
        const PAGE_READONLY: u32 = 0x02;
        const FILE_MAP_READ: u32 = 0x04;
        let _ = len;
        // SAFETY: the handle is a valid open file; the mapping object is closed right
        // after the view is created (the view keeps the mapping alive on windows).
        unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle().cast(),
                std::ptr::null_mut(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return Err("CreateFileMapping failed".into());
            }
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
            CloseHandle(mapping);
            if view.is_null() {
                return Err("MapViewOfFile failed".into());
            }
            Ok(view.cast::<u8>().cast_const())
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, len);
        Err("no file-mapping implementation for this OS".into())
    }
}

fn unmap_file(base: *const u8, len: usize) {
    #[cfg(unix)]
    // SAFETY: base/len came from a successful map_file and are unmapped exactly once.
    unsafe {
        libc::munmap(base.cast_mut().cast::<libc::c_void>(), len);
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn UnmapViewOfFile(base: *const core::ffi::c_void) -> i32;
        }
        let _ = len;
        // SAFETY: base came from MapViewOfFile and is unmapped exactly once.
        unsafe {
            UnmapViewOfFile(base.cast());
        }
    }
    #[cfg(not(any(unix, windows)))]
    let _ = (base, len);
}

/// A read-only file mapping.
///
/// File-backed pages are not charged to the process footprint: a mapped copy of the
/// weights costs disk and nothing measurable.
pub struct Mapping {
    base: *const u8,
    len: usize,
}

// SAFETY: the mapping is read-only and immutable for its whole lifetime.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Maps a file read-only.
    ///
    /// # Errors
    /// Returns an error when the file cannot be opened or mapped.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err("empty mapping".into());
        }
        Ok(Self {
            base: map_file(&file, len)?,
            len,
        })
    }

    #[must_use]
    pub fn base(&self) -> *const u8 {
        self.base
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unmap_file(self.base, self.len);
    }
}

pub struct Weights {
    base: *const u8,
    len: usize,
    source_path: PathBuf,
    full_file_sha256: OnceLock<[u8; 32]>,
    pub tensors: BTreeMap<String, Tensor>,
    /// GPU dispatch is opt-in per run so the CPU path stays available as the oracle.
    gpu: bool,
}

// SAFETY: the mapping is read-only and lives for the object's lifetime; no interior mutation.
unsafe impl Send for Weights {}
unsafe impl Sync for Weights {}

impl Weights {
    /// Maps a GGUF file read-only and indexes its tensors by name.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened, mapped, or parsed.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let document = crate::read(path)?;
        Self::open_with(&document, path)
    }

    /// Maps the file using an already-parsed document.
    ///
    /// Parsing a 4 GB GGUF header costs ~700 ms, and the server had three callers doing it
    /// independently.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be opened or mapped.
    pub fn open_with(
        document: &crate::Document,
        path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let base = map_file(&file, len)?;
        let mut tensors = BTreeMap::new();
        for t in &document.tensors {
            let mut ne = [1_u64; 4];
            for (i, d) in t.dimensions.iter().take(4).enumerate() {
                ne[i] = *d;
            }
            tensors.insert(
                t.name.clone(),
                Tensor {
                    offset: t.absolute_offset as usize,
                    bytes: t.byte_size as usize,
                    ggml_type: t.ggml_type,
                    ne,
                    n_dims: t.dimensions.len() as u32,
                },
            );
        }
        // Sharing/uploading the mapping to a GPU backend is the COMPOSITION ROOT's job
        // (imparo-model::backend + the bins/server), so this crate knows no backend
        // exists. The caller flips `gpu` via mark_gpu() after a backend's init_weights
        // succeeds.
        Ok(Self {
            base,
            len,
            source_path: path.to_path_buf(),
            full_file_sha256: OnceLock::new(),
            tensors,
            gpu: false,
        })
    }

    /// CPU-visible base of the weight mapping, for a backend's init_weights.
    #[must_use]
    pub fn base_ptr(&self) -> *const u8 {
        self.base
    }
    /// Byte length of the mapping.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.len as u64
    }
    /// Path originally used to open this mapping.
    ///
    /// This is provenance for diagnostics and discovery, not a request to reopen the
    /// pathname: identity is computed from the mapping already held by `Weights`.
    #[must_use]
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    /// SHA-256 of every byte in the held GGUF mapping.
    ///
    /// The first call touches the complete model and caches the result. Callers should
    /// therefore invoke it only after discovering a candidate tuned configuration that
    /// needs validation; ordinary startup performs no model-sized hashing pass.
    #[must_use]
    pub fn full_file_sha256(&self) -> [u8; 32] {
        *self.full_file_sha256.get_or_init(|| {
            // SAFETY: `base..base+len` is the immutable read-only mapping owned by self.
            let bytes = unsafe { std::slice::from_raw_parts(self.base, self.len) };
            Sha256::digest(bytes).into()
        })
    }

    /// Lowercase hexadecimal form of [`Self::full_file_sha256`].
    #[must_use]
    pub fn full_file_sha256_hex(&self) -> String {
        self.full_file_sha256().iter().fold(
            String::with_capacity(64),
            |mut out, byte| {
                use std::fmt::Write as _;
                let _ = write!(out, "{byte:02x}");
                out
            },
        )
    }
    /// The composition root calls this once a GPU backend has taken the weights.
    pub fn mark_gpu(&mut self) {
        self.gpu = true;
    }

    /// True when a GPU backend holds these weights and the GPU path may run.
    #[must_use]
    pub fn gpu_enabled(&self) -> bool {
        self.gpu
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// A tensor's bytes, without copying.
    ///
    /// # Panics
    /// Panics when the tensor lies outside the mapping.
    #[must_use]
    pub fn raw(&self, t: &Tensor) -> &[u8] {
        assert!(
            t.offset + t.bytes <= self.len,
            "tensor {t:?} outside mapping"
        );
        // SAFETY: bounds checked above; mapping is read-only for the object's lifetime.
        unsafe { std::slice::from_raw_parts(self.base.add(t.offset), t.bytes) }
    }

    /// Reads an F32 tensor as a slice without copying.
    ///
    /// # Panics
    ///
    /// Panics when the tensor is not F32.
    #[must_use]
    // GGUF tensor data starts at the header's declared alignment (32 bytes), so
    // the f32 view is aligned by the container's own contract (cast_ptr_alignment).
    #[allow(clippy::cast_ptr_alignment)]
    pub fn f32s(&self, t: &Tensor) -> &[f32] {
        assert_eq!(t.ggml_type, GGML_F32, "expected F32 tensor");
        let bytes = self.raw(t);
        // SAFETY: GGUF guarantees 4-byte alignment for F32 tensor data via its alignment field.
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4)
        }
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        unmap_file(self.base, self.len);
    }
}

/// Round to IEEE binary16 and back, the precision a GPU KV cache carries.
///
/// Round to nearest, ties to even -- a truncating shift biases every value the same way,
/// which would make any comparison built on this meaningless.
#[must_use]
pub fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

#[must_use]
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mant = x & 0x007F_FFFF;
    let exp = ((x >> 23) & 0xFF) as i32;
    if exp == 0xFF {
        // inf or nan
        return sign | 0x7C00 | (u16::from(mant != 0) * 0x0200);
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    } // overflows to infinity
    if e <= 0 {
        if e < -10 {
            return sign;
        } // underflows to zero
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut out = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let mut m = mant >> 13;
    let rem = mant & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
    }
    let mut ee = e as u32;
    if m == 0x400 {
        m = 0;
        ee += 1;
    }
    if ee >= 0x1F {
        return sign | 0x7C00;
    }
    sign | ((ee as u16) << 10) | m as u16
}

#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: renormalise
            let mut e = -1_i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            // 114 + e, not 113 + e. A subnormal is mant * 2^-24; after normalising with s
            // left shifts, value = (1+f) * 2^(-14-s), so the f32 exponent field is 113 - s,
            // and e = -1 - s makes that 114 + e. The old form returned exactly half.
            sign | (((127 - 15 + e + 2) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn minimal_gguf(marker: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(32, marker);
        bytes
    }

    fn temp_model(bytes: &[u8]) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "imparo-weights-identity-{}-{id}.gguf",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn full_identity_is_lazy_cached_and_uses_the_held_mapping() {
        let bytes = minimal_gguf(0xA5);
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        let path = temp_model(&bytes);
        {
            let weights = Weights::open(&path).unwrap();
            assert_eq!(weights.source_path(), path);
            assert!(weights.full_file_sha256.get().is_none());
            assert_eq!(weights.full_file_sha256(), expected);
            assert_eq!(weights.full_file_sha256.get(), Some(&expected));
            assert_eq!(weights.full_file_sha256_hex().len(), 64);
            assert_eq!(weights.full_file_sha256(), expected);
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn identity_covers_bytes_outside_the_parsed_directory() {
        let first_path = temp_model(&minimal_gguf(0x11));
        let second_path = temp_model(&minimal_gguf(0x22));
        let first = Weights::open(&first_path).unwrap();
        let second = Weights::open(&second_path).unwrap();
        assert_ne!(first.full_file_sha256(), second.full_file_sha256());
        drop((first, second));
        fs::remove_file(first_path).unwrap();
        fs::remove_file(second_path).unwrap();
    }
}

#[cfg(test)]
mod tm_rule_tests {
    use super::*;

    /// Every rule round-trips a synthetic tensor: convert, verify byte-for-byte, and the
    /// two address functions cover the whole tensor exactly once.
    #[test]
    fn rules_round_trip_and_cover() {
        for rule in &TM_RULES {
            let (n_in, n_out) = (rule.block_elems * 3, rule.unit_rows * 2);
            let blocks = n_in / rule.block_elems;
            let bb = rule.block_bytes();
            let src: Vec<u8> = (0..n_out * blocks * bb)
                .map(|i| (i * 7 + 3) as u8)
                .collect();
            let tm = rule.convert(&src, n_in, n_out);
            rule.verify(&src, &tm, n_in, n_out, rule.to_name).unwrap();
            let mut hit = vec![0_u8; src.len()];
            for r in 0..n_out {
                for b in 0..blocks {
                    let p = rule.payload_offset(r, b, blocks);
                    for i in 0..rule.payload_bytes() {
                        hit[p + i] += 1;
                    }
                    let s = rule.scale_offset(r, b, blocks, n_out);
                    for i in 0..rule.scale_bytes() {
                        hit[s + i] += 1;
                    }
                }
            }
            assert!(
                hit.iter().all(|&h| h == 1),
                "{}: addresses do not tile the tensor",
                rule.to_name
            );
            assert!(rule.convertible(n_in, n_out));
            assert!(!rule.convertible(n_in + 1, n_out));
            assert!(!rule.convertible(n_in, n_out + 1));
        }
    }

    /// The rule table and the geometry table describe the same blocks, so they must agree.
    /// `block_bytes_src` is repeated in the rule only to keep the address arithmetic
    /// `const`; a copy that drifts would put every payload at a wrong offset.
    #[test]
    fn every_rule_agrees_with_the_layout_table() {
        for rule in &TM_RULES {
            let src = crate::tensor_layout(rule.from).expect("rule source has a layout");
            assert_eq!(
                (src.block_elements as usize, src.block_bytes as usize),
                (rule.block_elems, rule.block_bytes_src),
                "{} geometry disagrees with tensor_layout",
                rule.from_name
            );
            // A tile-major kind keeps its source's geometry by construction: the layout
            // MOVES bytes. If the two ever differ, every size and bounds check is wrong.
            let dst = crate::tensor_layout(rule.to).expect("rule target has a layout");
            assert_eq!(
                (dst.block_elements, dst.block_bytes),
                (src.block_elements, src.block_bytes),
                "{} is not the same size as {}",
                rule.to_name,
                rule.from_name
            );
            // Spans partition the block: payload_spans() asserts no overlap and no
            // overrun, and the two counts must add up to the whole block.
            assert_eq!(
                rule.scale_bytes() + rule.payload_bytes(),
                rule.block_bytes_src,
                "{}: scale + payload is not the block",
                rule.to_name
            );
        }
    }

    #[test]
    fn q8_rule_matches_the_legacy_functions() {
        let rule = tm_rule_for(GGML_Q8_0).unwrap();
        for (r, b, blocks, n_out) in [(0, 0, 4, 16), (7, 3, 4, 16), (9, 1, 8, 24)] {
            assert_eq!(
                rule.payload_offset(r, b, blocks),
                q8_0_tm_payload_offset(r, b, blocks)
            );
            assert_eq!(
                rule.scale_offset(r, b, blocks, n_out),
                q8_0_tm_scale_offset(r, b, blocks, n_out)
            );
        }
    }

    #[test]
    fn applies_by_role_and_shape() {
        assert!(tm_applies("blk.0.ffn_up.weight", GGML_Q8_0, &[2048, 8192]).is_ok());
        assert_eq!(
            tm_applies("blk.0.ffn_up.weight", GGML_Q4_0, &[2048, 8192])
                .unwrap()
                .to,
            GGML_Q4_0_TM
        );
        assert!(tm_applies("token_embd.weight", GGML_Q8_0, &[2048, 65536]).is_err());
        assert!(tm_applies("blk.0.attn_norm.weight", GGML_F32, &[2048]).is_err());
        assert!(tm_applies("blk.0.x.weight", GGML_Q8_0, &[2048, 12]).is_err());
    }
}
