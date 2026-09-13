// THE PROGRAMS (task #158 step 3): what each architecture's layer DOES, as host data. A phase
// names one always-inline wrapper in imparo.metal (the bricks are the instruction set), the entry
// word that enables it, and whether a grid barrier follows. build.rs emits the kernel from this:
// the frame (prologue, entry loop, exit) is written once here and shared by every architecture,
// so a new architecture is a program, not a new kernel.
//
// Rules the emitted kernel must keep (measured, see the design doc):
//  - a phase wrapper is always_inline: a real call costs co-residency (constraint 8, #158 3a).
//  - a phase never takes a grid barrier itself; barriers belong to the sequence, which is what
//    makes the sequence data.
//  - a phase's condition is an ENTRY word, so it is uniform across the grid; a conditional
//    barrier is therefore safe (every threadgroup takes the same branch).
pub struct MegaPhase {
    /// `""` = always; otherwise the C expression that enables it (an entry word).
    pub when: &'static str,
    /// The wrapper's name; called as `NAME<HD, KVW>(MEGA_PH_CALL)`.
    pub call: &'static str,
    /// The wrapper returns bool and false means "this entry cannot run on this pipeline":
    /// the sequence reports the entry and leaves.
    pub fallible: bool,
    /// What separates this phase from the next.
    pub barrier: Barrier,
    /// THE ENTRY WORD NAMING HOW MANY OUTPUT ROWS THIS PHASE WRITES, or `""` when the phase
    /// does not stride tile-major units over the grid. The host turns rows into items (a unit
    /// is TM_UNIT_ROWS rows, one simdgroup's work) and picks the dispatch's simdgroup count so
    /// that the LAST WAVE of every declared phase is nearly full: a phase runs
    /// `ceil(items / simdgroups)` waves and the grid barrier ending it waits for the fullest
    /// simdgroup, so items that do not divide the grid are paid for as a whole wave. Measured
    /// on qwen35: the down projection's 640 units over 288 simdgroups ran THREE waves for
    /// 2.22 units of work and cost +33% over the same GEMV on the dispatch path, while the
    /// gated pair's 2176 units (7.56 each) were within 2%. A program that declares nothing
    /// keeps its seated simdgroup count.
    pub rows: &'static str,
}
/// What the next phase needs to see. A GRID barrier (mega_step) makes every threadgroup's writes
/// visible to every other one. A threadgroup barrier is enough when each consumer reads only what
/// its own threadgroup wrote -- the entry says when that is so.
pub enum Barrier {
    None,
    Grid,
    /// Grid, except while this entry condition holds; then threadgroup-local.
    GridUnlessLocal(&'static str),
}
const fn ph(when: &'static str, call: &'static str, barrier: Barrier) -> MegaPhase {
    MegaPhase { when, call, fallible: false, barrier, rows: "" }
}
const fn ph_try(when: &'static str, call: &'static str, barrier: Barrier) -> MegaPhase {
    MegaPhase { when, call, fallible: true, barrier, rows: "" }
}
/// A phase that strides tile-major units, naming the entry word that holds its row count.
const fn ph_rows(
    when: &'static str,
    call: &'static str,
    barrier: Barrier,
    rows: &'static str,
) -> MegaPhase {
    MegaPhase { when, call, fallible: false, barrier, rows }
}

/// gemma4 (E4B): input norm + q/k/v rows, head rows, attention, o_proj and the sandwich norms,
/// the gated FFN, the PLE gate and projection, the tail norm (and the next layer's input norm).
pub const G4_PROGRAM: &[MegaPhase] = &[
    ph("ent.u[G4_U_QKV_PHASE] != 0u", "g4_ph_in_norm_qkv_rows", Barrier::Grid),
    ph("ent.u[G4_U_QKV_PHASE] != 0u", "g4_ph_head_rows", Barrier::Grid),
    ph_try("ent.u[G4_U_ATTN_PHASE] != 0u", "g4_ph_attn", Barrier::Grid),
    ph("ent.u[G4_U_ATTN_PHASE] != 0u", "g4_ph_attn_combine", Barrier::Grid),
    ph("ent.u[G4_U_FRONT] != 0u", "g4_ph_oproj", Barrier::Grid),
    ph("ent.u[G4_U_FRONT] != 0u", "g4_ph_sandwich", Barrier::None),
    ph("ent.u[G4_U_FRONT] == 0u", "g4_ph_ffn_input_copy", Barrier::None),
    ph("", "g4_ph_ffn_gated", Barrier::Grid),
    ph("", "g4_ph_ffn_down", Barrier::Grid),
    ph("", "g4_ph_ple_gate", Barrier::Grid),
    ph("", "g4_ph_ple_proj", Barrier::Grid),
    ph("", "g4_ph_tail", Barrier::None),
];

/// LFM2: the operator norm, then the layer's mixer (short convolution or attention), then the
/// shared tail (residual, FFN norm, the gated FFN, the residual again).
pub const L2_PROGRAM: &[MegaPhase] = &[
    ph("L2_IS_CONV || L2_IS_ATTN", "l2_ph_op_norm", Barrier::None),
    ph("L2_IS_CONV", "l2_ph_conv_in", Barrier::Grid),
    ph("L2_IS_CONV", "l2_ph_conv_step_out", Barrier::Grid),
    ph("L2_IS_CONV", "l2_ph_conv_shift", Barrier::None),
    ph("L2_IS_ATTN", "l2_ph_qkv_rows", Barrier::None),
    ph("L2_IS_ATTN", "l2_ph_head_rows", Barrier::Grid),
    // Head h's one partial came from threadgroup h itself when the heads are not split and the
    // plain body ran, so the combine below needs no grid barrier then.
    ph_try("L2_IS_ATTN", "l2_ph_attn",
           Barrier::GridUnlessLocal("ent.u[L2_U_ATTN_BODY] == 0u && ent.u[L2_U_ATTN_SPLIT] == 1u")),
    ph("L2_IS_ATTN", "l2_ph_attn_combine", Barrier::Grid),
    ph("L2_IS_ATTN", "l2_ph_oproj", Barrier::Grid),
    ph("", "l2_ph_ffn_norm", Barrier::None),
    ph("", "l2_ph_ffn_gated", Barrier::Grid),
    ph("", "l2_ph_ffn_down", Barrier::Grid),
    ph("", "l2_ph_tail", Barrier::None),
];

/// Qwen3.8 (task #165): the layer's mixer, then the shared SwiGLU tail. `kind` selects the
/// mixer and is an entry word, so every threadgroup of the dispatch takes the same branch.
/// The gate and up projections sit in ONE phase with no barrier between them: unit `un`
/// belongs to simdgroup `un % sg_total` in both passes, so each simdgroup reads back the
/// rows it just wrote. Two passes rather than one loop because the two matrices can carry
/// different weight formats, and the format switch belongs outside the row loop.
pub const Q35_PROGRAM: &[MegaPhase] = &[
    ph("", "q35_ph_ffn_norm", Barrier::None),
    ph_rows("", "q35_ph_ffn_gated", Barrier::Grid, "Q35_U_N_FF"),
    ph_rows("", "q35_ph_ffn_down", Barrier::Grid, "MEGA_U_N_EMBD"),
    ph("", "q35_ph_tail", Barrier::None),
];
