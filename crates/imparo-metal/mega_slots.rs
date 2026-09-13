// THE MEGA ENTRY'S SLOT TABLES (task #158 step 2): the one source for the shader's slot
// constants (imparo.metal, at the @@MEGA_SLOTS@@ marker), the host bridge's enums
// (mega_slots.h) and the Rust consts (mega_slots_gen.rs). build.rs emits all three, so the
// three sides cannot drift; a wrong count still fails the kernel's entry-size check loudly.
//
// An entry is a table of GPU pointers, a table of weight offsets (local to the pointer's
// segment), a table of words and a table of floats. Words 0..HDR_U and floats 0..HDR_F are
// the HEADER every program fills the same way: the host bridge reads the attention geometry
// from it and writes the values it derives (the dispatch's threadgroup count, the scratch
// floats, the norm thread counts, the attention split / body / heads per item) back into it.
// The words after the header are the program's own.
pub const MEGA_NP: usize = 40;
pub const MEGA_NO: usize = 20;
pub const MEGA_NU: usize = 48;
pub const MEGA_NF: usize = 8;
pub const MEGA_HDR_U: &[&str] = &[
    "n_tg", "n_embd", "scratch", "norm_t", "n_heads", "n_kv", "kv_width", "window", "ring",
    "attn_split", "attn_body", "attn_hq", "had_k", "had_v", "n_rot", "norm_t_head",
    // FLOATS OF THREADGROUP MEMORY THIS ENTRY'S PHASES NEED -- and ZERO MEANS ZERO, not a
    // default: a phase that reads its activation from device memory needs no row at all.
    // This is not n_embd by rule:
    // a phase that reuses a row across its simdgroups wants the row there, and a phase that
    // touches each element once wants device memory and no row at all. The architecture
    // knows which of its phases do what, so the entry says the number and the bridge sizes
    // the dispatch, the admission and the refusal from it. It also stops the row's width
    // being a hard limit on the model's width (task #175).
    "tgmem_f",
];
pub const MEGA_HDR_F: &[&str] = &["eps", "rope_base"];
// gemma4 (E4B): Q4_0 rows, the PLE tail, the sandwich norms, an optional next-layer norm.
pub const MEGA_G4_PTR: &[&str] = &[
    "w_gate", "w_up", "w_down", "w_pg", "w_pp", "w_n1", "w_n2", "w_n3", "w_npa", "w_nf", "w_qn", "w_kn", "w_in",
    "w_o", "w_q", "w_k", "w_v", "kc_w", "vc_w", "x", "o", "cur", "g", "u", "gate", "per_layer", "back", "nxt",
    "attn", "attn_out", "kc", "vc", "kv_pt", "qin", "kbuf", "vbuf",
];
pub const MEGA_G4_OFF: &[&str] = &[
    "gate_off", "up_off", "down_off", "pg_off", "pp_off", "n1_off", "n2_off", "n3_off", "wo_off", "npa_off",
    "nf_off", "wq_off", "wk_off", "wv_off", "qn_off", "kn_off", "in_off",
];
pub const MEGA_G4_U: &[&str] = &[
    "n_mid", "ple", "pl_off", "has_next", "attn_in", "front", "attn_phase", "n_freqs", "has_kv", "qkv_phase", "q_rows",
];
pub const MEGA_G4_F: &[&str] = &["out_scale"];
// LFM2: Q8_0 tile-major units, a short-conv or attention mixer, a plain gated FFN tail.
// `state` and `state_out` are the recurrent PLANES: the conv reads one and its shift writes
// the next, so the plane it read survives as the step's rollback point (task #165). The host
// passes the same offset for both where the advance is in place (prefill). They are two slots
// because `shortconv_channel_shift` carries values FORWARD from the plane it reads -- handing
// it the out plane as the source reads a stale row.
pub const MEGA_L2_PTR: &[&str] = &[
    "w_gate", "w_up", "w_down", "w_fn", "w_op", "w_in", "w_conv", "w_out", "w_q", "w_k", "w_v", "w_o", "w_qn", "w_kn",
    "x", "o", "g", "u", "bcx", "state", "state_out", "snap", "qin", "kbuf", "vbuf", "attn", "kc_w", "vc_w", "kc", "vc",
    "kv_pt",
];
pub const MEGA_L2_OFF: &[&str] = &[
    "gate_off", "up_off", "down_off", "fn_off", "op_off", "in_off", "conv_off", "out_off", "wq_off", "wk_off",
    "wv_off", "wo_off", "qn_off", "kn_off",
];
pub const MEGA_L2_U: &[&str] = &["n_ff", "mixer", "kern", "has_snap"];
pub const MEGA_L2_F: &[&str] = &["q_scale"];

// Qwen3.8: 48 gated delta-net mixers and 16 full attention blocks over one SwiGLU tail.
// TWO THINGS THIS ARCHITECTURE NEEDS THAT THE OTHER TWO DID NOT:
//  - a WEIGHT FORMAT PER TENSOR. The file assigns quants by importance (one UD file: 53
//    distinct per-block signatures over 65 blocks), so the format is a WORD here, not a
//    function constant, and the row brick switches on it once per phase.
//  - a RECURRENT STATE addressed by PLANE INDEX. 149.6 MiB per conversation, read and
//    written once per token; the plane is where the state IS, never a gathered copy.
// The two arms share every slot they can: one `z` is the delta gate and the attention
// output gate, one `attn` is the delta core and the attention output. What is NOT here:
// alpha and beta, which the delta phase forms and consumes inside one dispatch, so they
// live in the block's own scratch (task #148) rather than costing two pointer slots.
pub const MEGA_Q35_PTR: &[&str] = &[
    "x", "o", "g", "u", "z", "cur", "w_an", "w_fn", "w_gate", "w_up", "w_down",
    // full attention: the packed [query | gate] projection, k, v, o, the head norms
    "w_q", "w_k", "w_v", "w_o", "w_qn", "w_kn",
    "qin", "kbuf", "vbuf", "attn", "kc", "vc", "kv_pt", "kc_w", "vc_w",
    // gated delta net: the packed [Q | K | V] projection, the z gate, alpha, beta, out
    "w_dqkv", "w_dgate", "w_alpha", "w_beta", "w_dout",
    "w_conv", "w_a", "w_dt", "w_ssmn",
    "mix", "conv", "state", "snap",
];
pub const MEGA_Q35_OFF: &[&str] = &[
    "an_off", "fn_off", "gate_off", "up_off", "down_off",
    "wq_off", "wk_off", "wv_off", "wo_off", "qn_off", "kn_off",
    "dqkv_off", "dgate_off", "dalpha_off", "dbeta_off", "dout_off",
    "conv_off", "a_off", "dt_off", "ssmn_off",
];
// One word per tensor the phases read, because the format is per tensor. `kind` says which
// arm runs; `plane` and `snap_plane` say WHICH copy of the recurrent state the delta phase
// reads and writes. The POSITION each plane describes is the host's bookkeeping, not the
// kernel's: the kernel writes the plane it is told, and the host refuses a rewind onto a
// plane whose position is not the tip.
pub const MEGA_Q35_U: &[&str] = &[
    "n_ff", "kind", "q_width", "qkv_width", "taps", "k_heads", "v_heads", "key_dim", "value_dim",
    "state_off", "conv_state_off", "plane", "snap_plane", "has_snap",
    "f_gate", "f_up", "f_down", "f_q", "f_k", "f_v", "f_o",
    "f_dqkv", "f_dgate", "f_alpha", "f_beta", "f_dout",
];
pub const MEGA_Q35_F: &[&str] = &["q_scale"];
