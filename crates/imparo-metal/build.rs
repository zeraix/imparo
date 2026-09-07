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
    // The mega entry's slot tables (task #158 step 2): one source, three views -- the shader's
    // constants at the marker, the bridge's enums, the Rust consts.
    println!("cargo:rerun-if-changed=mega_slots.rs");
    let slots = mega_slots::emit();
    assert!(
        msl.matches("// @@MEGA_SLOTS@@").count() == 1,
        "imparo.metal must carry exactly one @@MEGA_SLOTS@@ marker"
    );
    let msl = msl.replace("// @@MEGA_SLOTS@@", &slots.msl);
    // The kernels are emitted from the programs (task #158 step 3): the frame once, the
    // sequence per architecture. IMPARO_MEGA_DUMP names a file to write the emitted text to.
    println!("cargo:rerun-if-changed=mega_program.rs");
    assert!(
        msl.matches("// @@MEGA_KERNELS@@").count() == 1,
        "imparo.metal must carry exactly one @@MEGA_KERNELS@@ marker"
    );
    let kernels = mega_kernels::emit();
    std::fs::write(out.join("mega_kernels.metal"), &kernels)
        .expect("write mega_kernels.metal");
    let msl = msl.replace("// @@MEGA_KERNELS@@", &kernels);
    std::fs::write(out.join("mega_slots.h"), slots.cpp).expect("write mega_slots.h");
    std::fs::write(out.join("mega_slots_gen.rs"), slots.rust)
        .expect("write mega_slots_gen.rs");
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
    println!("cargo:rustc-link-lib=framework=IOKit");
}

mod mega_slots {
    include!("mega_slots.rs");

    pub struct Emitted {
        pub msl: String,
        pub cpp: String,
        pub rust: String,
    }
    /// The three views of the slot tables. Header names come first in every program's word and
    /// float tables (the same indices for every program); the program's own names follow.
    pub fn emit() -> Emitted {
        let words = |own: &[&str]| -> Vec<String> {
            MEGA_HDR_U
                .iter()
                .chain(own.iter())
                .map(|n| n.to_uppercase())
                .collect()
        };
        let floats = |own: &[&str]| -> Vec<String> {
            MEGA_HDR_F
                .iter()
                .chain(own.iter())
                .map(|n| n.to_uppercase())
                .collect()
        };
        let up = |xs: &[&str]| -> Vec<String> {
            xs.iter().map(|n| n.to_uppercase()).collect()
        };
        let programs: [(&str, Vec<String>, Vec<String>, Vec<String>, Vec<String>); 2] = [
            (
                "G4",
                up(MEGA_G4_PTR),
                up(MEGA_G4_OFF),
                words(MEGA_G4_U),
                floats(MEGA_G4_F),
            ),
            (
                "L2",
                up(MEGA_L2_PTR),
                up(MEGA_L2_OFF),
                words(MEGA_L2_U),
                floats(MEGA_L2_F),
            ),
        ];
        for (name, ptr, off, u, f) in &programs {
            assert!(
                ptr.len() <= MEGA_NP
                    && off.len() <= MEGA_NO
                    && u.len() <= MEGA_NU
                    && f.len() <= MEGA_NF,
                "{name}: a slot table exceeds its capacity"
            );
        }
        let hdr_u: Vec<String> = MEGA_HDR_U.iter().map(|n| n.to_uppercase()).collect();
        let hdr_f: Vec<String> = MEGA_HDR_F.iter().map(|n| n.to_uppercase()).collect();
        let list = |prefix: &str, names: &[String], sep: &str| -> String {
            names
                .iter()
                .enumerate()
                .map(|(i, n)| format!("{prefix}{n} = {i}u"))
                .collect::<Vec<_>>()
                .join(sep)
        };
        // MSL
        let mut msl = format!(
            "constant uint MEGA_NP = {MEGA_NP}u, MEGA_NO = {MEGA_NO}u, MEGA_NU = {MEGA_NU}u, MEGA_NF = {MEGA_NF}u;\n"
        );
        msl += &format!("constant uint {};\n", list("MEGA_U_", &hdr_u, ", "));
        msl += &format!("constant uint {};\n", list("MEGA_F_", &hdr_f, ", "));
        for (name, ptr, off, u, f) in &programs {
            msl +=
                &format!("constant uint {};\n", list(&format!("{name}_"), ptr, ", "));
            msl += &format!(
                "constant uint {};\n",
                list(&format!("{name}_O_"), off, ", ")
            );
            msl +=
                &format!("constant uint {};\n", list(&format!("{name}_U_"), u, ", "));
            msl +=
                &format!("constant uint {};\n", list(&format!("{name}_F_"), f, ", "));
        }
        // C++
        let mut cpp = String::from(
            "// Generated by build.rs from mega_slots.rs -- do not edit.\n#pragma once\n#include <cstdint>\n",
        );
        cpp += &format!(
            "constexpr uint32_t MEGA_NP = {MEGA_NP}u, MEGA_NO = {MEGA_NO}u, MEGA_NU = {MEGA_NU}u, MEGA_NF = {MEGA_NF}u;\n"
        );
        cpp += &format!("enum : uint32_t {{ {} }};\n", list("MEGA_U_", &hdr_u, ", "));
        cpp += &format!("enum : uint32_t {{ {} }};\n", list("MEGA_F_", &hdr_f, ", "));
        for (name, ptr, off, u, f) in &programs {
            cpp += &format!(
                "enum : uint32_t {{ {} }};\n",
                list(&format!("{name}_"), ptr, ", ")
            );
            cpp += &format!(
                "enum : uint32_t {{ {} }};\n",
                list(&format!("{name}_O_"), off, ", ")
            );
            cpp += &format!(
                "enum : uint32_t {{ {} }};\n",
                list(&format!("{name}_U_"), u, ", ")
            );
            cpp += &format!(
                "enum : uint32_t {{ {} }};\n",
                list(&format!("{name}_F_"), f, ", ")
            );
        }
        // Rust
        let rlist = |prefix: &str, names: &[String]| -> String {
            names
                .iter()
                .enumerate()
                .map(|(i, n)| format!("pub const {prefix}{n}: usize = {i};"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let mut rust = format!(
            "// Generated by build.rs from mega_slots.rs -- do not edit.\npub const MEGA_NP: usize = {MEGA_NP};\npub const MEGA_NO: usize = {MEGA_NO};\npub const MEGA_NU: usize = {MEGA_NU};\npub const MEGA_NF: usize = {MEGA_NF};\n"
        );
        rust += &rlist("MEGA_U_", &hdr_u);
        rust += "\n";
        rust += &rlist("MEGA_F_", &hdr_f);
        rust += "\n";
        for (name, ptr, off, u, f) in &programs {
            rust += &rlist(&format!("{name}_"), ptr);
            rust += "\n";
            rust += &rlist(&format!("{name}_O_"), off);
            rust += "\n";
            rust += &rlist(&format!("{name}_U_"), u);
            rust += "\n";
            rust += &rlist(&format!("{name}_F_"), f);
            rust += "\n";
        }
        Emitted { msl, cpp, rust }
    }
}

mod mega_kernels {
    include!("mega_program.rs");

    /// The kernel of one architecture: the frame (identical for every architecture, written
    /// here once) with the program's phase calls between its head and its tail.
    fn kernel(
        base: &str,
        pfx: &str,
        entry_check: &str,
        program: &[MegaPhase],
    ) -> String {
        let name = format!("{base}_t");
        let mut o = String::new();
        o += &format!(
            "// {name}: emitted by build.rs from mega_program.rs ({} phases). The frame is the\n// same for every architecture; only the entry check and the sequence below differ.\ntemplate <uint HD, uint KVW>\nkernel void {name}(\n",
            program.len()
        );
        o += "    constant MegaEntryG * prog   [[buffer(0)]],   // the layer's entry; the program form indexes it at runtime (task #153)\n    device atomic_uint * sync    [[buffer(1)]],   // word 4 the barrier counter, 13 exit, 15 error; the block's scratch from MEGA_SYNC_HDR\n    constant MegaToken & tok     [[buffer(2)]],\n    constant float * freqs       [[buffer(3)]],   // the model's rope factor table (a 1-float dummy when it has none)\n    threadgroup float4 * xn      [[threadgroup(0)]], // n_embd floats: the row every threadgroup forms\n    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n    uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]],\n    uint nsg  [[simdgroups_per_threadgroup]])\n{\n";
        o += "    threadgroup float partial[32];\n    const uint sub  = lane % LANES_PER_ROW;\n    const uint slot = lane / LANES_PER_ROW;\n    const uint sg_global = tgid * nsg + sgid;\n    const uint tcount    = nsg * 32u;\n    device atomic_uint * ctr  = sync + 4;\n    device atomic_uint * exitc = sync + 13;\n    device atomic_uint * err  = sync + 15;\n    threadgroup uint tg_err;\n    if (mega_enter(err, tok.entry_bytes, (uint)sizeof(MegaEntryG), &tg_err, tid)) { return; }\n    // The block's own scratch: attention partials live in the sync buffer, never in an activation\n    // buffer (the arena overlaps its groups, task #148), past MEGA_SYNC_HDR so no plain store\n    // shares a cache line with the barrier counters.\n    device float * apart = (device float *)(sync + MEGA_SYNC_HDR);\n    uint phase = 0u;\n    // The program form (task #153) walks tok.n_entries consecutive entries in this one dispatch,\n    // a grid barrier between them; the per-layer form is the loop at one entry, prog[0].\n    const uint n_entries = MEGA_PROGRAM ? max(tok.n_entries, 1u) : 1u;\n    for (uint ei = 0u; ei < n_entries; ++ei) {\n    constant MegaEntryG & ent = prog[MEGA_PROGRAM ? tok.entry_index + ei : 0u];\n    const uint dbg_seq = min(tok.dbg_seq + ei, 63u);\n";
        o += &format!(
            "    // An entry every threadgroup must read the same way: its n_tg is the dispatch's and its\n    // widths are non-zero (a zero row stride would loop forever). Otherwise report and leave.\n    if ({entry_check}) {{\n        mega_fail_entry(err, dbg_seq, tgid, tid);\n        return;\n    }}\n"
        );
        o += &format!(
            "    const uint sg_total = ent.u[{pfx}_U_N_TG] * nsg;\n    // Debug records (MEGA_DBG only): 32 work items x 8 words per (region slot, entry), after the scratch.\n    device atomic_uint * dbg_rec = sync + MEGA_SYNC_HDR + ent.u[{pfx}_U_SCRATCH] + ((ulong)(tok.dbg_slot * 64u + dbg_seq) * 32u) * MEGA_DBG_REC;\n    const uint w4 = ent.u[{pfx}_U_N_EMBD] / 4u;\n"
        );
        o += "    // THE SEQUENCE: which phases run, in what order, with which grid barriers between them.\n";
        let step = "mega_step(ctr, phase, tok.n_tg, err, tid, tgid);";
        for p in program {
            let body = if p.fallible {
                format!(
                    "if (!{}<HD, KVW>(MEGA_PH_CALL)) {{ mega_fail_entry(err, dbg_seq, tgid, tid); return; }}",
                    p.call
                )
            } else {
                format!("{}<HD, KVW>(MEGA_PH_CALL);", p.call)
            };
            let tail = match p.barrier {
                Barrier::None => String::new(),
                Barrier::Grid => format!(" {step}"),
                Barrier::GridUnlessLocal(c) => {
                    format!(
                        " if ({c}) {{ threadgroup_barrier(mem_flags::mem_device); }} else {{ {step} }}"
                    )
                }
            };
            if p.when.is_empty() {
                o += &format!("    {body}{tail}\n");
            } else {
                o += &format!("    if ({}) {{ {body}{tail} }}\n", p.when);
            }
        }
        o += &format!(
            "    // The program form: the next entry reads what threadgroup 0 just wrote.\n    if (MEGA_PROGRAM && ei + 1u < n_entries) {{ {step} }}\n    }}   // entries\n    mega_exit(ctr, exitc, tok.n_tg, tid);\n}}\n"
        );
        // One instantiation per head-dim slot the model declared: the attention phases' bodies are
        // compile-time in the head dim, the other phases do not depend on it.
        o += &format!(
            "#define IMPARO_MEGA_INST(SLOT, HDV, KVWV) \\\ntemplate [[host_name(\"{base}_s\" #SLOT)]] kernel void {name}<HDV, KVWV>( \\\n    constant MegaEntryG *, device atomic_uint *, constant MegaToken &, constant float *, \\\n    threadgroup float4 *, uint, uint, uint, uint, uint);\n#if IMPARO_HD0 && (IMPARO_HD0 % 32 == 0)\nIMPARO_MEGA_INST(0, IMPARO_HD0, IMPARO_KVW0)\n#endif\n#if IMPARO_HD1 && (IMPARO_HD1 % 32 == 0)\nIMPARO_MEGA_INST(1, IMPARO_HD1, IMPARO_KVW1)\n#endif\n#undef IMPARO_MEGA_INST\n"
        );
        o
    }
    pub fn emit() -> String {
        let mut o = String::from(
            "// Generated by build.rs from mega_program.rs -- do not edit here; edit the program.\n",
        );
        o += &kernel(
            "imparo_mega_layer",
            "G4",
            "ent.u[G4_U_N_TG] != tok.n_tg || ent.u[G4_U_N_EMBD] == 0u || ent.u[G4_U_N_MID] == 0u",
            G4_PROGRAM,
        );
        o += "\n";
        o += &kernel(
            "imparo_mega_lfm2_layer",
            "L2",
            "ent.u[L2_U_N_TG] != tok.n_tg || ent.u[L2_U_N_EMBD] == 0u || ent.u[L2_U_N_FF] == 0u\n        || ent.u[L2_U_MIXER] > MEGA_LFM2_MIXER_ATTENTION\n        || (L2_IS_CONV && (ent.u[L2_U_KERN] == 0u || ent.u[L2_U_KERN] - 1u > SHORTCONV_MAX_HISTORY))\n        || (L2_IS_ATTN && (ent.u[L2_U_N_HEADS] == 0u || ent.u[L2_U_N_KV] == 0u || ent.u[L2_U_N_HEADS] % ent.u[L2_U_N_KV] != 0u\n                           || ent.u[L2_U_ATTN_SPLIT] == 0u || ent.u[L2_U_N_ROT] == 0u || nsg % (HD / Q8_TM_UNIT_ROWS) != 0u))",
            L2_PROGRAM,
        );
        o
    }
}
