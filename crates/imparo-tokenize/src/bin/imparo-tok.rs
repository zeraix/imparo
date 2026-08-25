//! Dev tool: encode/decode with the GGUF tokenizer, for comparison against llama.cpp.

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path =
        PathBuf::from(args.next().ok_or("usage: imparo-tok MODEL.gguf TEXT...")?);
    let rest: Vec<String> = args.collect();
    // --lines FILE: one JSON-escaped string per line. Avoids shell escaping changing
    // the input, which made a newline case look like a tokenizer difference.
    let json_mode = rest.first().map(String::as_str) == Some("--lines");
    let text: Vec<String> = if json_mode {
        let f = rest.get(1).ok_or("--lines needs a file")?;
        std::fs::read_to_string(f)?
            .lines()
            // NO empty filter: dropping the empty-string case shifted every later
            // comparison and made correct output look like a tokenizer difference.
            .map(unescape)
            .collect()
    } else {
        rest
    };
    let t0 = std::time::Instant::now();
    let tok = imparo_tokenize::Tokenizer::from_gguf(&path)?;
    println!(
        "load  ms={:.1} vocab={} bos={:?} eos={:?} eot={:?} add_bos={} add_space_prefix={}",
        t0.elapsed().as_secs_f64() * 1e3,
        tok.vocab_size(),
        tok.bos,
        tok.eos,
        tok.eot,
        tok.add_bos,
        tok.add_space_prefix
    );
    if json_mode {
        for t in &text {
            let ids = tok.encode(t, true);
            let list: Vec<String> = ids.iter().map(u32::to_string).collect();
            println!("[{}]", list.join(","));
        }
        return Ok(());
    }
    for t in &text {
        let ids = tok.encode(t, true);
        let back = tok.decode(&ids);
        println!("encode {t:?}\n  ids={ids:?}\n  decode={back:?}");
    }
    Ok(())
}

/// Minimal unescaping for the line format: \\n, \\t, \\\\ only.
fn unescape(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') | None => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    out
}
