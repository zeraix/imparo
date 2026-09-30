//! imparo-repack: write a GGUF whose 2-D projection weights are in the tile-major layout
//! the engine's GEMM reads with aligned, coalesced loads. Which kinds convert and where
//! every byte goes is `imparo_gguf::weights::TM_RULES` (Q8_0 -> Q8_0_TM, Q4_0 -> Q4_0_TM);
//! the design and the reason are in docs/q8-tile-major-weights.md. The same rules drive
//! the engine's load-time transform of an original file's fast tier, so a converted file
//! and a transformed-at-load original hold identical bytes.
//!
//! The converted file is a byte-for-byte copy of the input except for the converted
//! tensors' data (rewritten in place: same size, same offset) and their type id in the
//! tensor-info table. Nothing else in the container moves, so the engine mmaps the result
//! zero-copy exactly as it mmaps the original.
//!
//!     imparo-repack <in.gguf | org/repo[:selector]> <out.gguf> [--force] [--only KIND]
//!                   [--write-unread] [--original PATH] [--endpoint URL] [--mirror URL]
//!
//! The source is a local file or a Hugging Face model id. For an id the tool lists the
//! repo's files, picks the one .gguf (or the one whose name contains `:selector`), and
//! converts WHILE downloading: bytes stream into the output file, and each projection is
//! rewritten tile-major the moment its last byte has arrived, so the conversion costs no
//! wall-clock of its own. `HF_ENDPOINT` (or --endpoint) names the hub; a connection
//! failure or timeout on it falls back to the mirror (default https://hf-mirror.com). An
//! interrupted download resumes from `<out>.download` on the next run. `--original PATH`
//! also keeps the untouched download. `--only Q8_0` converts one row-major kind. A rule
//! whose tile-major kind no backend reads yet (`TmRule::readers` empty) is skipped unless
//! `--write-unread`: the engine refuses such a file at load rather than misread it.
//!
//! Every converted tensor is verified before the tool reports success: each value byte and
//! each scale is read back at the address the rule gives and compared with the row-major
//! block. A mismatch is an error, not a warning.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use imparo_gguf::weights::{Mapping, TmRule, tm_applies};

fn bytes_of(map: &Mapping, off: usize, len: usize) -> &[u8] {
    assert!(off + len <= map.len(), "tensor span outside the mapping");
    // SAFETY: the mapping is read-only, immutable and at least off+len bytes long.
    unsafe { std::slice::from_raw_parts(map.base().add(off), len) }
}

/// What the user asked for, from the command line.
struct Opts {
    force: bool,
    write_unread: bool,
    only: Option<String>,
    original: Option<PathBuf>,
    endpoint: String,
    mirror: String,
}

fn usage() -> ! {
    eprintln!(
        "usage: imparo-repack <in.gguf | org/repo[:selector]> <out.gguf> [--force] [--only KIND] \
         [--write-unread] [--original PATH] [--endpoint URL] [--mirror URL]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = Opts {
        force: false,
        write_unread: false,
        only: None,
        original: None,
        endpoint: std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| "https://huggingface.co".to_string()),
        mirror: "https://hf-mirror.com".to_string(),
    };
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let value = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| usage())
        };
        match a.as_str() {
            "--force" => opts.force = true,
            "--write-unread" => opts.write_unread = true,
            "--only" => opts.only = Some(value(&mut i)),
            "--original" => opts.original = Some(PathBuf::from(value(&mut i))),
            "--endpoint" => opts.endpoint = value(&mut i),
            "--mirror" => opts.mirror = value(&mut i),
            _ if a.starts_with("--") => usage(),
            _ => positional.push(a.clone()),
        }
        i += 1;
    }
    if positional.len() != 2 {
        usage();
    }
    let dst_path = Path::new(&positional[1]);
    if dst_path.exists() && !opts.force && !Path::new(&sidecar_path(dst_path)).exists()
    {
        return Err(format!(
            "{} exists; pass --force to overwrite",
            dst_path.display()
        )
        .into());
    }
    let src = &positional[0];
    if Path::new(src).is_file() {
        convert_file(Path::new(src), dst_path, &opts)
    } else if looks_like_hf_id(src) {
        download_and_convert(src, dst_path, &opts)
    } else {
        Err(format!("{src}: not a file, and not a Hugging Face id of the form org/repo[:selector]").into())
    }
}

fn looks_like_hf_id(s: &str) -> bool {
    let id = s.split(':').next().unwrap_or("");
    let parts: Vec<&str> = id.split('/').collect();
    parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && !p.contains(char::is_whitespace))
}

/// The tensors a document converts under these options, and the ones kept with the reason.
fn plan_tensors(
    doc: &imparo_gguf::Document,
    opts: &Opts,
) -> (
    Vec<(imparo_gguf::TensorInfo, &'static TmRule)>,
    Vec<(String, &'static str, &'static str)>,
) {
    let mut plan = Vec::new();
    let mut kept = Vec::new();
    for t in &doc.tensors {
        match tm_applies(&t.name, t.ggml_type, &t.dimensions) {
            Ok(rule) => {
                if opts.only.as_deref().is_some_and(|k| k != rule.from_name) {
                    kept.push((t.name.clone(), rule.from_name, "kept by --only"));
                } else if !rule.has_readers_for(&t.dimensions) && !opts.write_unread {
                    kept.push((t.name.clone(), rule.from_name, "no backend reads the tile-major kind yet (--write-unread to write it)"));
                } else {
                    plan.push((t.clone(), rule));
                }
            }
            Err(why) => {
                if let Some(rule) = imparo_gguf::weights::tm_rule_for(t.ggml_type) {
                    kept.push((t.name.clone(), rule.from_name, why));
                }
            }
        }
    }
    (plan, kept)
}

fn convert_file(
    src_path: &Path,
    dst_path: &Path,
    opts: &Opts,
) -> Result<(), Box<dyn std::error::Error>> {
    let doc = imparo_gguf::read(src_path)?;
    let src = Mapping::open(src_path)?;
    let (plan, kept) = plan_tensors(&doc, opts);

    if plan.is_empty() {
        return Err("no 2-D projection tensor with a tile-major rule to convert; nothing written".into());
    }

    std::fs::copy(src_path, dst_path)?;
    let mut out = std::fs::OpenOptions::new().write(true).open(dst_path)?;
    let mut total = 0_usize;
    for (t, rule) in &plan {
        let (n_in, n_out) = (t.dimensions[0] as usize, t.dimensions[1] as usize);
        let bytes = bytes_of(&src, t.absolute_offset as usize, t.byte_size as usize);
        let tm = rule.convert(bytes, n_in, n_out);
        rule.verify(bytes, &tm, n_in, n_out, &t.name)?;
        out.seek(SeekFrom::Start(t.absolute_offset))?;
        out.write_all(&tm)?;
        out.seek(SeekFrom::Start(t.type_field_offset))?;
        out.write_all(&rule.to.to_le_bytes())?;
        total += tm.len();
        println!(
            "converted {:<40} {} -> {} [{n_in} x {n_out}]  {:>10} bytes",
            t.name,
            rule.from_name,
            rule.to_name,
            tm.len()
        );
    }
    out.flush()?;
    drop(out);

    // Read the OUTPUT back as a document: the header must parse, the converted tensors
    // must carry the new type at the same offset and size, and their bytes must verify
    // against the input a second time -- from the file on disk, not from the buffer.
    let out_doc = imparo_gguf::read(dst_path)?;
    let out_map = Mapping::open(dst_path)?;
    for (t, rule) in &plan {
        let o = out_doc
            .tensors
            .iter()
            .find(|o| o.name == t.name)
            .ok_or_else(|| format!("{}: missing from the output", t.name))?;
        if o.ggml_type != rule.to
            || o.absolute_offset != t.absolute_offset
            || o.byte_size != t.byte_size
        {
            return Err(
                format!("{}: output header type/offset/size differ", t.name).into()
            );
        }
        let (n_in, n_out) = (t.dimensions[0] as usize, t.dimensions[1] as usize);
        rule.verify(
            bytes_of(&src, t.absolute_offset as usize, t.byte_size as usize),
            bytes_of(&out_map, o.absolute_offset as usize, o.byte_size as usize),
            n_in,
            n_out,
            &t.name,
        )?;
    }
    for (name, kind, why) in &kept {
        println!("kept      {name:<40} {kind} row-major: {why}");
    }
    println!(
        "ALL {} tensors converted ({} MiB), {} tensors kept, output verified byte-for-byte: {}",
        plan.len(),
        total >> 20,
        kept.len(),
        dst_path.display()
    );
    Ok(())
}

// ---- Hugging Face download, converting while the bytes arrive ---------------------------

fn sidecar_path(dst: &Path) -> PathBuf {
    let mut p = dst.as_os_str().to_owned();
    p.push(".download");
    PathBuf::from(p)
}

/// One curl invocation. The tool shells out to curl rather than linking an HTTP stack: it
/// is present on macOS, Linux and Windows 10+, speaks HTTP/2 and TLS, follows the hub's
/// redirects to its CDN, and resumes with a Range request. Exit codes 6 / 7 / 28 (cannot
/// resolve, cannot connect, timeout) are the connection failures that trigger the mirror.
fn curl_connect_failed(code: Option<i32>) -> bool {
    matches!(code, Some(6 | 7 | 28 | 35))
}

fn curl_text(url: &str) -> Result<String, (Option<i32>, String)> {
    let out = Command::new("curl")
        .args([
            "-sS",
            "-L",
            "--fail",
            "--connect-timeout",
            "15",
            "--max-time",
            "120",
            url,
        ])
        .output()
        .map_err(|e| (None, format!("curl: {e}")))?;
    if !out.status.success() {
        return Err((
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Try the primary endpoint, then the mirror on a connection failure; other failures
/// (404, 401) are reported from the primary as they are.
fn with_fallback<T>(
    opts: &Opts,
    what: &str,
    f: impl Fn(&str) -> Result<T, (Option<i32>, String)>,
) -> Result<(T, String), Box<dyn std::error::Error>> {
    match f(&opts.endpoint) {
        Ok(v) => Ok((v, opts.endpoint.clone())),
        Err((code, msg))
            if curl_connect_failed(code) && opts.mirror != opts.endpoint =>
        {
            eprintln!(
                "{what}: {} unreachable ({msg}); trying the mirror {}",
                opts.endpoint, opts.mirror
            );
            match f(&opts.mirror) {
                Ok(v) => Ok((v, opts.mirror.clone())),
                Err((_, msg2)) => {
                    Err(format!("{what}: {} failed too: {msg2}", opts.mirror).into())
                }
            }
        }
        Err((_, msg)) => Err(format!("{what} at {}: {msg}", opts.endpoint).into()),
    }
}

/// `"rfilename":"..."` values of the model API's JSON, without a JSON dependency: the
/// field is a plain string on the hub (no escapes in file names it serves).
fn rfilenames(json: &str) -> Vec<String> {
    let key = "\"rfilename\":\"";
    let mut out = Vec::new();
    let mut rest = json;
    while let Some(i) = rest.find(key) {
        rest = &rest[i + key.len()..];
        if let Some(j) = rest.find('"') {
            out.push(rest[..j].to_string());
            rest = &rest[j..];
        }
    }
    out
}

fn download_and_convert(
    id: &str,
    dst_path: &Path,
    opts: &Opts,
) -> Result<(), Box<dyn std::error::Error>> {
    let (repo, selector) = match id.split_once(':') {
        Some((r, sel)) => (r, Some(sel)),
        None => (id, None),
    };
    // 1. Which file. One .gguf, or the one the selector names.
    let (listing, endpoint) = with_fallback(opts, "listing the repo", |ep| {
        curl_text(&format!("{ep}/api/models/{repo}"))
    })?;
    let ggufs: Vec<String> = rfilenames(&listing)
        .into_iter()
        .filter(|f| {
            std::path::Path::new(f)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    let file = match selector {
        Some(sel) => {
            let hits: Vec<&String> = ggufs.iter().filter(|f| f.contains(sel)).collect();
            match hits.len() {
                1 => hits[0].clone(),
                0 => {
                    return Err(format!(
                        "{repo}: no .gguf matches '{sel}'; files: {}",
                        ggufs.join(", ")
                    )
                    .into());
                }
                _ => {
                    return Err(format!(
                        "{repo}: '{sel}' matches several: {}",
                        hits.iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                    .into());
                }
            }
        }
        None => match ggufs.len() {
            1 => ggufs[0].clone(),
            0 => return Err(format!("{repo}: no .gguf in the repo").into()),
            _ => {
                return Err(format!(
                    "{repo}: several .gguf files, add :selector -- {}",
                    ggufs.join(", ")
                )
                .into());
            }
        },
    };
    let url_of = |ep: &str| format!("{ep}/{repo}/resolve/main/{file}");
    // 2. Size, from a HEAD (the hub answers with the CDN redirect's Content-Length).
    let (head, endpoint) = with_fallback(
        &Opts {
            endpoint: endpoint.clone(),
            ..clone_opts(opts)
        },
        "reading the file size",
        |ep| {
            let out = Command::new("curl")
                .args([
                    "-sSIL",
                    "--fail",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "60",
                    &url_of(ep),
                ])
                .output()
                .map_err(|e| (None, format!("curl: {e}")))?;
            if !out.status.success() {
                return Err((
                    out.status.code(),
                    String::from_utf8_lossy(&out.stderr).trim().to_string(),
                ));
            }
            Ok(String::from_utf8_lossy(&out.stdout).to_string())
        },
    )?;
    let total: u64 = head
        .lines()
        .rfind(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .ok_or("no Content-Length in the HEAD response")?;
    println!("{repo}: {file} ({} MiB) from {endpoint}", total >> 20);

    // 3. Resume state: `<out>.download` holds the bytes received and the converted tensors.
    let side = sidecar_path(dst_path);
    let mut received: u64 = 0;
    let mut converted: Vec<String> = Vec::new();
    if side.exists() && dst_path.exists() {
        for line in std::fs::read_to_string(&side)?.lines() {
            if let Some(v) = line.strip_prefix("received=") {
                received = v.trim().parse().unwrap_or(0);
            }
            if let Some(v) = line.strip_prefix("total=") {
                let t: u64 = v.trim().parse().unwrap_or(0);
                if t != total {
                    received = 0;
                    converted.clear();
                }
            }
            if let Some(v) = line.strip_prefix("converted=") {
                converted = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
            }
        }
        let on_disk = std::fs::metadata(dst_path)?.len();
        if on_disk < received {
            received = 0;
            converted.clear();
        }
        if received > 0 {
            println!(
                "resuming at {} MiB ({} tensors already converted)",
                received >> 20,
                converted.len()
            );
        }
    } else {
        received = 0;
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .truncate(received == 0)
        .open(dst_path)?;
    out.set_len(received)?;
    let mut orig = match &opts.original {
        Some(p) => {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(received == 0)
                .open(p)?;
            f.set_len(received)?;
            Some(f)
        }
        None => None,
    };
    let write_side = |received: u64, converted: &[String]| -> std::io::Result<()> {
        std::fs::write(
            &side,
            format!(
                "total={total}\nreceived={received}\nconverted={}\n",
                converted.join(",")
            ),
        )
    };

    // 4. Stream. Each chunk lands in the file; when the header has arrived the tensor table
    //    is known; a tensor whose last byte has arrived is converted in place at once.
    let mut doc: Option<imparo_gguf::Document> = None;
    let mut plan: Vec<(imparo_gguf::TensorInfo, &'static TmRule)> = Vec::new();
    let mut kept: Vec<(String, &'static str, &'static str)> = Vec::new();
    let mut ep = endpoint.clone();
    let mut attempts = 0;
    let mut next_report = received + (256u64 << 20);
    while received < total {
        attempts += 1;
        if attempts > 12 {
            return Err("download: too many retries".into());
        }
        let url = url_of(&ep);
        let mut child = Command::new("curl")
            .args([
                "-sS",
                "-L",
                "--fail",
                "--connect-timeout",
                "15",
                "--speed-time",
                "60",
                "--speed-limit",
                "1024",
                "-C",
                &received.to_string(),
                "-o",
                "-",
                &url,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("piped");
        let mut buf = vec![0_u8; 8 << 20];
        loop {
            let n = match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    eprintln!("read: {e}");
                    break;
                }
            };
            let chunk = &buf[..n];
            out.seek(SeekFrom::Start(received))?;
            out.write_all(chunk)?;
            if let Some(o) = orig.as_mut() {
                o.seek(SeekFrom::Start(received))?;
                o.write_all(chunk)?;
            }
            received += n as u64;
            if doc.is_none() && received >= (1u64 << 20).min(total) {
                // The header parses as soon as its bytes are here; before that the parser
                // fails on a short read, which is not an error yet.
                out.flush()?;
                let f = std::fs::File::open(dst_path)?;
                if let Ok(d) = imparo_gguf::read_from(std::io::BufReader::new(f), total)
                {
                    let (p, k) = plan_tensors(&d, opts);
                    println!(
                        "header: {} tensors, {} to convert, data at {}",
                        d.tensors.len(),
                        p.len(),
                        d.data_offset
                    );
                    plan = p;
                    kept = k;
                    doc = Some(d);
                }
            }
            if doc.is_some() {
                convert_arrived(&mut out, &plan, received, &mut converted)?;
            }
            if received >= next_report {
                println!(
                    "  {} / {} MiB, {} tensors converted",
                    received >> 20,
                    total >> 20,
                    converted.len()
                );
                next_report += 256u64 << 20;
            }
            write_side(received, &converted)?;
        }
        let status = child.wait()?;
        if received >= total {
            break;
        }
        let err = {
            let mut e = String::new();
            if let Some(mut s) = child.stderr.take() {
                let _ = s.read_to_string(&mut e);
            }
            e.trim().to_string()
        };
        eprintln!(
            "download interrupted at {} MiB ({}, {err}); resuming",
            received >> 20,
            status
        );
        if curl_connect_failed(status.code()) && ep != opts.mirror {
            eprintln!("switching to the mirror {}", opts.mirror);
            ep.clone_from(&opts.mirror);
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    if doc.is_none() {
        return Err("the file ended before its header parsed; not a GGUF?".into());
    }
    out.flush()?;
    drop(out);
    if let Some(o) = orig {
        drop(o);
    }

    // 5. Read the output back as a document and check every planned tensor carries the
    //    tile-major type; the bytes were verified as each tensor was converted.
    let out_doc = imparo_gguf::read(dst_path)?;
    for (t, rule) in &plan {
        let o = out_doc
            .tensors
            .iter()
            .find(|o| o.name == t.name)
            .ok_or_else(|| format!("{}: missing from the output", t.name))?;
        if o.ggml_type != rule.to
            || o.absolute_offset != t.absolute_offset
            || o.byte_size != t.byte_size
        {
            return Err(
                format!("{}: output header type/offset/size differ", t.name).into()
            );
        }
        if !converted.contains(&t.name) {
            return Err(format!(
                "{}: never converted (download state inconsistent)",
                t.name
            )
            .into());
        }
    }
    let _ = std::fs::remove_file(&side);
    for (name, kind, why) in &kept {
        println!("kept      {name:<40} {kind} row-major: {why}");
    }
    println!(
        "ALL {} tensors converted while downloading ({} MiB), {} tensors kept, verified byte-for-byte: {}{}",
        plan.len(),
        total >> 20,
        kept.len(),
        dst_path.display(),
        opts.original
            .as_ref()
            .map(|p| format!(" (original kept at {})", p.display()))
            .unwrap_or_default()
    );
    Ok(())
}

fn clone_opts(o: &Opts) -> Opts {
    Opts {
        force: o.force,
        write_unread: o.write_unread,
        only: o.only.clone(),
        original: o.original.clone(),
        endpoint: o.endpoint.clone(),
        mirror: o.mirror.clone(),
    }
}

/// Convert every planned tensor whose bytes have fully arrived and is not converted yet:
/// read its row-major bytes from the file, convert, verify, write back in place, patch the
/// type field, read the span back and compare.
fn convert_arrived(
    out: &mut std::fs::File,
    plan: &[(imparo_gguf::TensorInfo, &'static TmRule)],
    received: u64,
    converted: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    for (t, rule) in plan {
        if converted.contains(&t.name) || t.absolute_offset + t.byte_size > received {
            continue;
        }
        let (n_in, n_out) = (t.dimensions[0] as usize, t.dimensions[1] as usize);
        let mut src = vec![0_u8; t.byte_size as usize];
        out.seek(SeekFrom::Start(t.absolute_offset))?;
        out.read_exact(&mut src)?;
        let tm = rule.convert(&src, n_in, n_out);
        rule.verify(&src, &tm, n_in, n_out, &t.name)?;
        out.seek(SeekFrom::Start(t.absolute_offset))?;
        out.write_all(&tm)?;
        out.seek(SeekFrom::Start(t.type_field_offset))?;
        out.write_all(&rule.to.to_le_bytes())?;
        out.flush()?;
        let mut back = vec![0_u8; tm.len()];
        out.seek(SeekFrom::Start(t.absolute_offset))?;
        out.read_exact(&mut back)?;
        if back != tm {
            return Err(format!("{}: bytes read back from disk differ", t.name).into());
        }
        converted.push(t.name.clone());
        println!(
            "converted {:<40} {} -> {} [{n_in} x {n_out}]  {:>10} bytes",
            t.name,
            rule.from_name,
            rule.to_name,
            tm.len()
        );
    }
    Ok(())
}
