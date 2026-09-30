//! Minimal HTTP/1.1 request reading and response writing.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    /// Lower-cased header names to values. Carries `x-conversation-id`, which the KV pool
    /// design uses purely for addressing -- it never gates correctness.
    pub headers: Vec<(String, String)>,
}

impl Request {
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Reads one request. Returns Ok(None) on a closed or empty connection.
///
/// # Errors
///
/// Returns an I/O error, or an error when the header or body exceeds its bound.
pub fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let peer = stream.try_clone()?;
    let mut reader = BufReader::new(peer);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0_usize;
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut header_bytes = line.len();
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h)?;
        if n == 0 {
            break;
        }
        header_bytes += n;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(std::io::Error::other("headers exceed 64 KiB"));
        }
        let t = h.trim_end();
        if t.is_empty() {
            break;
        }
        if let Some((k, v)) = t.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let value = v.trim().to_string();
            if key == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((key, value));
        }
    }
    if content_length > MAX_BODY_BYTES {
        return Err(std::io::Error::other("body exceeds bound"));
    }
    let mut body = vec![0_u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some(Request {
        method,
        path,
        body,
        headers,
    }))
}

/// # Errors
///
/// Returns an I/O error when the socket cannot be written.
pub fn json<W: Write + ?Sized>(
    stream: &mut W,
    status: u16,
    value: &serde_json::Value,
) -> std::io::Result<()> {
    let body = value.to_string();
    write!(
        stream,
        "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()
}

/// # Errors
///
/// Returns an I/O error when the socket cannot be written.
pub fn sse_headers<W: Write + ?Sized>(stream: &mut W) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    stream.flush()
}

/// One SSE frame, `data: <json>\n\n`, appended to `buf`; nothing reaches the socket until
/// `sse_send`. A token's frames (a reasoning delta and a visible delta, or the tail's four)
/// therefore go out as ONE write, and the frame itself is serialised into memory first
/// (task #202). It used to be `write!(stream, "data: {value}\n\n")` on the bare `TcpStream`:
/// a `write!` hands every piece the formatter produces to `write_all`, and `Value`'s
/// `Display` is the serializer writing piece by piece -- each brace, quote, key and colon
/// became its own write(2). Measured ~40 us per token for a 200-byte frame, flat in the
/// response length; a socket write is a few microseconds.
///
/// # Errors
///
/// Returns an I/O error when the value cannot be serialised (it always can for the JSON
/// this server builds; the type is what `serde_json::to_writer` returns).
pub fn sse_frame(buf: &mut Vec<u8>, value: &serde_json::Value) -> std::io::Result<()> {
    buf.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut *buf, value).map_err(std::io::Error::other)?;
    buf.extend_from_slice(b"\n\n");
    Ok(())
}

/// Write everything accumulated by `sse_frame` in one call and empty the buffer.
///
/// # Errors
///
/// Returns an I/O error when the socket cannot be written.
pub fn sse_send<W: Write + ?Sized>(
    stream: &mut W,
    buf: &mut Vec<u8>,
) -> std::io::Result<()> {
    if buf.is_empty() {
        return Ok(());
    }
    let sent = stream.write_all(buf);
    buf.clear();
    sent?;
    stream.flush()
}

/// One frame sent on its own (the role preamble, error frames).
///
/// # Errors
///
/// Returns an I/O error when the socket cannot be written.
pub fn sse<W: Write + ?Sized>(
    stream: &mut W,
    value: &serde_json::Value,
) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(256);
    sse_frame(&mut buf, value)?;
    sse_send(stream, &mut buf)
}

/// A response written by the engine loop and sent by the connection's own thread, so a slow
/// or stalled client holds its thread, never the engine. Each `flush` is one message: the
/// helpers above flush once per response piece and once per token, so the socket sees the
/// writes it saw when the engine wrote to it directly (task #202).
pub struct Outbox {
    buf: Vec<u8>,
    tx: std::sync::mpsc::Sender<Vec<u8>>,
}

impl Outbox {
    #[must_use]
    pub fn new(tx: std::sync::mpsc::Sender<Vec<u8>>) -> Self {
        Self {
            buf: Vec::with_capacity(512),
            tx,
        }
    }
}

impl Write for Outbox {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    /// # Errors
    ///
    /// `BrokenPipe` once the connection's thread has gone: the client closed the socket, or
    /// a write to it failed.
    fn flush(&mut self) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::replace(&mut self.buf, Vec::with_capacity(512));
        self.tx.send(bytes).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "the client has gone")
        })
    }
}

/// The connection's side of an `Outbox`: every message to the socket, until the engine drops
/// the outbox.
///
/// # Errors
///
/// Returns an I/O error when the socket cannot be written.
pub fn pump(
    stream: &mut TcpStream,
    rx: &std::sync::mpsc::Receiver<Vec<u8>>,
) -> std::io::Result<()> {
    for bytes in rx {
        stream.write_all(&bytes)?;
        stream.flush()?;
    }
    Ok(())
}
