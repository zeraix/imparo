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
pub fn json(
    stream: &mut TcpStream,
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
pub fn sse_headers(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    stream.flush()
}

/// # Errors
///
/// Returns an I/O error when the socket cannot be written.
pub fn sse(stream: &mut TcpStream, value: &serde_json::Value) -> std::io::Result<()> {
    write!(stream, "data: {value}\n\n")?;
    stream.flush()
}
