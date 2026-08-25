//! The gemma4 chat-format codec: renders messages into the model's turn markup,
//! and parses its output back (thought channel, tool calls).
//!
//!
//! The format is taken from the model's own template, verified against llama.cpp's
//! `/apply-template` output. Model-specific; a second architecture adds a module here,
//! not a branch in the server.

use serde_json::Value;

pub const TURN_OPEN: &str = "<|turn>";
pub const TURN_CLOSE: &str = "<turn|>";
pub const THINK: &str = "<|think|>";
pub const Q: &str = "<|\"|>";
pub const TOOL_OPEN: &str = "<|tool>";
pub const TOOL_CLOSE: &str = "<tool|>";
pub const CALL_OPEN: &str = "<|tool_call>";
pub const CALL_CLOSE: &str = "<tool_call|>";
pub const CHANNEL_OPEN: &str = "<|channel>";
pub const CHANNEL_CLOSE: &str = "<channel|>";
pub const RESP_OPEN: &str = "<|tool_response>";
pub const RESP_CLOSE: &str = "<tool_response|>";

fn quoted(s: &str) -> String {
    format!("{Q}{s}{Q}")
}

/// Renders a JSON-schema type name the way the template does: upper-cased.
fn type_name(v: &Value) -> String {
    v.get("type")
        .and_then(Value::as_str)
        .unwrap_or("string")
        .to_uppercase()
}

fn render_properties(props: &Value) -> String {
    let Some(map) = props.as_object() else {
        return String::new();
    };
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort(); // the template uses dictsort
    let parts: Vec<String> = keys
        .iter()
        .map(|k| {
            let v = &map[*k];
            let mut inner = Vec::new();
            if let Some(d) = v.get("description").and_then(Value::as_str) {
                inner.push(format!("description:{}", quoted(d)));
            }
            inner.push(format!("type:{}", quoted(&type_name(v))));
            format!("{k}:{{{}}}", inner.join(","))
        })
        .collect();
    parts.join(",")
}

/// One `<|tool>declaration:...<tool|>` block for a tools array.
#[must_use]
pub fn render_tools(tools: &[Value]) -> String {
    let mut out = String::new();
    for t in tools {
        let Some(f) = t.get("function") else { continue };
        let name = f.get("name").and_then(Value::as_str).unwrap_or("");
        let mut body = Vec::new();
        if let Some(d) = f.get("description").and_then(Value::as_str) {
            body.push(format!("description:{}", quoted(d)));
        }
        if let Some(p) = f.get("parameters") {
            let props = p
                .get("properties")
                .map(render_properties)
                .unwrap_or_default();
            let req: Vec<String> = p
                .get("required")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(quoted).collect())
                .unwrap_or_default();
            let mut inner = vec![format!("properties:{{{props}}}")];
            if !req.is_empty() {
                inner.push(format!("required:[{}]", req.join(",")));
            }
            inner.push(format!("type:{}", quoted(&type_name(p))));
            body.push(format!("parameters:{{{}}}", inner.join(",")));
        }
        out.push_str(&format!(
            "{TOOL_OPEN}declaration:{name}{{{}}}{TOOL_CLOSE}",
            body.join(",")
        ));
    }
    out
}

fn render_args(arguments: &str) -> String {
    let parsed: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let Some(map) = parsed.as_object() else {
        return String::new();
    };
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    keys.iter()
        .map(|k| {
            let v = &map[*k];
            let rendered = v.as_str().map_or_else(|| v.to_string(), quoted);
            format!("{k}:{rendered}")
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Renders an OpenAI message list into a gemma4 prompt.
///
/// Tool messages fold into the preceding model turn, and consecutive assistant
/// messages continue one turn -- both are what the template does.
#[must_use]
/// FALLBACK renderer, hand-coded for gemma4. Since task #15 the shipping path renders
/// the GGUF-embedded jinja template (template.rs, minijinja); this survives only for
/// template-less GGUFs and render failures, because hand-coding a template is what
/// produced the 7-missing-tokens bug (task #7). Verified byte-equal to the fork's
/// /apply-template for gemma4 at the time of writing.
pub fn render(
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
) -> String {
    let mut out = String::new();
    let mut prev_non_tool_role = String::new();
    let mut turn_open = false;
    let mut tools_emitted = false;

    // The model's template emits a system turn EVEN WHEN THE REQUEST HAS NONE -- just the
    // think marker (and the tools, when given). Skipping it made every prompt 7 tokens
    // shorter than llama.cpp's rendering of the same messages (1378 vs 1385, task #7),
    // and dropped the tools entirely for system-less requests. Verified byte-equal
    // against the fork's /apply-template for the no-system case.
    let has_system = messages
        .iter()
        .any(|m| m.get("role").and_then(Value::as_str) == Some("system"));
    if !has_system {
        out.push_str(&format!(
            "{TURN_OPEN}system
"
        ));
        out.push_str(THINK);
        out.push('\n');
        if !tools.is_empty() {
            out.push_str(&render_tools(tools));
        }
        tools_emitted = true;
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content").and_then(Value::as_str).unwrap_or("");

        if role == "tool" {
            let name = m.get("name").and_then(Value::as_str).unwrap_or("");
            out.push_str(&format!(
                "{RESP_OPEN}response:{name}{{value:{}}}{RESP_CLOSE}",
                quoted(content)
            ));
            continue;
        }

        let mapped = if role == "assistant" { "model" } else { role };
        let continues = mapped == "model" && prev_non_tool_role == "assistant";
        if !continues {
            if turn_open {
                out.push_str(TURN_CLOSE);
                out.push('\n');
            }
            out.push_str(&format!("{TURN_OPEN}{mapped}\n"));
            turn_open = true;
        }
        if mapped == "system" && !tools_emitted {
            out.push_str(THINK);
            out.push('\n');
        }
        out.push_str(content);
        if mapped == "system" && !tools_emitted && !tools.is_empty() {
            out.push_str(&render_tools(tools));
            tools_emitted = true;
        }
        if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let f = call.get("function");
                let name = f
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let args = f
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                out.push_str(&format!(
                    "{CALL_OPEN}call:{name}{{{}}}{CALL_CLOSE}",
                    render_args(args)
                ));
            }
        }
        prev_non_tool_role = role.to_string();
    }
    if turn_open {
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }
    if add_generation_prompt {
        out.push_str(&format!("{TURN_OPEN}model\n"));
    }
    out
}

/// The output parser's markers that `template_src` never mentions.
///
/// A chat template only BUILDS prompts; nothing in it drives parsing, so a template
/// paired with the wrong model renders fine while `split_channels`/`parse_tool_calls`
/// silently match nothing. Sniffing the template source for the parse markers is how
/// llama.cpp selects its template families (`tmpl_contains` in llama-chat.cpp); with
/// one model it powers a load-time warning, and a second model turns it into the
/// family selector.
#[must_use]
pub fn markers_missing_from(template_src: &str) -> Vec<&'static str> {
    [TURN_OPEN, CHANNEL_OPEN, CALL_OPEN]
        .into_iter()
        .filter(|m| !template_src.contains(m))
        .collect()
}

/// Splits `<|channel>thought\n...<channel|>` out of the output.
///
/// gemma4 emits its reasoning on a named channel. Leaving it in `content` both corrupts
/// the visible answer and hides tool calls that follow it.
#[must_use]
pub fn split_channels(text: &str) -> (String, String) {
    let mut reasoning = String::new();
    let mut visible = String::new();
    let mut rest = text;
    while let Some(at) = rest.find(CHANNEL_OPEN) {
        visible.push_str(&rest[..at]);
        let after = &rest[at + CHANNEL_OPEN.len()..];
        // the channel name runs to the first newline
        let (name, body) = match after.find('\n') {
            Some(nl) => (&after[..nl], &after[nl + 1..]),
            None => ("", after),
        };
        if let Some(end) = body.find(CHANNEL_CLOSE) {
            if name.trim() == "thought" {
                reasoning.push_str(&body[..end]);
            }
            rest = &body[end + CHANNEL_CLOSE.len()..];
        } else {
            // unterminated channel: everything left belongs to it
            if name.trim() == "thought" {
                reasoning.push_str(body);
            }
            rest = "";
        }
    }
    visible.push_str(rest);
    (reasoning, visible)
}

/// Length of the longest strict suffix of `s` that could still grow into `marker`.
///
/// Streaming splits channels on a PREFIX of the final text, and a marker can straddle
/// two deltas: text ending in `<|chan` is visible today and a channel open tomorrow.
/// Emitting it and then retracting it is not possible over SSE, so the splitter holds
/// back any tail that is a proper prefix of a marker until the next token settles it.
#[must_use]
pub fn partial_marker_len(s: &str, marker: &str) -> usize {
    let max = (marker.len() - 1).min(s.len());
    (1..=max)
        .rev()
        .find(|&k| s.ends_with(&marker[..k]))
        .unwrap_or(0)
}

/// Extracts `<|tool_call>call:name{args}<tool_call|>` blocks, returning
/// (visible_text, tool_calls) where each call is (name, json_arguments).
#[must_use]
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut visible = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(CALL_OPEN) {
        visible.push_str(&rest[..at]);
        let after = &rest[at + CALL_OPEN.len()..];
        let Some(end) = after.find(CALL_CLOSE) else {
            break;
        };
        let body = &after[..end];
        if let Some(stripped) = body.strip_prefix("call:") {
            if let Some(brace) = stripped.find('{') {
                let name = stripped[..brace].trim().to_string();
                let args = stripped[brace + 1..].trim_end_matches('}');
                calls.push((name, args_to_json(args)));
            }
        }
        rest = &after[end + CALL_CLOSE.len()..];
    }
    visible.push_str(rest);
    (visible, calls)
}

/// Turns `path:<|"|>a.rs<|"|>,n:3` back into JSON.
fn args_to_json(args: &str) -> String {
    let mut map = serde_json::Map::new();
    for part in split_top_level(args) {
        let Some(colon) = part.find(':') else {
            continue;
        };
        let key = part[..colon].trim().to_string();
        let raw = part[colon + 1..].trim();
        let value = if let Some(inner) =
            raw.strip_prefix(Q).and_then(|r| r.strip_suffix(Q))
        {
            Value::String(inner.to_string())
        } else {
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
        };
        map.insert(key, value);
    }
    Value::Object(map).to_string()
}

/// Splits on commas that are not inside a quoted marker or nested braces.
fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0_i32;
    let mut in_q = false;
    let mut cur = String::new();
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if s[byte_index(&bytes, i)..].starts_with(Q) {
            in_q = !in_q;
            cur.push_str(Q);
            i += Q.chars().count();
            continue;
        }
        let c = bytes[i];
        match c {
            '{' | '[' if !in_q => {
                depth += 1;
                cur.push(c);
            }
            '}' | ']' if !in_q => {
                depth -= 1;
                cur.push(c);
            }
            ',' if !in_q && depth == 0 => {
                parts.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
        i += 1;
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    parts
}

fn byte_index(chars: &[char], i: usize) -> usize {
    chars.iter().take(i).map(|c| c.len_utf8()).sum()
}

/// gemma4's half of the chat seam.
///
/// `render` ignores `bos`: this fallback never emitted one, and the tokenizer's
/// `add_bos_token` supplies it. LFM2's format starts with the spelling instead -- that is
/// a real difference between the two templates, not a switch.
pub struct Codec;

impl crate::chat::ChatCodec for Codec {
    fn markers_missing_from(&self, template_src: &str) -> Vec<&'static str> {
        markers_missing_from(template_src)
    }
    fn render(
        &self,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: bool,
        _bos: &str,
    ) -> String {
        render(messages, tools, add_generation_prompt)
    }
    fn split_channels(&self, text: &str) -> (String, String) {
        split_channels(text)
    }
    /// A different marker per side: while emitting reasoning the risk is a partial
    /// CHANNEL_CLOSE, and while emitting visible text it is a partial CHANNEL_OPEN.
    fn holdback(&self, text: &str, side: crate::chat::Channel) -> usize {
        partial_marker_len(
            text,
            match side {
                crate::chat::Channel::Reasoning => CHANNEL_CLOSE,
                crate::chat::Channel::Visible => CHANNEL_OPEN,
            },
        )
    }
    fn parse_tool_calls(&self, text: &str) -> (String, Vec<(String, String)>) {
        parse_tool_calls(text)
    }
    fn user_turn_open(&self) -> &'static str {
        // `render` writes `{TURN_OPEN}{role}\n` for every turn, and gemma4 maps
        // assistant -> model but leaves user alone.
        concat!("<|turn>", "user")
    }
}

#[cfg(test)]
mod tests {
    /// The opener the server scans for must be the one `render` writes. Two
    /// spellings of one rule is how they drift apart.
    #[test]
    fn the_user_turn_opener_is_what_render_emits() {
        use crate::chat::ChatCodec;
        let open = super::Codec.user_turn_open();
        assert!(open.starts_with(super::TURN_OPEN), "{open} vs {}", super::TURN_OPEN);
        assert!(open.ends_with("user"));
        let rendered = super::Codec.render(
            &[serde_json::json!({"role": "user", "content": "hi"})],
            &[],
            false,
            "",
        );
        assert!(rendered.contains(open), "render did not emit {open}: {rendered}");
    }

    use super::*;

    /// Feeds text one byte at a time through the streaming prefix-split and asserts the
    /// emitted pieces reassemble to exactly what one split of the full text produces —
    /// markers straddling deltas must be held back, never emitted and reclassified.
    fn stream_split(text: &str) -> (String, String) {
        let (mut out_r, mut out_v) = (String::new(), String::new());
        let (mut sent_r, mut sent_v) = (0usize, 0usize);
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let (r, v) = split_channels(&text[..end]);
            let r_safe = r.len() - partial_marker_len(&r, CHANNEL_CLOSE);
            let v_safe = v.len() - partial_marker_len(&v, CHANNEL_OPEN);
            assert!(
                r.starts_with(&out_r) || r_safe >= sent_r,
                "reasoning retracted"
            );
            assert!(
                v.starts_with(&out_v) || v_safe >= sent_v,
                "visible retracted"
            );
            if r_safe > sent_r {
                out_r.push_str(&r[sent_r..r_safe]);
                sent_r = r_safe;
            }
            if v_safe > sent_v {
                out_v.push_str(&v[sent_v..v_safe]);
                sent_v = v_safe;
            }
        }
        let (r, v) = split_channels(text);
        out_r.push_str(&r[sent_r..]);
        out_v.push_str(&v[sent_v..]);
        (out_r, out_v)
    }

    #[test]
    fn streamed_split_matches_whole_split() {
        for text in [
            "plain answer, no channel",
            "<|channel>thought\nhidden reasoning<channel|>visible answer",
            "before <|channel>thought\nmid<channel|> after",
            "<|channel>thought\nunterminated reasoning tail",
            "<|channel>other\ndropped<channel|>kept",
            "two<|channel>thought\na<channel|>x<|channel>thought\nb<channel|>y",
            "ends mid-open <|chan",
            "<|channel>thought\nends mid-close <chan",
            "unicode caté <|channel>thought\nré—flé<channel|>ok™",
        ] {
            let whole = split_channels(text);
            let streamed = stream_split(text);
            assert_eq!(streamed, whole, "text: {text:?}");
        }
    }

    #[test]
    fn partial_marker_len_basics() {
        assert_eq!(partial_marker_len("abc<|chan", CHANNEL_OPEN), 6);
        assert_eq!(partial_marker_len("abc<", CHANNEL_OPEN), 1);
        assert_eq!(partial_marker_len("abc", CHANNEL_OPEN), 0);
        // a FULL marker in the text is not a partial one
        assert_eq!(partial_marker_len("x<channel|>", CHANNEL_CLOSE), 0);
        assert_eq!(partial_marker_len("", CHANNEL_OPEN), 0);
    }
}
