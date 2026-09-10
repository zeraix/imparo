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
/// What OPENS the assistant's turn. `ends_inside` scans only past the last one: a
/// `<|channel>` in an older turn or in user text is not this turn opening one.
pub const ASSISTANT_TURN_OPEN: &str = "<|turn>model\n";

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
///
/// `starts_inside` means this piece BEGINS inside the thought channel, because the previous
/// piece ended there. It matters only for streaming, which classifies one settled piece at a
/// time -- but it matters completely: ignoring it files a continued reasoning body as the
/// visible answer.
#[must_use]
pub fn split_channels(text: &str, starts_inside: bool) -> (String, String) {
    let mut reasoning = String::new();
    let mut visible = String::new();
    let mut rest = text;
    if starts_inside {
        let Some(end) = rest.find(CHANNEL_CLOSE) else {
            reasoning.push_str(rest);
            return (reasoning, visible);
        };
        reasoning.push_str(&rest[..end]);
        rest = &rest[end + CHANNEL_CLOSE.len()..];
    }
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
///
/// A streaming caller cuts with `unsettled_in_visible` first; see [`crate::chat::ChatCodec`].
#[must_use]
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut visible = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(CALL_OPEN) {
        let after = &rest[at + CALL_OPEN.len()..];
        // NOTHING IS PUSHED UNTIL THE BLOCK IS KNOWN TO CLOSE. Pushing `rest[..at]` first and
        // then breaking left `rest` still pointing at the opener, so the tail push below
        // emitted the text before the call a SECOND time: `abc<|tool_call>x` came back as
        // `abcabc<|tool_call>x`.
        let Some(end) = after.find(CALL_CLOSE) else {
            break;
        };
        visible.push_str(&rest[..at]);
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
    fn split_channels(&self, text: &str, starts_inside: bool) -> (String, String) {
        split_channels(text, starts_inside)
    }
    /// gemma4's generation prompt opens a TURN, never the channel: the model emits
    /// `<|channel>` itself when it reasons. So the answer is read off the prompt like every
    /// other format, and for this one it is false in practice rather than by assumption.
    fn prompt_ends_in_reasoning(&self, prompt: &str) -> bool {
        crate::chat::think::ends_inside(prompt, ASSISTANT_TURN_OPEN, CHANNEL_OPEN, CHANNEL_CLOSE)
    }
    /// gemma4 hands the turn to the caller with RESP_OPEN after a tool call. See the trait.
    fn turn_ends_at(&self, text: &str) -> Option<usize> {
        text.find(RESP_OPEN)
    }
    /// Two things can be mid-arrival in gemma4's raw output, and the longer hold wins:
    ///
    /// ```text
    ///   ...<|chan                       a partial marker of any kind
    ///   ...<|channel>thou               a channel whose NAME has not arrived
    /// ```
    ///
    /// A channel that is OPEN with a finished name is settled -- `thought` streams as
    /// reasoning, and any other name's body is dropped, so neither waits for the close.
    fn unsettled_in_raw(&self, text: &str) -> usize {
        let mut hold = partial_marker_len(text, CHANNEL_OPEN)
            .max(partial_marker_len(text, CHANNEL_CLOSE))
            .max(partial_marker_len(text, RESP_OPEN));
        // A CHANNEL'S NAME DECIDES WHETHER ITS BODY IS KEPT, and the split reads it up to the
        // first newline. Classify `<|channel>thou` early and the body is dropped as an unknown
        // channel, then reclassified as reasoning once the name finishes.
        if let Some(at) = text.rfind(CHANNEL_OPEN) {
            let after = &text[at + CHANNEL_OPEN.len()..];
            let unsettled = match after.find('\n') {
                // The name line has not arrived: nothing about this channel is decided.
                None => true,
                // Named and still open. A `thought` body streams piece by piece -- that is the
                // point of a thinking channel. Any other name's body is dropped by the split,
                // so holding it to the close costs nothing and keeps the carried state a bool.
                Some(nl) => {
                    !after[nl..].contains(CHANNEL_CLOSE) && after[..nl].trim() != "thought"
                }
            };
            if unsettled {
                hold = hold.max(text.len() - at);
            }
        }
        hold
    }
    fn unsettled_in_visible(&self, text: &str) -> usize {
        crate::chat::think::unsettled_pair(text, CALL_OPEN, CALL_CLOSE)
    }
    /// True when `text` leaves us inside the THOUGHT channel specifically -- the one the
    /// split keeps. Walked the same way the split walks it, on a settled piece.
    ///
    /// `think::state_after` is not enough here: gemma4's channels are NAMED, so "inside a
    /// channel" and "inside the channel whose body is reasoning" are different answers, and
    /// `unsettled_in_raw` holds an open channel with any other name rather than cut inside it.
    fn channel_state_after(&self, text: &str, before: bool) -> bool {
        let mut rest = text;
        if before {
            let Some(end) = rest.find(CHANNEL_CLOSE) else {
                return true;
            };
            rest = &rest[end + CHANNEL_CLOSE.len()..];
        }
        while let Some(at) = rest.find(CHANNEL_OPEN) {
            let after = &rest[at + CHANNEL_OPEN.len()..];
            let (name, body) = match after.find('\n') {
                Some(nl) => (&after[..nl], &after[nl + 1..]),
                None => ("", after),
            };
            match body.find(CHANNEL_CLOSE) {
                Some(end) => rest = &body[end + CHANNEL_CLOSE.len()..],
                None => return name.trim() == "thought",
            }
        }
        false
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
        assert!(
            open.starts_with(super::TURN_OPEN),
            "{open} vs {}",
            super::TURN_OPEN
        );
        assert!(open.ends_with("user"));
        let rendered = super::Codec.render(
            &[serde_json::json!({"role": "user", "content": "hi"})],
            &[],
            false,
            "",
        );
        assert!(
            rendered.contains(open),
            "render did not emit {open}: {rendered}"
        );
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
            let (r, v) = split_channels(&text[..end], false);
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
        let (r, v) = split_channels(text, false);
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
            let whole = split_channels(text, false);
            let streamed = stream_split(text);
            assert_eq!(streamed, whole, "text: {text:?}");
        }
    }

    /// The turn ENDS at the handover marker, and the text after it is the caller's.
    ///
    /// The fixture is what gemma-4-E4B actually generated on the second step of a two-tool
    /// loop (dev_harness/toolcall_kv.py). Everything from `<|tool_response>` on is the
    /// model writing the TOOL's answer -- `15C, cloudy` was never returned by any tool.
    #[test]
    fn the_turn_ends_where_the_model_hands_it_over() {
        use crate::chat::ChatCodec;
        let generated = concat!(
            r#"<|tool_call>call:get_weather{city:<|"|>Oslo<|"|>}<tool_call|>"#,
            r#"<|tool_response>response:get_weather{value:<|"|>15C, cloudy<|"|>}<tool_call|>"#,
            "<|tool_response>",
        );
        let at = Codec
            .turn_ends_at(generated)
            .expect("the handover marker is in the text");
        let turn = &generated[..at];
        assert_eq!(
            turn,
            r#"<|tool_call>call:get_weather{city:<|"|>Oslo<|"|>}<tool_call|>"#
        );
        let (visible, calls) = Codec.parse_tool_calls(turn);
        assert_eq!(visible, "", "the turn is the call and nothing else");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1, r#"{"city":"Oslo"}"#);
        assert!(
            !visible.contains("15C"),
            "the fabricated tool answer must not reach the client"
        );
    }

    /// An ordinary answer has no handover, so nothing is cut.
    #[test]
    fn plain_text_has_no_turn_end() {
        use crate::chat::ChatCodec;
        assert_eq!(Codec.turn_ends_at("The capital of France is Paris."), None);
        assert_eq!(
            Codec.turn_ends_at("<|channel>thought\nthinking<channel|>an answer"),
            None
        );
    }

    /// The RAW cut answers what the SPLIT cannot decide, and nothing about tool calls.
    #[test]
    fn the_raw_cut_holds_markers_and_an_unnamed_channel() {
        use crate::chat::ChatCodec;
        let hold = |t: &str| Codec.unsettled_in_raw(t);
        assert_eq!(hold("answer<|chan"), 6);
        assert_eq!(hold("answer<|tool_resp"), 11); // the turn-end marker, never content
        assert_eq!(hold("answer<|"), 2);
        // A channel whose NAME line has not arrived -- the split reads the name up to the
        // first newline and keeps only `thought`, so classifying early drops the body.
        assert_eq!(hold("a<|channel>thou"), 14);
        assert_eq!(hold("a<|channel>thought\nbody"), 0);
        assert_eq!(hold("an ordinary answer"), 0);
        assert_eq!(hold(""), 0);
        // A CALL OPENER IS NOT THIS CUT'S BUSINESS. Asking here would hold the reasoning
        // channel hostage to a close that a thinking model is never going to write.
        assert_eq!(hold("a<|tool_call>call:f{x"), 0);
    }

    /// The VISIBLE cut answers what the PARSE cannot decide, and only that.
    #[test]
    fn the_visible_cut_holds_an_open_call() {
        use crate::chat::ChatCodec;
        let hold = |t: &str| Codec.unsettled_in_visible(t);
        assert_eq!(hold("a<|tool_call>call:f{x"), 20);
        assert_eq!(hold(r#"a<|tool_call>call:f{x:<|"|>1<|"|>}<tool_call|>"#), 0);
        assert_eq!(hold("plain answer"), 0);
        assert_eq!(hold("answer<|tool_c"), 8);
    }

    /// THE CASE THE ONE-CUT FORM BROKE: a `<|tool_call>` the model writes inside its
    /// THINKING is prose, and no close is coming. Cutting on raw text held it, and with it
    /// the `<channel|>` and the entire visible answer, to the end of generation. Split
    /// first and it never reaches the call cut at all.
    #[test]
    fn a_call_marker_inside_reasoning_does_not_stall_the_stream() {
        use crate::chat::ChatCodec;
        let text = "<|channel>thought\nmaybe <|tool_call> would help<channel|>the answer";
        assert_eq!(Codec.unsettled_in_raw(text), 0, "raw text is fully settled");
        // What the ONE-cut form answered on this text, which is why it stalled: everything
        // from the opener on, so the `<channel|>` and the answer after it never went out.
        assert_eq!(
            crate::chat::think::unsettled_pair(text, CALL_OPEN, CALL_CLOSE),
            text.len() - "<|channel>thought\nmaybe ".len()
        );
        let (r, v) = Codec.split_channels(text, false);
        assert_eq!(v, "the answer");
        assert!(r.contains("<|tool_call>"), "it stays in the reasoning, as prose");
        // And the visible half, which is what the parse sees, has nothing pending.
        assert_eq!(Codec.unsettled_in_visible(&v), 0);
    }

    /// Each settled piece says where it leaves the channel for the next one; a piece with
    /// no marker changes nothing.
    #[test]
    fn the_channel_state_carries_from_piece_to_piece() {
        use crate::chat::ChatCodec;
        assert!(!Codec.channel_state_after("plain", false));
        assert!(Codec.channel_state_after("plain", true));
        assert!(Codec.channel_state_after("x<|channel>thought\n", false));
        assert!(!Codec.channel_state_after("mid<channel|>after", true));
    }

    #[test]
    fn streaming_tool_parse_never_retracts_and_never_leaks() {
        use crate::chat::ChatCodec;
        let text = concat!(
            "before",
            r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#,
            "after",
        );
        let close_at = text.find(CALL_CLOSE).unwrap() + CALL_CLOSE.len();
        let (mut prior_visible, mut prior_calls) = (String::new(), 0_usize);
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let settled = &text[..end];
            let settled = &settled[..settled.len() - Codec.unsettled_in_visible(settled)];
            let (visible, calls) = parse_tool_calls(settled);
            assert!(visible.starts_with(&prior_visible), "visible retracted at {end}");
            assert!(!visible.contains(CALL_OPEN), "leaked an opener at {end}");
            assert!(!visible.contains(CALL_CLOSE), "leaked a close at {end}");
            assert!(calls.len() >= prior_calls, "a call was retracted at {end}");
            if end < close_at {
                assert!(calls.is_empty(), "call completed before its closing marker");
            }
            prior_visible = visible;
            prior_calls = calls.len();
        }
        let streamed = parse_tool_calls(text);
        assert_eq!(streamed.0, "beforeafter");
        assert_eq!(streamed.1.len(), 1);
    }

    /// The parse SHOWS what will never close -- dropping it would silently lose text the
    /// model wrote. Holding it while it might still close is the CUT's job, not a flag here.
    #[test]
    fn an_unclosed_call_is_shown_by_the_parse_and_held_by_the_cut() {
        use crate::chat::ChatCodec;
        let text = r#"visible<|tool_call>call:get_weather{ci"#;
        assert_eq!(Codec.parse_tool_calls(text), (text.to_string(), Vec::new()));
        assert_eq!(Codec.unsettled_in_visible(text), text.len() - "visible".len());
        // Cut first, and what is left parses to exactly the settled text.
        let settled = &text[..text.len() - Codec.unsettled_in_visible(text)];
        assert_eq!(Codec.parse_tool_calls(settled), ("visible".to_string(), Vec::new()));
    }

    /// The text before an unclosed call was emitted TWICE: `rest[..at]` was pushed before
    /// the close check, and the break left `rest` still pointing at the opener for the tail
    /// push. Invisible on a complete answer -- only a call truncated at max_tokens reaches
    /// the branch -- and wrong on every prefix, which is what streaming parses.
    #[test]
    fn an_unclosed_call_does_not_duplicate_the_text_before_it() {
        assert_eq!(
            parse_tool_calls("abc<|tool_call>call:x{").0,
            "abc<|tool_call>call:x{"
        );
    }

    /// The SERVER's streaming loop, byte by byte, with each cut in its own stage: the raw
    /// cut before the split, the visible cut before the parse. What the client ends up with
    /// must equal one final parse of the whole text, and no delta may carry call markup.
    #[test]
    fn the_streamed_visible_text_equals_the_final_one() {
        use crate::chat::ChatCodec;
        for text in [
            r#"plain answer<|tool_call>call:f{a:<|"|>b<|"|>}<tool_call|>"#,
            concat!(
                "<|channel>thought\nreasoning<channel|>",
                r#"visible<|tool_call>call:f{a:<|"|>b<|"|>}<tool_call|>tail"#,
            ),
            "no call at all",
            "<|channel>thought\nunterminated reasoning",
            // An opener the model writes in its THINKING, which never closes.
            "<|channel>thought\nI could <|tool_call>call:f{ ...<channel|>the answer",
        ] {
            let (mut fed_raw, mut inside) = (0usize, false);
            let (mut fed_vis, mut vis_split) = (0usize, String::new());
            let (mut r_out, mut v_out) = (String::new(), String::new());
            for end in 1..=text.len() {
                if !text.is_char_boundary(end) {
                    continue;
                }
                let raw_tail = &text[..end][fed_raw..];
                let settled = &raw_tail[..raw_tail.len() - Codec.unsettled_in_raw(raw_tail)];
                if !settled.is_empty() {
                    let (r, v) = Codec.split_channels(settled, inside);
                    r_out.push_str(&r);
                    vis_split.push_str(&v);
                    inside = Codec.channel_state_after(settled, inside);
                    fed_raw += settled.len();
                }
                let vis_tail = &vis_split[fed_vis..];
                let ready = &vis_tail[..vis_tail.len() - Codec.unsettled_in_visible(vis_tail)];
                if !ready.is_empty() {
                    let v = Codec.parse_tool_calls(ready).0;
                    assert!(!v.contains(CALL_OPEN), "streamed an opener: {v:?}");
                    v_out.push_str(&v);
                    fed_vis += ready.len();
                }
            }
            // The tail flush: whatever was still held when generation ended.
            let (r_final, body) = Codec.split_channels(text, false);
            let (v_final, _) = Codec.parse_tool_calls(&body);
            assert!(v_final.starts_with(&v_out), "visible was not a growing prefix: {v_out:?}");
            assert!(r_final.starts_with(&r_out), "reasoning was not a growing prefix: {r_out:?}");
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
