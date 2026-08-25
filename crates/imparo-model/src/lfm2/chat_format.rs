//! LFM2.5 fallback chat codec.
//!
//! The shipping path should render `tokenizer.chat_template` from the GGUF. This
//! module is the verified fallback for a template-less file (or a template engine
//! failure), plus the architecture-specific reasoning and tool-call output parser.
//! It deliberately contains no token ids: BOS/EOS remain vocabulary metadata facts.

use serde_json::Value;

pub const TURN_OPEN: &str = "<|im_start|>";
pub const TURN_CLOSE: &str = "<|im_end|>";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const CALL_OPEN: &str = "<|tool_call_start|>";
pub const CALL_CLOSE: &str = "<|tool_call_end|>";

const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Renders an OpenAI-shaped message list with the LFM2.5 GGUF fallback format.
/// `bos_token` is supplied from tokenizer metadata; the server removes this prefix
/// before encoding when the tokenizer is configured to add BOS automatically.
#[must_use]
pub fn render(
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
    bos_token: &str,
) -> String {
    let mut out = String::from(bos_token);
    let mut first_message = 0_usize;
    let mut system_prompt = String::new();

    if messages.first().and_then(role_of) == Some("system") {
        if let Some(content) = messages[0].get("content") {
            system_prompt = render_content(content);
        }
        first_message = 1;
    }

    if !tools.is_empty() {
        if !system_prompt.is_empty() {
            system_prompt.push('\n');
        }
        system_prompt.push_str("List of tools: [");
        for (index, tool) in tools.iter().enumerate() {
            if index > 0 {
                system_prompt.push_str(", ");
            }
            if let Some(tool) = tool.as_str() {
                system_prompt.push_str(tool);
            } else {
                system_prompt.push_str(&compact_json(tool));
            }
        }
        system_prompt.push(']');
    }

    if !system_prompt.is_empty() {
        out.push_str(TURN_OPEN);
        out.push_str("system\n");
        out.push_str(&system_prompt);
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    let remaining = &messages[first_message..];
    let last_user = remaining
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (role_of(message) == Some("user")).then_some(index)
        })
        .next_back();

    for (index, message) in remaining.iter().enumerate() {
        let role = role_of(message).unwrap_or("user");
        out.push_str(TURN_OPEN);
        out.push_str(role);
        out.push('\n');

        if role == "assistant" {
            // The embedded template discards old hidden reasoning, but retains an
            // assistant continuation after the final user message.
            let keep_thinking = last_user.is_none_or(|last| index > last);
            if keep_thinking {
                if let Some(thinking) = thinking_of(message) {
                    out.push_str(THINK_OPEN);
                    out.push_str(thinking);
                    out.push_str(THINK_CLOSE);
                }
            }

            let mut content = message
                .get("content")
                .map(render_content)
                .unwrap_or_default();
            if !keep_thinking {
                if let Some((_, answer)) = content.rsplit_once(THINK_CLOSE) {
                    content = answer.trim().to_string();
                }
            }
            let continue_final = content.ends_with(CONTINUE_FINAL_MESSAGE_TAG);
            if continue_final {
                content.truncate(content.len() - CONTINUE_FINAL_MESSAGE_TAG.len());
            }
            out.push_str(&content);

            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                if !calls.is_empty() {
                    out.push_str(&render_tool_calls(calls));
                }
            }
            if continue_final {
                out.push_str(CONTINUE_FINAL_MESSAGE_TAG);
            }
        } else if let Some(content) = message.get("content") {
            out.push_str(&render_content(content));
        }

        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    if add_generation_prompt {
        out.push_str(TURN_OPEN);
        out.push_str("assistant\n");
        out.push_str(THINK_OPEN);
    }
    out
}

fn role_of(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn thinking_of(message: &Value) -> Option<&str> {
    ["thinking", "reasoning", "reasoning_content"]
        .into_iter()
        .find_map(|key| {
            message
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
}

fn render_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(items) => {
            let mut out = String::new();
            for item in items {
                match item {
                    Value::String(text) => out.push_str(text),
                    Value::Object(map)
                        if map.get("type").and_then(Value::as_str) == Some("image") =>
                    {
                        out.push_str("<image>");
                    }
                    Value::Object(map)
                        if map.get("type").and_then(Value::as_str) == Some("text") =>
                    {
                        if let Some(text) = map.get("text").and_then(Value::as_str) {
                            out.push_str(text);
                        }
                    }
                    other => out.push_str(&compact_json(other)),
                }
            }
            out
        }
        other => compact_json(other),
    }
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

/// Renders one LFM tool block: `[name(arg=value), ...]`.
#[must_use]
pub fn render_tool_calls(calls: &[Value]) -> String {
    let rendered = calls
        .iter()
        .filter_map(|call| {
            let function = call.get("function").unwrap_or(call);
            let name = function.get("name").and_then(Value::as_str)?;
            let arguments = function.get("arguments");
            let owned;
            let arguments = match arguments {
                Some(Value::Object(map)) => Some(map),
                // OpenAI wire requests commonly carry JSON-encoded arguments. The
                // embedded template rejects these, so the fallback safely decodes them.
                Some(Value::String(text))
                    if !text.trim().is_empty() && text.trim() != "null" =>
                {
                    owned = serde_json::from_str::<Value>(text).ok()?;
                    owned.as_object()
                }
                _ => None,
            };
            let args = arguments
                .into_iter()
                .flat_map(|map| map.iter())
                .map(|(key, value)| format!("{key}={}", render_arg_value(value)))
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("{name}({args})"))
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{CALL_OPEN}[{rendered}]{CALL_CLOSE}")
}

fn render_arg_value(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        let escaped = text
            .replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace('\n', "\\n")
            .replace('\r', "\\r");
        format!("'{escaped}'")
    } else if value.is_object() || value.is_array() {
        compact_json(value)
    } else {
        value.to_string()
    }
}

/// Markers required by the output codec but not derivable from a generic ChatML
/// parser. Callers can use this to validate that the built-in fallback matches a
/// GGUF-provided template before selecting it.
#[must_use]
pub fn markers_missing_from(template_src: &str) -> Vec<&'static str> {
    [TURN_OPEN, THINK_CLOSE, CALL_OPEN, CALL_CLOSE]
        .into_iter()
        .filter(|marker| !template_src.contains(marker))
        .collect()
}

/// Splits LFM's `<think>...</think>` channel into `(reasoning, visible)`.
///
/// `starts_in_reasoning` must be true when the generation prompt ended in
/// `<think>` (the target GGUF does), because generated deltas then begin inside the
/// channel and usually contain only the closing marker.
#[must_use]
pub fn split_reasoning(text: &str, starts_in_reasoning: bool) -> (String, String) {
    let mut reasoning = String::new();
    let mut visible = String::new();
    let mut rest = text;
    let mut in_reasoning = starts_in_reasoning;

    while !rest.is_empty() {
        if in_reasoning {
            let open = rest.find(THINK_OPEN);
            let close = rest.find(THINK_CLOSE);
            match earliest(open, close) {
                Some((at, Marker::Open)) => {
                    reasoning.push_str(&rest[..at]);
                    rest = &rest[at + THINK_OPEN.len()..];
                }
                Some((at, Marker::Close)) => {
                    reasoning.push_str(&rest[..at]);
                    rest = &rest[at + THINK_CLOSE.len()..];
                    in_reasoning = false;
                }
                None => {
                    reasoning.push_str(rest);
                    break;
                }
            }
        } else if let Some(at) = rest.find(THINK_OPEN) {
            visible.push_str(&rest[..at]);
            rest = &rest[at + THINK_OPEN.len()..];
            in_reasoning = true;
        } else {
            visible.push_str(rest);
            break;
        }
    }
    (reasoning, visible)
}

/// Convenience for the target template, whose generation prompt ends in `<think>`.
#[must_use]
pub fn split_channels(text: &str) -> (String, String) {
    split_reasoning(text, true)
}

#[derive(Clone, Copy)]
enum Marker {
    Open,
    Close,
}

fn earliest(open: Option<usize>, close: Option<usize>) -> Option<(usize, Marker)> {
    match (open, close) {
        (Some(open), Some(close)) if open <= close => Some((open, Marker::Open)),
        (Some(_) | None, Some(close)) => Some((close, Marker::Close)),
        (Some(open), None) => Some((open, Marker::Open)),
        (None, None) => None,
    }
}

/// Length of the longest strict suffix that can still grow into `marker` on the
/// next streaming delta.
#[must_use]
pub fn partial_marker_len(text: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(text.len());
    (1..=max)
        .rev()
        .find(|&length| text.ends_with(&marker[..length]))
        .unwrap_or(0)
}

/// Maximum streaming holdback needed for either LFM reasoning boundary.
#[must_use]
pub fn reasoning_marker_holdback(text: &str) -> usize {
    partial_marker_len(text, THINK_OPEN).max(partial_marker_len(text, THINK_CLOSE))
}

/// Extracts `<|tool_call_start|>[name(arg=value)]<|tool_call_end|>` blocks,
/// returning `(visible_text, [(name, json_arguments), ...])`.
#[must_use]
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    parse_tool_calls_impl(text, false)
}

/// Parses every completed tool block in a streaming prefix while holding back a
/// complete-but-unclosed block and any suffix that can still become [`CALL_OPEN`].
/// The returned visible text is monotonic as more bytes arrive, so an SSE caller
/// never has to retract tool syntax it already emitted as ordinary content.
#[must_use]
pub fn parse_tool_calls_prefix(text: &str) -> (String, Vec<(String, String)>) {
    parse_tool_calls_impl(text, true)
}

fn parse_tool_calls_impl(
    text: &str,
    hold_incomplete: bool,
) -> (String, Vec<(String, String)>) {
    let mut visible = String::new();
    let mut calls = Vec::new();
    let mut rest = text;

    while let Some(at) = rest.find(CALL_OPEN) {
        visible.push_str(&rest[..at]);
        let after_open = &rest[at + CALL_OPEN.len()..];
        let Some(end) = after_open.find(CALL_CLOSE) else {
            if !hold_incomplete {
                visible.push_str(&rest[at..]);
            }
            return (visible, calls);
        };
        let whole = &rest[at..at + CALL_OPEN.len() + end + CALL_CLOSE.len()];
        if let Some(parsed) = parse_tool_block(&after_open[..end]) {
            calls.extend(parsed);
        } else {
            visible.push_str(whole);
        }
        rest = &after_open[end + CALL_CLOSE.len()..];
    }
    if hold_incomplete {
        let hold = partial_marker_len(rest, CALL_OPEN);
        visible.push_str(&rest[..rest.len() - hold]);
    } else {
        visible.push_str(rest);
    }
    (visible, calls)
}

fn parse_tool_block(body: &str) -> Option<Vec<(String, String)>> {
    let inner = body.trim().strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    split_top_level(inner, ',')
        .into_iter()
        .map(parse_one_call)
        .collect()
}

fn parse_one_call(call: &str) -> Option<(String, String)> {
    let call = call.trim();
    let open = find_top_level(call, '(')?;
    if !call.ends_with(')') {
        return None;
    }
    let name = call[..open].trim();
    if name.is_empty() {
        return None;
    }
    let arguments = &call[open + 1..call.len() - 1];
    let mut map = serde_json::Map::new();
    for argument in split_top_level(arguments, ',') {
        if argument.trim().is_empty() {
            continue;
        }
        let equals = find_top_level(argument, '=')?;
        let key = argument[..equals].trim();
        if key.is_empty() {
            return None;
        }
        map.insert(
            key.to_string(),
            parse_arg_value(argument[equals + 1..].trim()),
        );
    }
    Some((name.to_string(), Value::Object(map).to_string()))
}

fn parse_arg_value(raw: &str) -> Value {
    serde_json::from_str(raw)
        .ok()
        .or_else(|| PythonLiteralParser::new(raw).parse_complete())
        .unwrap_or_else(|| Value::String(raw.to_string()))
}

/// A deliberately small data-only parser for the Python literals emitted by the
/// reference LFM tool grammar. It accepts strings, lists, dictionaries, numbers and
/// `True`/`False`/`None`; it has no names, calls, attributes or expression evaluation.
/// JSON is attempted first, so this parser only supplies Python compatibility.
struct PythonLiteralParser<'a> {
    source: &'a str,
    cursor: usize,
}

impl<'a> PythonLiteralParser<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, cursor: 0 }
    }

    fn parse_complete(mut self) -> Option<Value> {
        let value = self.parse_value()?;
        self.skip_space();
        (self.cursor == self.source.len()).then_some(value)
    }

    fn parse_value(&mut self) -> Option<Value> {
        self.skip_space();
        match self.peek()? {
            '\'' | '"' => self.parse_string().map(Value::String),
            '[' => self.parse_list(),
            '{' => self.parse_dict(),
            _ => self.parse_atom(),
        }
    }

    fn parse_list(&mut self) -> Option<Value> {
        self.expect('[')?;
        let mut values = Vec::new();
        self.skip_space();
        if self.consume(']') {
            return Some(Value::Array(values));
        }
        loop {
            values.push(self.parse_value()?);
            self.skip_space();
            if self.consume(']') {
                return Some(Value::Array(values));
            }
            self.expect(',')?;
            self.skip_space();
            if self.consume(']') {
                return Some(Value::Array(values));
            }
        }
    }

    fn parse_dict(&mut self) -> Option<Value> {
        self.expect('{')?;
        let mut values = serde_json::Map::new();
        self.skip_space();
        if self.consume('}') {
            return Some(Value::Object(values));
        }
        loop {
            self.skip_space();
            let key = match self.peek()? {
                '\'' | '"' => self.parse_string()?,
                _ => self.parse_identifier()?,
            };
            self.skip_space();
            self.expect(':')?;
            values.insert(key, self.parse_value()?);
            self.skip_space();
            if self.consume('}') {
                return Some(Value::Object(values));
            }
            self.expect(',')?;
            self.skip_space();
            if self.consume('}') {
                return Some(Value::Object(values));
            }
        }
    }

    fn parse_string(&mut self) -> Option<String> {
        let quote = self.bump()?;
        if !matches!(quote, '\'' | '"') {
            return None;
        }
        let mut value = String::new();
        loop {
            let ch = self.bump()?;
            if ch == quote {
                return Some(value);
            }
            if ch != '\\' {
                value.push(ch);
                continue;
            }
            match self.bump()? {
                'n' => value.push('\n'),
                'r' => value.push('\r'),
                't' => value.push('\t'),
                'b' => value.push('\u{0008}'),
                'f' => value.push('\u{000c}'),
                next @ ('\\' | '\'' | '"') => value.push(next),
                other => {
                    // Python preserves an unknown escape in the resulting string.
                    value.push('\\');
                    value.push(other);
                }
            }
        }
    }

    fn parse_atom(&mut self) -> Option<Value> {
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|ch| !ch.is_whitespace() && !matches!(ch, ',' | ']' | '}'))
        {
            self.bump();
        }
        let atom = &self.source[start..self.cursor];
        match atom {
            "True" => Some(Value::Bool(true)),
            "False" => Some(Value::Bool(false)),
            "None" => Some(Value::Null),
            "" => None,
            _ => serde_json::from_str(atom).ok(),
        }
    }

    fn parse_identifier(&mut self) -> Option<String> {
        let start = self.cursor;
        while self
            .peek()
            .is_some_and(|ch| ch == '_' || ch.is_alphanumeric())
        {
            self.bump();
        }
        (self.cursor > start).then(|| self.source[start..self.cursor].to_string())
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.bump();
        }
    }

    fn expect(&mut self, expected: char) -> Option<()> {
        (self.bump()? == expected).then_some(())
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<char> {
        self.source[self.cursor..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.cursor += ch.len_utf8();
        Some(ch)
    }
}

fn split_top_level(text: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0_usize;
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;

    for (byte, ch) in text.char_indices() {
        if let Some(active) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' => close_delimiter(&mut stack, '('),
            ']' => close_delimiter(&mut stack, '['),
            '}' => close_delimiter(&mut stack, '{'),
            _ if ch == delimiter && stack.is_empty() => {
                parts.push(&text[start..byte]);
                start = byte + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

fn find_top_level(text: &str, needle: char) -> Option<usize> {
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    for (byte, ch) in text.char_indices() {
        if let Some(active) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active {
                quote = None;
            }
            continue;
        }
        if ch == needle && stack.is_empty() {
            return Some(byte);
        }
        match ch {
            '\'' | '"' => quote = Some(ch),
            '(' | '[' | '{' => stack.push(ch),
            ')' => close_delimiter(&mut stack, '('),
            ']' => close_delimiter(&mut stack, '['),
            '}' => close_delimiter(&mut stack, '{'),
            _ => {}
        }
    }
    None
}

fn close_delimiter(stack: &mut Vec<char>, expected: char) {
    if stack.last() == Some(&expected) {
        stack.pop();
    }
}

/// LFM2's half of the chat seam.
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
        bos: &str,
    ) -> String {
        render(messages, tools, add_generation_prompt, bos)
    }
    fn split_channels(&self, text: &str) -> (String, String) {
        split_channels(text)
    }
    /// The SAME rule on both sides: either `<think>` marker can straddle a delta
    /// wherever the split currently is.
    fn holdback(&self, text: &str, _side: crate::chat::Channel) -> usize {
        reasoning_marker_holdback(text)
    }
    fn parse_tool_calls(&self, text: &str) -> (String, Vec<(String, String)>) {
        parse_tool_calls(text)
    }
    fn user_turn_open(&self) -> &'static str {
        concat!("<|im_start|>", "user")
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

    use serde_json::json;

    use super::*;

    #[test]
    fn fallback_render_matches_target_turn_shape_without_hardcoded_bos_id() {
        let messages = [json!({"role": "user", "content": "Hello"})];
        assert_eq!(
            render(&messages, &[], true, "<bos>"),
            "<bos><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn fallback_strips_past_thinking_and_renders_tools() {
        let tools = [
            json!({"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}),
        ];
        let messages = [
            json!({"role":"system","content":"Be concise."}),
            json!({"role":"user","content":"first"}),
            json!({"role":"assistant","content":"<think>old secret</think> old answer"}),
            json!({"role":"user","content":"again"}),
            json!({
                "role":"assistant",
                "thinking":"fresh thought",
                "content":"answer",
                "tool_calls":[{"function":{"name":"weather","arguments":{"city":"Xi'an", "days":2}}}]
            }),
        ];
        let rendered = render(&messages, &tools, false, "");
        assert!(rendered.starts_with(&format!(
            "<|im_start|>system\nBe concise.\nList of tools: [{}]<|im_end|>\n",
            compact_json(&tools[0])
        )));
        assert!(rendered.contains("<|im_start|>assistant\nold answer<|im_end|>\n"));
        assert!(rendered.contains("<think>fresh thought</think>answer"));
        assert!(rendered.contains(
            "<|tool_call_start|>[weather(city='Xi\\'an', days=2)]<|tool_call_end|>"
        ));
        assert!(!rendered.contains("old secret"));
    }

    fn streamed(text: &str, starts_in_reasoning: bool) -> (String, String) {
        let mut emitted_reasoning = String::new();
        let mut emitted_visible = String::new();
        let mut sent_reasoning = 0_usize;
        let mut sent_visible = 0_usize;
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let (reasoning, visible) =
                split_reasoning(&text[..end], starts_in_reasoning);
            let reasoning_safe =
                reasoning.len() - reasoning_marker_holdback(&reasoning);
            let visible_safe = visible.len() - partial_marker_len(&visible, THINK_OPEN);
            if reasoning_safe > sent_reasoning {
                emitted_reasoning.push_str(&reasoning[sent_reasoning..reasoning_safe]);
                sent_reasoning = reasoning_safe;
            }
            if visible_safe > sent_visible {
                emitted_visible.push_str(&visible[sent_visible..visible_safe]);
                sent_visible = visible_safe;
            }
        }
        let (reasoning, visible) = split_reasoning(text, starts_in_reasoning);
        emitted_reasoning.push_str(&reasoning[sent_reasoning..]);
        emitted_visible.push_str(&visible[sent_visible..]);
        (emitted_reasoning, emitted_visible)
    }

    #[test]
    fn reasoning_split_is_safe_when_markers_cross_stream_deltas() {
        for (text, starts) in [
            ("hidden reasoning</think>visible answer", true),
            ("<think>hidden</think>visible", false),
            ("before<think>hidden</think>after", false),
            ("unicode réason</think>答案", true),
            ("unfinished <thi", false),
            ("unfinished </thi", true),
        ] {
            assert_eq!(
                streamed(text, starts),
                split_reasoning(text, starts),
                "{text:?}"
            );
        }
        assert_eq!(
            split_channels("step one</think>final"),
            ("step one".to_string(), "final".to_string())
        );
    }

    #[test]
    fn tool_parser_round_trips_nested_values_and_escapes() {
        let text = "before<|tool_call_start|>[weather(city='Xi\\'an', opts={\"days\":[1,2]}), ping()]<|tool_call_end|>after";
        let (visible, calls) = parse_tool_calls(text);
        assert_eq!(visible, "beforeafter");
        assert_eq!(calls[0].0, "weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({"city":"Xi'an", "opts":{"days":[1,2]}})
        );
        assert_eq!(calls[1], ("ping".to_string(), "{}".to_string()));
    }

    #[test]
    fn tool_parser_accepts_reference_python_literals_without_evaluation() {
        let text = "<|tool_call_start|>[pkg.weather(opts={'active': True, 'missing': None, 'days': [1, 2], 'label': 'Xi\\'an'}, dry=False)]<|tool_call_end|>";
        let (visible, calls) = parse_tool_calls(text);
        assert!(visible.is_empty());
        assert_eq!(calls[0].0, "pkg.weather");
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].1).unwrap(),
            json!({
                "opts": {
                    "active": true,
                    "missing": null,
                    "days": [1, 2],
                    "label": "Xi'an"
                },
                "dry": false
            })
        );
    }

    #[test]
    fn malformed_or_unterminated_tool_blocks_remain_visible() {
        for text in [
            "x<|tool_call_start|>[broken]<|tool_call_end|>y",
            "x<|tool_call_start|>[ping()]",
        ] {
            assert_eq!(parse_tool_calls(text), (text.to_string(), Vec::new()));
        }
    }

    #[test]
    fn streaming_tool_parser_never_exposes_a_valid_call_as_content() {
        let text =
            "before<|tool_call_start|>[weather(city='Paris')]<|tool_call_end|>after";
        let close_at = text.find(CALL_CLOSE).unwrap() + CALL_CLOSE.len();
        let mut prior_visible = String::new();
        let mut prior_calls = 0_usize;
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let (visible, calls) = parse_tool_calls_prefix(&text[..end]);
            assert!(
                visible.starts_with(&prior_visible),
                "visible output retracted at {end}"
            );
            assert!(!visible.contains(CALL_OPEN));
            assert!(!visible.contains(CALL_CLOSE));
            assert!(
                calls.len() >= prior_calls,
                "completed calls retracted at {end}"
            );
            if end < close_at {
                assert!(calls.is_empty(), "call completed before its closing marker");
            }
            prior_visible = visible;
            prior_calls = calls.len();
        }

        let streamed = parse_tool_calls_prefix(text);
        let complete = parse_tool_calls(text);
        assert_eq!(streamed, complete);
        assert_eq!(streamed.0, "beforeafter");
        assert_eq!(streamed.1.len(), 1);
    }

    #[test]
    fn streaming_tool_parser_holds_partial_and_unclosed_openers() {
        assert_eq!(
            parse_tool_calls_prefix("visible<|tool_call_sta"),
            ("visible".to_string(), Vec::new())
        );
        assert_eq!(
            parse_tool_calls_prefix("visible<|tool_call_start|>[ping()"),
            ("visible".to_string(), Vec::new())
        );
        // A final/non-stream parse loses no malformed model output.
        assert_eq!(
            parse_tool_calls("visible<|tool_call_start|>[ping()"),
            ("visible<|tool_call_start|>[ping()".to_string(), Vec::new())
        );
    }
}
