//! LFM2.5 fallback chat codec.
//!
//! The shipping path should render `tokenizer.chat_template` from the GGUF. This
//! module is the verified fallback for a template-less file (or a template engine
//! failure), plus the architecture-specific reasoning and tool-call output parser.
//! It deliberately contains no token ids: BOS/EOS remain vocabulary metadata facts.

use serde_json::Value;

pub const TURN_OPEN: &str = "<|im_start|>";
pub const TURN_CLOSE: &str = "<|im_end|>";
/// What OPENS the assistant's turn. `ends_inside` scans only past the last one: a
/// `<think>` in an older turn or in user text is not this turn opening a channel.
pub const ASSISTANT_TURN_OPEN: &str = "<|im_start|>assistant\n";
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
///
/// The walk itself is `crate::chat::think`: qwen35 wraps its reasoning in the same
/// two markers, and one reader of a marker pair is one piece of code.
#[must_use]
pub fn split_reasoning(text: &str, starts_in_reasoning: bool) -> (String, String) {
    crate::chat::think::split(text, THINK_OPEN, THINK_CLOSE, starts_in_reasoning)
}

/// Convenience for the target template, whose generation prompt ends in `<think>`.
#[must_use]
pub fn split_channels(text: &str) -> (String, String) {
    split_reasoning(text, true)
}

/// Length of the longest strict suffix that can still grow into `marker` on the
/// next streaming delta.
#[must_use]
pub fn partial_marker_len(text: &str, marker: &str) -> usize {
    crate::chat::think::partial_marker_len(text, marker)
}

/// Maximum streaming holdback needed for either LFM reasoning boundary.
#[must_use]
pub fn reasoning_marker_holdback(text: &str) -> usize {
    crate::chat::think::holdback(text, THINK_OPEN, THINK_CLOSE)
}

/// Extracts `<|tool_call_start|>[name(arg=value)]<|tool_call_end|>` blocks,
/// returning `(visible_text, [(name, json_arguments), ...])`.
///
/// Parses everything it is given: a block that never closes is prose, so it stays visible.
/// A streaming caller cuts with `unsettled_in_visible` first -- see
/// [`crate::chat::ChatCodec`].
#[must_use]
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut visible = String::new();
    let mut calls = Vec::new();
    let mut rest = text;

    while let Some(at) = rest.find(CALL_OPEN) {
        let after_open = &rest[at + CALL_OPEN.len()..];
        // Nothing is pushed until the block is known to close: on the break `rest` still
        // points at the opener, and `visible_end` below decides how much of it to show.
        let Some(end) = after_open.find(CALL_CLOSE) else {
            break;
        };
        visible.push_str(&rest[..at]);
        let block = &rest[at..at + CALL_OPEN.len() + end + CALL_CLOSE.len()];
        if let Some(parsed) = parse_tool_block(&after_open[..end]) {
            calls.extend(parsed);
        } else {
            // A closed block that does not parse is prose, not a call: show it.
            visible.push_str(block);
        }
        rest = &after_open[end + CALL_CLOSE.len()..];
    }
    visible.push_str(rest);
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
    fn split_channels(&self, text: &str, starts_inside: bool) -> (String, String) {
        split_reasoning(text, starts_inside)
    }
    fn prompt_ends_in_reasoning(&self, prompt: &str) -> bool {
        crate::chat::think::ends_inside(prompt, ASSISTANT_TURN_OPEN, THINK_OPEN, THINK_CLOSE)
    }
    /// ChatML ends the assistant's turn at TURN_CLOSE, which is the EOT token -- the decode
    /// loop already stops there, and a tool result arrives as a NEW `<|im_start|>tool` turn
    /// rather than inside this one. So nothing in the generated TEXT ends the turn.
    fn turn_ends_at(&self, _text: &str) -> Option<usize> {
        None
    }
    /// Either `<think>` marker can straddle a delta, and a tool block that has opened and
    /// not closed is not yet known to be a call at all -- both are unsettled. A channel that
    /// is open but unclosed is settled: its text is reasoning either way.
    fn unsettled_in_raw(&self, text: &str) -> usize {
        reasoning_marker_holdback(text)
    }
    fn unsettled_in_visible(&self, text: &str) -> usize {
        crate::chat::think::unsettled_pair(text, CALL_OPEN, CALL_CLOSE)
    }
    fn channel_state_after(&self, text: &str, before: bool) -> bool {
        crate::chat::think::state_after(text, THINK_OPEN, THINK_CLOSE, before)
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
    /// The RENDERED PROMPT decides which channel the answer starts in, and only the
    /// CURRENT assistant turn counts: a `<think>` a user pasted, or one left in an older
    /// turn, is not this turn opening a channel.
    #[test]
    fn response_channel_follows_the_current_assistant_turn() {
        use crate::chat::ChatCodec;
        let codec = super::Codec;
        for (prompt, inside) in [
            ("<|im_start|>assistant\n<think>", true),
            ("<|im_start|>assistant\n<think></think>", false),
            ("<|im_start|>assistant\n<think></think>\n\n", false),
            // In USER text, so not the model's.
            ("<|im_start|>user\n<think><|im_end|>\n<|im_start|>assistant\n", false),
            // In an OLDER assistant turn, so not this one's.
            ("<|im_start|>assistant\n<think>old<|im_end|>\n<|im_start|>assistant\n", false),
            ("<|im_start|>assistant\n<think></think><think>", true),
        ] {
            assert_eq!(codec.prompt_ends_in_reasoning(prompt), inside, "{prompt:?}");
            let got = codec.split_channels("normal answer", inside);
            let want = if inside {
                ("normal answer".to_string(), String::new())
            } else {
                (String::new(), "normal answer".to_string())
            };
            assert_eq!(got, want, "{prompt:?}");
        }
        assert_eq!(
            codec.split_channels("before<think>hidden</think>after", false),
            ("hidden".to_string(), "beforeafter".to_string())
        );
    }

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
        use crate::chat::ChatCodec;
        let text =
            "before<|tool_call_start|>[weather(city='Paris')]<|tool_call_end|>after";
        let close_at = text.find(CALL_CLOSE).unwrap() + CALL_CLOSE.len();
        let mut prior_visible = String::new();
        let mut prior_calls = 0_usize;
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let seen = &text[..end];
            let settled = &seen[..seen.len() - Codec.unsettled_in_visible(seen)];
            let (visible, calls) = parse_tool_calls(settled);
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

        let streamed = parse_tool_calls(text);
        assert_eq!(streamed.0, "beforeafter");
        assert_eq!(streamed.1.len(), 1);
    }

    /// The parse always SHOWS an unclosed opener -- text the model wrote is never silently
    /// eaten. Holding it while it might still close is `unsettled_in_visible`, one stage
    /// earlier, on text the split has already classified as visible.
    #[test]
    fn an_unclosed_opener_is_shown_by_the_parse_and_held_by_the_cut() {
        use crate::chat::ChatCodec;
        fn cut(t: &str) -> &str {
            &t[..t.len() - Codec.unsettled_in_visible(t)]
        }
        assert_eq!(
            parse_tool_calls(cut("visible<|tool_call_sta")),
            ("visible".to_string(), Vec::new())
        );
        assert_eq!(
            parse_tool_calls(cut("visible<|tool_call_start|>[ping()")),
            ("visible".to_string(), Vec::new())
        );
        // A final parse -- past the cut, at the end of generation -- loses nothing.
        assert_eq!(
            parse_tool_calls("visible<|tool_call_start|>[ping()"),
            ("visible<|tool_call_start|>[ping()".to_string(), Vec::new())
        );
    }

    /// The RAW cut is about the CHANNEL marker, the VISIBLE cut about the CALL block, and
    /// they run in that order. A `<|tool_call_start|>` the model writes inside `<think>` is
    /// prose that never closes: asking the call question on raw text held the reasoning,
    /// the `</think>` and the whole answer after it to the end of generation.
    #[test]
    fn the_two_cuts_ask_one_question_each() {
        use crate::chat::ChatCodec;
        assert_eq!(Codec.unsettled_in_raw("answer</thi"), 5);
        assert_eq!(Codec.unsettled_in_raw("plain text"), 0);
        assert_eq!(Codec.unsettled_in_raw("a<|tool_call_start|>[ping()"), 0);
        assert_eq!(Codec.unsettled_in_visible("a<|tool_call_start|>[ping()"), 26);
        assert_eq!(Codec.unsettled_in_visible("plain text"), 0);

        let text = "<think>I could <|tool_call_start|>[ping()] here</think>the answer";
        assert_eq!(Codec.unsettled_in_raw(text), 0, "raw text is fully settled");
        // What the ONE-cut form answered on this text, which is why it stalled: everything
        // from the opener on, so the `</think>` and the answer after it never went out.
        assert_eq!(
            crate::chat::think::unsettled_pair(text, CALL_OPEN, CALL_CLOSE),
            text.len() - "<think>I could ".len()
        );
        let (r, v) = Codec.split_channels(text, false);
        assert_eq!(v, "the answer");
        assert!(r.contains(CALL_OPEN), "it stays in the reasoning, as prose");
        assert_eq!(Codec.unsettled_in_visible(&v), 0);
    }
}
