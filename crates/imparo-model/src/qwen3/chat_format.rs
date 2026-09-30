//! The Qwen3 chat-format codec: ChatML turns, a `<think>` reasoning channel, and a
//! JSON tool-call block.
//!
//! The shipping path renders `tokenizer.chat_template` from the GGUF. What lives here is
//! the fallback for a template-less file, plus the OUTPUT parser, which no template can
//! provide.
//!
//! TWO THINGS DIFFER FROM Qwen3.5, and both were read from this model's own template
//! rather than carried over:
//!
//!   - A tool call is JSON inside the block, not nested XML:
//!
//!     ```text
//!     <tool_call>
//!     {"name": "get_weather", "arguments": {"city": "Paris"}}
//!     </tool_call>
//!     ```
//!
//!   - THE GENERATION PROMPT DOES NOT OPEN THE REASONING CHANNEL. With thinking on the
//!     template ends at `<|im_start|>assistant\n` and the model writes `<think>` itself;
//!     with `enable_thinking` false the template writes a CLOSED empty channel,
//!     `<think>\n\n</think>\n\n`. Qwen3.5 and LFM2 both leave the channel open instead, so
//!     assuming a model's answer starts inside it would file this model's whole reply as
//!     reasoning and return empty content.
//!
//! A tool result comes back as a `user` turn carrying `<tool_response>`, which is why
//! `user_turn_open` is what it is.

use serde_json::Value;

pub const TURN_OPEN: &str = "<|im_start|>";
pub const TURN_CLOSE: &str = "<|im_end|>";
pub const ASSISTANT_TURN_OPEN: &str = "<|im_start|>assistant\n";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const CALL_OPEN: &str = "<tool_call>";
pub const CALL_CLOSE: &str = "</tool_call>";
const RESPONSE_OPEN: &str = "<tool_response>";
const RESPONSE_CLOSE: &str = "</tool_response>";

/// Markers this parser needs that a rendered template must also emit.
///
/// A template for another model renders fine while reasoning and tool-call extraction
/// match nothing at all; the server reports that at load instead of returning every
/// answer with its reasoning inlined.
#[must_use]
pub fn markers_missing_from(template_src: &str) -> Vec<&'static str> {
    [TURN_OPEN, THINK_CLOSE, CALL_OPEN, CALL_CLOSE]
        .into_iter()
        .filter(|marker| !template_src.contains(marker))
        .collect()
}

fn role_of(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

/// The text of a message's content, whether it is a string or the multi-part list the
/// OpenAI shape allows.
fn render_content(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn compact_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".into())
}

/// Renders messages as ChatML, the fallback when the file carries no usable template.
#[must_use]
pub fn render(
    messages: &[Value],
    tools: &[Value],
    add_generation_prompt: bool,
    _bos: &str,
) -> String {
    let mut out = String::new();
    let mut first = 0_usize;
    let mut system = String::new();

    if messages.first().and_then(role_of) == Some("system") {
        if let Some(content) = messages[0].get("content") {
            system = render_content(content);
        }
        first = 1;
    }
    if !tools.is_empty() {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(
            "# Tools\n\nYou may call one or more functions to assist with the user \
             query.\n\nYou are provided with function signatures within <tools></tools> \
             XML tags:\n<tools>",
        );
        for tool in tools {
            system.push('\n');
            if let Some(text) = tool.as_str() {
                system.push_str(text);
            } else {
                system.push_str(&compact_json(tool));
            }
        }
        system.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function \
             name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
             {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );
    }
    if !system.is_empty() {
        out.push_str(TURN_OPEN);
        out.push_str("system\n");
        out.push_str(&system);
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    let rest = &messages[first.min(messages.len())..];
    let mut index = 0_usize;
    while index < rest.len() {
        let message = &rest[index];
        let role = role_of(message).unwrap_or("user");
        let content = message
            .get("content")
            .map(render_content)
            .unwrap_or_default();

        // A RUN OF TOOL RESULTS IS ONE USER TURN. The template opens the turn at the
        // first of the run and closes it after the last, so results that arrive together
        // are not split into turns the model never saw in training.
        if role == "tool" {
            out.push_str(TURN_OPEN);
            out.push_str("user");
            while index < rest.len() && role_of(&rest[index]) == Some("tool") {
                let body = rest[index]
                    .get("content")
                    .map(render_content)
                    .unwrap_or_default();
                out.push('\n');
                out.push_str(RESPONSE_OPEN);
                out.push('\n');
                out.push_str(&body);
                out.push('\n');
                out.push_str(RESPONSE_CLOSE);
                index += 1;
            }
            out.push_str(TURN_CLOSE);
            out.push('\n');
            continue;
        }

        out.push_str(TURN_OPEN);
        out.push_str(role);
        out.push('\n');
        out.push_str(&content);
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let call = call.get("function").unwrap_or(call);
                let Some(name) = call.get("name").and_then(Value::as_str) else {
                    continue;
                };
                out.push('\n');
                out.push_str(CALL_OPEN);
                out.push_str("\n{\"name\": \"");
                out.push_str(name);
                out.push_str("\", \"arguments\": ");
                // The template passes a string through unchanged and serialises anything
                // else, so arguments already carried as JSON text are not double-encoded.
                match call.get("arguments") {
                    Some(Value::String(text)) => out.push_str(text),
                    Some(value) => out.push_str(&compact_json(value)),
                    None => out.push_str("{}"),
                }
                out.push_str("}\n");
                out.push_str(CALL_CLOSE);
            }
        }
        out.push_str(TURN_CLOSE);
        out.push('\n');
        index += 1;
    }

    if add_generation_prompt {
        out.push_str(TURN_OPEN);
        out.push_str("assistant\n");
        // AND NOTHING ELSE. The model opens its own `<think>`; see the module note.
    }
    out
}

/// Splits `<think>...</think>` into `(reasoning, visible)`.
///
/// `starts_inside` comes from the rendered prompt, never from this model: both of its
/// template branches leave the channel closed, but a caller that supplies its own prompt
/// may not.
#[must_use]
pub fn split_channels(text: &str, starts_inside: bool) -> (String, String) {
    crate::chat::think::split(text, THINK_OPEN, THINK_CLOSE, starts_inside)
}

/// Did the rendered prompt leave the reasoning channel open?
#[must_use]
pub fn prompt_ends_in_reasoning(prompt: &str) -> bool {
    crate::chat::think::ends_inside(
        prompt,
        ASSISTANT_TURN_OPEN,
        THINK_OPEN,
        THINK_CLOSE,
    )
}

/// Holdback for either reasoning boundary, whichever suffix could still complete.
#[must_use]
pub fn reasoning_marker_holdback(text: &str) -> usize {
    crate::chat::think::holdback(text, THINK_OPEN, THINK_CLOSE)
}

/// Extracts every closed `<tool_call>` block, returning `(visible_text, [(name, json)])`.
///
/// The block body is one JSON object with `name` and `arguments`. A body that does not
/// parse, or that names nothing, is left in the visible text rather than reported as a
/// call the caller would then try to run.
#[must_use]
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut visible = String::new();
    let mut calls = Vec::new();
    let mut rest = text;

    while let Some(at) = rest.find(CALL_OPEN) {
        let after = &rest[at + CALL_OPEN.len()..];
        let Some(end) = after.find(CALL_CLOSE) else {
            break;
        };
        match parse_one_call(&after[..end]) {
            Some(call) => {
                visible.push_str(&rest[..at]);
                calls.push(call);
            }
            // Not a call: keep the whole block, markers included, as text.
            None => {
                visible
                    .push_str(&rest[..at + CALL_OPEN.len() + end + CALL_CLOSE.len()]);
            }
        }
        rest = &after[end + CALL_CLOSE.len()..];
    }
    visible.push_str(rest);
    (visible, calls)
}

/// One `{"name": ..., "arguments": ...}` body.
fn parse_one_call(body: &str) -> Option<(String, String)> {
    let parsed: Value = serde_json::from_str(body.trim()).ok()?;
    let name = parsed
        .get("name")
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    if name.is_empty() {
        return None;
    }
    // Arguments may arrive as an object or as a JSON string; the caller is given one
    // spelling, the compact object text.
    let arguments = match parsed.get("arguments") {
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).map_or_else(
            |_| compact_json(&Value::String(text.clone())),
            |v| compact_json(&v),
        ),
        Some(value) => compact_json(value),
        None => "{}".into(),
    };
    Some((name, arguments))
}

/// The codec, keyed on the architecture by [`crate::chat::codec`].
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
        split_channels(text, starts_inside)
    }
    fn prompt_ends_in_reasoning(&self, prompt: &str) -> bool {
        prompt_ends_in_reasoning(prompt)
    }
    /// ChatML ends the assistant's turn at TURN_CLOSE, which is the EOT token the decode
    /// loop already stops at; a tool result arrives as a NEW turn rather than inside this
    /// one, so nothing in the generated TEXT ends the turn.
    fn turn_ends_at(&self, _text: &str) -> Option<usize> {
        None
    }
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
    use super::*;
    use crate::chat::ChatCodec;
    use serde_json::json;

    /// THE PROMPT DECIDES THE CHANNEL, and this model's generation prompt closes it in
    /// both branches: thinking on ends at the assistant opener, thinking off writes an
    /// empty closed channel. Reading either as "inside" empties every answer.
    #[test]
    fn the_generation_prompt_does_not_open_the_reasoning_channel() {
        for prompt in [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        ] {
            assert!(!Codec.prompt_ends_in_reasoning(prompt), "{prompt:?}");
        }
        // A caller that supplies its own open channel is still read as open.
        assert!(Codec.prompt_ends_in_reasoning("<|im_start|>assistant\n<think>\n"));
        let rendered =
            render(&[json!({"role": "user", "content": "hi"})], &[], true, "");
        assert!(rendered.ends_with(ASSISTANT_TURN_OPEN), "{rendered:?}");
        assert!(!Codec.prompt_ends_in_reasoning(&rendered));
    }

    /// The model writes `<think>` itself, so a reply that opens and closes the channel
    /// splits into both halves from a prompt that was outside it.
    #[test]
    fn a_reply_that_opens_its_own_channel_splits() {
        let (reasoning, visible) =
            Codec.split_channels("<think>\nweighing\n</think>\n\nthe answer", false);
        assert_eq!(reasoning.trim(), "weighing");
        assert_eq!(visible.trim(), "the answer");
    }

    /// Tool calls are JSON here. Arguments given as a JSON string and as an object must
    /// come back the same way, because the caller runs one of them.
    #[test]
    fn json_tool_calls_parse_either_spelling_of_arguments() {
        let text = "before<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \
                    {\"city\": \"Paris\"}}\n</tool_call>after";
        let (visible, calls) = Codec.parse_tool_calls(text);
        assert_eq!(visible, "beforeafter");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "get_weather");
        assert_eq!(calls[0].1, "{\"city\":\"Paris\"}");

        let as_string = "<tool_call>\n{\"name\": \"f\", \"arguments\": \"{\\\"k\\\": 1}\"}\n\
                         </tool_call>";
        let (_, calls) = Codec.parse_tool_calls(as_string);
        assert_eq!(calls[0].1, "{\"k\":1}");
    }

    /// A block whose body is not a call is prose: it keeps its markers and reports
    /// nothing, so a caller never invokes a function the model did not ask for.
    #[test]
    fn an_unparsable_block_stays_visible_and_reports_no_call() {
        let text = "see <tool_call>\nnot json\n</tool_call> above";
        let (visible, calls) = Codec.parse_tool_calls(text);
        assert_eq!(visible, text);
        assert!(calls.is_empty());
    }

    /// What `render` emits and what the server scans for must be one string.
    #[test]
    fn the_user_turn_opener_is_what_render_emits() {
        let open = Codec.user_turn_open();
        let rendered =
            render(&[json!({"role": "user", "content": "hi"})], &[], false, "");
        assert!(rendered.starts_with(open), "{rendered:?} vs {open}");
    }

    /// Consecutive tool results share one user turn, as the template writes them.
    #[test]
    fn a_run_of_tool_results_is_one_turn() {
        let rendered = render(
            &[
                json!({"role": "tool", "content": "first"}),
                json!({"role": "tool", "content": "second"}),
                json!({"role": "user", "content": "and?"}),
            ],
            &[],
            false,
            "",
        );
        assert_eq!(rendered.matches("<|im_start|>user").count(), 2);
        assert_eq!(rendered.matches(RESPONSE_OPEN).count(), 2);
    }

    /// An assistant turn carrying a call renders the JSON body the parser reads back.
    #[test]
    fn rendering_a_call_and_reading_it_back_round_trips() {
        let rendered = render(
            &[json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "sql", "arguments": {"q": "select 1"}}}]
            })],
            &[],
            false,
            "",
        );
        let (_, calls) = Codec.parse_tool_calls(&rendered);
        assert_eq!(calls, vec![("sql".into(), "{\"q\":\"select 1\"}".into())]);
    }
}
