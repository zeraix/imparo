//! The Qwen3.5 chat-format codec: ChatML turns, a `<think>` reasoning channel, and an
//! XML tool-call block.
//!
//! The shipping path renders `tokenizer.chat_template` from the GGUF (this file's model
//! carries one, 9993 characters). What lives here is the fallback for a template-less
//! file or a template that fails to compile, plus the OUTPUT parser -- which the template
//! cannot provide, and which is why rendering and reading back are one trait.
//!
//! The reasoning channel is `crate::chat::think`, shared with LFM2: the same two markers,
//! the same rule that the generation prompt ends inside the channel. The tool-call block
//! is NOT shared. LFM2 writes JSON between `<|tool_call_start|>` markers; this model
//! writes nested XML, straight from its own template:
//!
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! No token ids anywhere: BOS/EOS stay vocabulary metadata.

use serde_json::Value;

pub const TURN_OPEN: &str = "<|im_start|>";
pub const TURN_CLOSE: &str = "<|im_end|>";
/// What OPENS the assistant's turn. `ends_inside` scans only past the last one: a
/// `<think>` in an older turn or in user text is not this turn opening a channel.
pub const ASSISTANT_TURN_OPEN: &str = "<|im_start|>assistant\n";
pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";
pub const CALL_OPEN: &str = "<tool_call>";
pub const CALL_CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// Markers this parser needs that a rendered template must also emit.
///
/// A template for a different model renders fine while reasoning and tool-call
/// extraction match nothing at all; the server reports that at load rather than
/// returning every response with its reasoning inlined.
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
/// OpenAI shape allows. Non-text parts (this model's template also handles images and
/// video) contribute nothing to a fallback render.
fn render_content(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.to_string();
    }
    let Some(parts) = content.as_array() else {
        return String::new();
    };
    let mut out = String::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            out.push_str(text);
        }
    }
    out
}

fn compact_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

/// Renders an OpenAI-shaped message list in this model's ChatML fallback.
///
/// `bos` is the BOS token's SPELLING from tokenizer metadata. This format does not begin
/// with it -- Qwen's tokenizer adds none -- so the argument is accepted and ignored, the
/// same way gemma4's codec uses it and LFM2's does not.
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
            "# Tools\n\nYou have access to the following functions:\n\n<tools>",
        );
        for tool in tools {
            system.push('\n');
            if let Some(text) = tool.as_str() {
                system.push_str(text);
            } else {
                system.push_str(&compact_json(tool));
            }
        }
        system.push_str("\n</tools>");
    }
    if !system.is_empty() {
        out.push_str(TURN_OPEN);
        out.push_str("system\n");
        out.push_str(&system);
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    for message in &messages[first.min(messages.len())..] {
        let role = role_of(message).unwrap_or("user");
        let content = message
            .get("content")
            .map(render_content)
            .unwrap_or_default();
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
                out.push('\n');
                out.push_str(FN_OPEN);
                out.push_str(name);
                out.push_str(">\n");
                if let Some(args) = call.get("arguments").and_then(Value::as_object) {
                    for (key, value) in args {
                        out.push_str(PARAM_OPEN);
                        out.push_str(key);
                        out.push_str(">\n");
                        out.push_str(&value.as_str().map_or_else(
                            || compact_json(value),
                            std::string::ToString::to_string,
                        ));
                        out.push('\n');
                        out.push_str(PARAM_CLOSE);
                        out.push('\n');
                    }
                }
                out.push_str(FN_CLOSE);
                out.push('\n');
                out.push_str(CALL_CLOSE);
            }
        }
        out.push_str(TURN_CLOSE);
        out.push('\n');
    }

    if add_generation_prompt {
        out.push_str(TURN_OPEN);
        out.push_str("assistant\n");
        // The template opens the channel and leaves it open (its thinking-on branch), so
        // generated text starts INSIDE the reasoning channel. `split_channels` says so too.
        out.push_str(THINK_OPEN);
        out.push('\n');
    }
    out
}

/// Splits `<think>...</think>` into `(reasoning, visible)`.
///
/// `starts_inside` is NOT a property of this model. Its template branches: the generation
/// prompt ends in `<think>\n` with thinking on, and in `<think>\n\n</think>\n\n` with it
/// off. This used to be hardcoded true, so a request with `enable_thinking: false` had its
/// whole answer filed as reasoning and returned an empty `content` -- blank in every client.
/// Ask [`prompt_ends_in_reasoning`].
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
/// The parameters are XML, and their VALUES are free text that may span lines -- so a
/// parameter ends at its own `</parameter>` and nothing else, and the JSON this returns
/// is built from the strings rather than parsed out of the block.
///
/// A block that never closes is prose, so it stays in the visible text. A streaming caller
/// cuts with `unsettled_in_visible` first -- see [`crate::chat::ChatCodec`].
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
        visible.push_str(&rest[..at]);
        if let Some(call) = parse_one_call(&after[..end]) {
            calls.push(call);
        }
        rest = &after[end + CALL_CLOSE.len()..];
    }
    visible.push_str(rest);
    (visible, calls)
}

/// One `<function=NAME> <parameter=KEY>value</parameter>... </function>` body.
fn parse_one_call(body: &str) -> Option<(String, String)> {
    let at = body.find(FN_OPEN)?;
    let after = &body[at + FN_OPEN.len()..];
    let close = after.find('>')?;
    let name = after[..close].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut rest = &after[close + 1..];
    // Parameters are read only up to the function's own close, so a stray `<parameter=`
    // after it cannot be absorbed into this call.
    if let Some(fn_end) = rest.find(FN_CLOSE) {
        rest = &rest[..fn_end];
    }
    while let Some(at) = rest.find(PARAM_OPEN) {
        let after = &rest[at + PARAM_OPEN.len()..];
        let Some(close) = after.find('>') else { break };
        let key = after[..close].trim().to_string();
        let value_and_rest = &after[close + 1..];
        let Some(end) = value_and_rest.find(PARAM_CLOSE) else {
            break;
        };
        // The template writes a newline after the `>` and before the closing tag; they
        // are separators, not part of the value, and a multi-line value keeps its middle.
        let value = value_and_rest[..end]
            .trim_start_matches('\n')
            .trim_end_matches('\n');
        args.insert(key, Value::String(value.to_string()));
        rest = &value_and_rest[end + PARAM_CLOSE.len()..];
    }
    Some((name, compact_json(&Value::Object(args))))
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
        // The prompt decides, not this model: see `split_channels` above.
        split_channels(text, starts_inside)
    }
    fn prompt_ends_in_reasoning(&self, prompt: &str) -> bool {
        prompt_ends_in_reasoning(prompt)
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
    use super::*;
    use crate::chat::ChatCodec;
    use serde_json::json;

    /// The RENDERED PROMPT decides which channel the answer starts in, and only the
    /// CURRENT assistant turn counts -- a `<think>` in user text is not the model's.
    #[test]
    fn response_channel_uses_the_current_assistant_turn() {
        for (prompt, inside) in [
            ("<|im_start|>assistant\n<think>\n", true),
            ("<|im_start|>assistant\n<think>\n\n</think>\n\n", false),
            (
                "<|im_start|>user\n<think><|im_end|>\n<|im_start|>assistant\n",
                false,
            ),
        ] {
            assert_eq!(Codec.prompt_ends_in_reasoning(prompt), inside, "{prompt:?}");
            let (reasoning, visible) = Codec.split_channels("answer", inside);
            assert_eq!(reasoning.is_empty(), !inside, "{prompt:?}");
            assert_eq!(visible.is_empty(), inside, "{prompt:?}");
        }
        assert_eq!(
            Codec.split_channels("before<think>hidden</think>after", false),
            ("hidden".into(), "beforeafter".into())
        );
    }

    /// The opener the server scans for must be the one `render` writes. Two spellings of
    /// one rule is how they drift apart.
    #[test]
    fn the_user_turn_opener_is_what_render_emits() {
        let open = Codec.user_turn_open();
        assert!(open.starts_with(TURN_OPEN), "{open} vs {TURN_OPEN}");
        assert!(open.ends_with("user"));
        let rendered =
            Codec.render(&[json!({"role": "user", "content": "hi"})], &[], false, "");
        assert!(
            rendered.contains(open),
            "render did not emit {open}: {rendered}"
        );
    }

    #[test]
    fn render_is_chatml_and_the_generation_prompt_opens_the_channel() {
        let out = Codec.render(
            &[
                json!({"role": "system", "content": "be brief"}),
                json!({"role": "user", "content": "hi"}),
            ],
            &[],
            true,
            "<BOS>",
        );
        assert_eq!(
            out,
            "<|im_start|>system\nbe brief<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n"
        );
        // The BOS spelling is accepted and not written: this format does not begin with it.
        assert!(!out.contains("<BOS>"));
    }

    #[test]
    fn the_split_starts_inside_the_channel_when_the_prompt_left_it_open() {
        let (reasoning, visible) =
            Codec.split_channels("weighing it\n</think>\n\nthe answer", true);
        assert_eq!(reasoning, "weighing it\n");
        assert_eq!(visible, "\n\nthe answer");
    }

    /// THE THINKING-OFF BRANCH. This model's template ends the prompt with
    /// `<think>\n\n</think>\n\n` when `enable_thinking` is false, so generation begins
    /// OUTSIDE the channel and every byte is visible. Hardcoding `true` here returned an
    /// empty `content` for every such request, with the whole answer filed as reasoning.
    #[test]
    fn the_split_starts_outside_the_channel_when_the_prompt_closed_it() {
        let (reasoning, visible) =
            Codec.split_channels("the answer, with no channel", false);
        assert_eq!(reasoning, "");
        assert_eq!(visible, "the answer, with no channel");
    }

    /// And which one it is comes from the PROMPT, not from a flag we carried along.
    #[test]
    fn the_prompt_says_which_side_of_the_channel_generation_begins_on() {
        assert!(prompt_ends_in_reasoning("<|im_start|>assistant\n<think>\n"));
        assert!(!prompt_ends_in_reasoning(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        assert!(!prompt_ends_in_reasoning("<|im_start|>assistant\n"));
    }

    #[test]
    fn a_tool_call_is_xml_and_its_values_may_span_lines() {
        let text = "before\n<tool_call>\n<function=send>\n<parameter=to>\nada\n</parameter>\n\
                    <parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>after";
        let (visible, calls) = Codec.parse_tool_calls(text);
        assert_eq!(visible, "before\nafter");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "send");
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["to"], "ada");
        assert_eq!(args["body"], "line one\nline two");
    }

    /// An unclosed block is either STILL ARRIVING or it is prose. The parse always SHOWS
    /// it (never silently eats text the model wrote); holding it while it might still close
    /// is `unsettled_in_visible`, one stage earlier -- never shown and then retracted, which
    /// SSE cannot do.
    #[test]
    fn an_unclosed_tool_call_is_shown_by_the_parse_and_held_by_the_cut() {
        let text = "text <tool_call>\n<function=x>\n";
        assert_eq!(Codec.parse_tool_calls(text), (text.to_string(), Vec::new()));
        fn cut(t: &str) -> &str {
            &t[..t.len() - Codec.unsettled_in_visible(t)]
        }
        assert_eq!(
            Codec.parse_tool_calls(cut(text)),
            ("text ".to_string(), Vec::new())
        );
        // A suffix that could still grow into the opener is held too.
        assert_eq!(
            Codec.parse_tool_calls(cut("text <tool_c")),
            ("text ".to_string(), Vec::new())
        );
    }

    /// Streaming parses a PREFIX: the visible text may only grow, and must never carry
    /// call markup a client would render and then be asked to un-render.
    #[test]
    fn streaming_tool_parse_never_retracts_and_never_leaks() {
        let text = "before<tool_call>\n<function=weather>\n                    <parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>after";
        let close_at = text.find(CALL_CLOSE).unwrap() + CALL_CLOSE.len();
        let (mut prior_visible, mut prior_calls) = (String::new(), 0_usize);
        for end in 1..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let seen = &text[..end];
            let settled = &seen[..seen.len() - Codec.unsettled_in_visible(seen)];
            let (visible, calls) = parse_tool_calls(settled);
            assert!(
                visible.starts_with(&prior_visible),
                "visible retracted at {end}"
            );
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

    /// The RAW cut is about the CHANNEL marker and nothing else.
    #[test]
    fn the_raw_cut_covers_a_marker_split_across_two_deltas() {
        use crate::chat::ChatCodec;
        // "</thi" is five bytes and a prefix of "</think>", so five are undecided.
        assert_eq!(Codec.unsettled_in_raw("answer</thi"), 5);
        assert_eq!(Codec.unsettled_in_raw("plain text"), 0);
        // A CALL OPENER IS NOT THIS CUT'S BUSINESS: inside `<think>` it is prose that never
        // closes, and cutting here would hold the reasoning -- and the answer after it --
        // to the end of generation.
        assert_eq!(Codec.unsettled_in_raw("a<tool_call>\n<function=x>"), 0);
        assert!(Codec.channel_state_after("a<think>b", false));
        assert!(!Codec.channel_state_after("a</think>b", true));
        assert!(Codec.channel_state_after("no marker", true));
    }

    /// The VISIBLE cut is about the CALL block, on text the split has already classified.
    #[test]
    fn the_visible_cut_holds_an_open_call() {
        use crate::chat::ChatCodec;
        assert_eq!(Codec.unsettled_in_visible("a<tool_call>\n<function=x>"), 24);
        assert_eq!(Codec.unsettled_in_visible("plain text"), 0);
        assert_eq!(Codec.unsettled_in_visible("answer<tool_c"), 7);
    }

    /// THE CASE THE ONE-CUT FORM BROKE: a `<tool_call>` written inside the thinking is
    /// prose. Split first and the call cut never sees it, so `</think>` and the visible
    /// answer stream normally instead of being held to the end of generation.
    #[test]
    fn a_call_marker_inside_reasoning_does_not_stall_the_stream() {
        use crate::chat::ChatCodec;
        let text = "<think>I could <tool_call> here</think>the answer";
        assert_eq!(Codec.unsettled_in_raw(text), 0, "raw text is fully settled");
        let (r, v) = Codec.split_channels(text, false);
        assert_eq!(v, "the answer");
        assert!(r.contains(CALL_OPEN), "it stays in the reasoning, as prose");
        assert_eq!(Codec.unsettled_in_visible(&v), 0);
    }

    /// The template this model ships must carry every marker the parser needs.
    #[test]
    fn the_shipped_template_carries_every_marker_the_parser_needs() {
        let src = "<|im_start|> ... <think> </think> ... <tool_call> </tool_call>";
        assert!(markers_missing_from(src).is_empty());
        assert_eq!(
            markers_missing_from("<|im_start|> only"),
            vec![THINK_CLOSE, CALL_OPEN, CALL_CLOSE]
        );
    }
}
