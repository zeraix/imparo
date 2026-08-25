//! How a model's prompts are rendered and its output is read back.
//!
//! Two halves of one contract, which is why they are one trait: a template that renders
//! turns the parser cannot find is worse than no template, because it produces plausible
//! text with the reasoning and tool calls silently left inside it.
//!
//! Keyed on the ARCHITECTURE and reachable from the plan alone, like [`crate::load`]. The
//! server sniffs a chat template for the parser's markers before it builds a model, so a
//! codec that needed one would be reachable too late.

use serde_json::Value;

/// Which side of a split a streamed fragment came from.
///
/// The holdback rule differs per side for some models and not for others -- gemma4 waits
/// on a different marker depending on which channel it is emitting; LFM2 waits on either
/// of its two whichever side it is on. That is a fact about each format, so the format
/// answers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// The model's reasoning, which a client shows separately or not at all.
    Reasoning,
    /// What the user sees.
    Visible,
}

/// One architecture's chat format.
///
/// Implemented by a unit type per model; the bodies live in that model's
/// `chat_format` module beside the markers they use.
pub trait ChatCodec: Sync {
    /// Marker spellings this codec's PARSER needs that `template_src` never emits.
    ///
    /// A template for a different model renders fine while reasoning and tool-call
    /// extraction match nothing at all, so the server says so at load rather than
    /// letting every response come back with its reasoning inlined.
    fn markers_missing_from(&self, template_src: &str) -> Vec<&'static str>;

    /// Renders a prompt without a template: for a GGUF that carries none, or one whose
    /// template failed to compile.
    ///
    /// `bos` is the BOS token's SPELLING, from tokenizer metadata. A codec whose format
    /// starts with it uses it; one whose tokenizer prepends it instead ignores it. No
    /// codec knows a token id.
    fn render(
        &self,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: bool,
        bos: &str,
    ) -> String;

    /// Splits generated text into (reasoning, visible).
    fn split_channels(&self, text: &str) -> (String, String);

    /// Trailing bytes of a streamed fragment that must NOT be emitted yet, because they
    /// may be the start of a marker that completes in the next delta.
    ///
    /// Without it a marker straddling two deltas is emitted as visible text and then
    /// silently reclassified, which the client has already shown.
    fn holdback(&self, text: &str, side: Channel) -> usize;

    /// Splits the visible text into (text, calls), each call a (name, json-args) pair.
    fn parse_tool_calls(&self, text: &str) -> (String, Vec<(String, String)>);

    /// The exact text that OPENS a user turn in this format.
    ///
    /// The server needs the token where the last user message starts, to put a
    /// checkpoint there. It used to find it by re-rendering the conversation minus
    /// that message and tokenizing the result -- a second full tokenization of the
    /// whole prompt, measured at 4.5-6.0 ms on 2.8k tokens and growing linearly,
    /// paid on every request. Tokenized once at load, this turns the same question
    /// into a scan over token ids the prompt already produced.
    ///
    /// Without the trailing newline on purpose: what follows a turn opener is
    /// content, and a tokenizer is free to merge a newline with whatever comes
    /// after it, so the shortest stable spelling is the one to match.
    fn user_turn_open(&self) -> &'static str;
}

/// The chat codec for a plan's architecture.
///
/// The one place an architecture name maps to a codec, and the companion to
/// [`crate::load`]: that turns a description into something that runs, this turns it into
/// something that can be talked to.
///
/// # Errors
/// When the architecture has no codec.
pub fn codec(plan: &crate::ModelPlan) -> Result<&'static dyn ChatCodec, String> {
    match plan.config.architecture.as_str() {
        "gemma4" => Ok(&crate::gemma4::chat_format::Codec),
        "lfm2" => Ok(&crate::lfm2::chat_format::Codec),
        other => Err(format!("no chat codec for architecture '{other}'")),
    }
}
