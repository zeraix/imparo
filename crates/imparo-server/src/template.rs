//! Jinja chat-template rendering (task #15).
//!
//! Source order: (1) a user-supplied template file (`--chat-template-file`, llama.cpp's
//! flag), (2) the template embedded in the GGUF (`tokenizer.chat_template`), (3) the
//! hand-coded gemma4 renderer in `chat.rs` as the fallback for template-less GGUFs --
//! kept because it is verified byte-equal for that model, and ONLY as a fallback: the
//! hand-coding approach is what produced the 7-missing-tokens bug (task #7).
//!
//! Correctness gate: rendered prompts byte-equal to the reference fork's
//! /apply-template for the same messages (dev_harness/template_agree.py, using
//! IMPARO_DUMP_PROMPT).

use minijinja::{Environment, Error, ErrorKind, Value, context};

pub struct Template {
    env: Environment<'static>,
}

impl Template {
    /// Compiles a template. Returns Err with minijinja's message when it does not parse;
    /// the caller decides whether that is fatal (a user-supplied file: yes) or a
    /// fall-back-to-hand-coded (an embedded template we cannot render yet).
    pub fn compile(src: &str) -> Result<Self, String> {
        let mut env = Environment::new();
        // The shims llama.cpp's minja provides and real-world templates rely on.
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String {
            // Deterministic renders matter more here than wall-clock dates; templates
            // use this for "today's date" headers, which this engine renders empty the
            // way llama.cpp does when no date is supplied.
            let _ = fmt;
            String::new()
        });
        env.set_trim_blocks(false);
        env.set_lstrip_blocks(false);
        env.add_template_owned("chat".to_string(), src.to_string())
            .map_err(|e| format!("chat template does not parse: {e}"))?;
        Ok(Self { env })
    }

    pub fn render(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        add_generation_prompt: bool,
        bos: &str,
    ) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let msgs = Value::from_serialize(messages);
        let tls = Value::from_serialize(tools);
        let ctx = if tools.is_empty() {
            context! { messages => msgs, add_generation_prompt => add_generation_prompt,
            bos_token => bos }
        } else {
            context! { messages => msgs, tools => tls,
            add_generation_prompt => add_generation_prompt, bos_token => bos }
        };
        tmpl.render(ctx)
            .map_err(|e| format!("chat template render failed: {e}"))
    }
}
