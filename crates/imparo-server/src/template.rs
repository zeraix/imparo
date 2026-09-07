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

use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value, context};

pub struct Template {
    env: Environment<'static>,
    /// The template treats `tool_calls[].function.arguments` as a mapping rather than the
    /// JSON string the OpenAI wire format carries: gemma4 spells a mapping as
    /// `{city:<|"|>Oslo<|"|>}`, LFM2.5 raises on a string. llama.cpp decides this by
    /// watching whether the template reads a key inside the arguments object; the
    /// observable equivalent used here: render one tool call with the arguments as an
    /// object and again as the same JSON text, and parse strings into objects when the
    /// two renders differ or the string form fails to render.
    requires_object_arguments: bool,
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
        // `{% generation %}...{% endgeneration %}` marks assistant spans for training-time
        // token masks (HF transformers); it renders as nothing. minijinja rejects the
        // unknown statement, which made LFM2.5's embedded template unparseable ("unknown
        // statement generation (in chat:86)") and sent every LFM2 render to the built-in
        // renderer. llama.cpp's minja accepts the tag; strip it before compiling.
        let src = strip_generation_tags(src);
        let src = src.as_str();
        // Real templates are written against Python's Jinja: `message.get('reasoning')`,
        // `args.items()`, `text.split(tag)`, `s.endswith(tag)`. minijinja has no such
        // methods on plain values; without this every gemma4 and LFM2 render failed on
        // `.get` and fell back to the hand-written renderer (which always writes the
        // thinking token), so the model's own template was never what the model saw.
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        // `tojson` the way Python's Jinja and llama.cpp's minja spell it: `", "` and `": "`
        // separators, keys in the order the request gave them (preserve_order on both
        // serde_json and minijinja), `indent=N` honoured. minijinja's own filter prints
        // compact JSON, which is not what the model saw in training or what the reference
        // server renders (template_agree with_tools: theirs `{"type": "function", ...`).
        env.add_filter("tojson", py_tojson);
        env.add_template_owned("chat".to_string(), src.to_string())
            .map_err(|e| format!("chat template does not parse: {e}"))?;
        let mut t = Self {
            env,
            requires_object_arguments: false,
        };
        let probe = |arguments: serde_json::Value| {
            let msgs = vec![
                serde_json::json!({"role": "user", "content": "Hey"}),
                serde_json::json!({"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "ipython", "arguments": arguments}}]}),
            ];
            let tools = vec![serde_json::json!({"type": "function", "function": {
                "name": "ipython", "description": "Runs code.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                               "required": ["code"]}}})];
            t.render_as_given(&msgs, &tools, false, "", &serde_json::Map::new())
        };
        let as_object =
            probe(serde_json::json!({"argument_needle": "print('Hello World!')"}));
        let as_text = probe(serde_json::json!(
            "{\"argument_needle\": \"print('Hello World!')\"}"
        ));
        t.requires_object_arguments = match (as_object, as_text) {
            (Ok(o), Ok(s)) => o != s,
            (Ok(_), Err(_)) => true,
            (Err(_), _) => false,
        };
        Ok(t)
    }

    pub fn requires_object_arguments(&self) -> bool {
        self.requires_object_arguments
    }

    /// `kwargs` are the request's `chat_template_kwargs` (llama.cpp's field): extra
    /// template variables such as `enable_thinking`, laid under the fixed ones.
    pub fn render(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        add_generation_prompt: bool,
        bos: &str,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, String> {
        if self.requires_object_arguments {
            let msgs = with_object_arguments(messages);
            return self.render_as_given(
                &msgs,
                tools,
                add_generation_prompt,
                bos,
                kwargs,
            );
        }
        self.render_as_given(messages, tools, add_generation_prompt, bos, kwargs)
    }

    fn render_as_given(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        add_generation_prompt: bool,
        bos: &str,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String, String> {
        let tmpl = self.env.get_template("chat").map_err(|e| e.to_string())?;
        let msgs = Value::from_serialize(messages);
        let tls = Value::from_serialize(tools);
        let extra = Value::from_serialize(kwargs);
        let ctx = if tools.is_empty() {
            context! { messages => msgs, add_generation_prompt => add_generation_prompt,
            bos_token => bos, ..extra }
        } else {
            context! { messages => msgs, tools => tls,
            add_generation_prompt => add_generation_prompt, bos_token => bos, ..extra }
        };
        tmpl.render(ctx)
            .map_err(|e| format!("chat template render failed: {e}"))
    }
}

/// A copy of `messages` with every `tool_calls[].function.arguments` that is a JSON
/// string parsed into the value it encodes; anything that does not parse is left as is.
fn with_object_arguments(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut out = messages.to_vec();
    for m in &mut out {
        let Some(calls) = m
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        for c in calls {
            let Some(args) = c.get_mut("function").and_then(|f| f.get_mut("arguments"))
            else {
                continue;
            };
            if let Some(text) = args.as_str() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
                    *args = v;
                }
            }
        }
    }
    out
}

/// Python `json.dumps` formatting (see the filter registration for why).
fn py_tojson(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let json = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    let mut out = Vec::new();
    let r = match indent {
        Some(n) => {
            let pad = " ".repeat(n);
            let f = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut out, f);
            serde::Serialize::serialize(&json, &mut ser)
        }
        None => {
            let mut ser =
                serde_json::Serializer::with_formatter(&mut out, PySeparators {});
            serde::Serialize::serialize(&json, &mut ser)
        }
    };
    r.map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    String::from_utf8(out)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))
}

/// serde_json's compact formatter with Python's default separators `", "` and `": "`.
struct PySeparators {}

impl serde_json::ser::Formatter for PySeparators {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
    ) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// Removes `{% generation %}` / `{% endgeneration %}` (any `-` trim spelling) from a
/// template source; they mark spans, they render nothing.
fn strip_generation_tags(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find("{%") {
        let Some(close) = rest[i..].find("%}") else {
            break;
        };
        let tag = &rest[i + 2..i + close];
        let word = tag
            .trim()
            .trim_start_matches('-')
            .trim_end_matches('-')
            .trim();
        if word == "generation" || word == "endgeneration" {
            out.push_str(&rest[..i]);
        } else {
            out.push_str(&rest[..i + close + 2]);
        }
        rest = &rest[i + close + 2..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_tags_are_stripped_and_nothing_else() {
        let src = "a{%- generation -%}b{% endgeneration %}c{% if x %}d{% endif %}";
        assert_eq!(strip_generation_tags(src), "abc{% if x %}d{% endif %}");
        assert_eq!(strip_generation_tags("no tags"), "no tags");
    }

    #[test]
    fn python_methods_render() {
        let t = Template::compile(
            "{{ messages[0].get('reasoning', 'none') }}|{{ 'a</b>c'.split('</b>')[-1] }}|\
             {%- for k, v in tools[0].items() %}{{ k }}={{ v }}{% endfor %}",
        )
        .unwrap();
        let msgs = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let tools = vec![serde_json::json!({"n": 1})];
        let out = t
            .render(&msgs, &tools, false, "", &serde_json::Map::new())
            .unwrap();
        assert_eq!(out, "none|c|n=1");
    }

    #[test]
    fn tojson_is_python_shaped() {
        let t = Template::compile(
            "{{ tools[0] | tojson }}|{{ tools[0].b | tojson(indent=2) }}",
        )
        .unwrap();
        let tools = vec![
            serde_json::from_str(
                r#"{"type": "function", "b": {"z": [1, 2], "a": "x<y"}}"#,
            )
            .unwrap(),
        ];
        let out = t
            .render(&[], &tools, false, "", &serde_json::Map::new())
            .unwrap();
        assert_eq!(
            out,
            "{\"type\": \"function\", \"b\": {\"z\": [1, 2], \"a\": \"x<y\"}}|{\n  \"z\": [\n    1,\n    2\n  ],\n  \"a\": \"x<y\"\n}"
        );
    }

    #[test]
    fn string_arguments_become_objects_only_when_the_template_needs_them() {
        // A template that indexes the arguments: strings fail the probe.
        let needs = Template::compile(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             {% for k, v in c.function.arguments.items() %}{{ k }}={{ v }}{% endfor %}\
             {% endfor %}{% endif %}{% endfor %}",
        )
        .unwrap();
        assert!(needs.requires_object_arguments());
        // A template that serialises the arguments: objects and their JSON text differ
        // (a string would be quoted), so objects are what it wants -- Llama-3 style.
        let serialises = Template::compile(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             {{ c.function.arguments | tojson }}{% endfor %}{% endif %}{% endfor %}",
        )
        .unwrap();
        assert!(serialises.requires_object_arguments());
        // A template that never looks at the arguments: strings stay strings.
        let ignores = Template::compile(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             {{ c.function.name }}{% endfor %}{% endif %}{% endfor %}",
        )
        .unwrap();
        assert!(!ignores.requires_object_arguments());
        let msgs = vec![
            serde_json::json!({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": "{\"city\": \"Oslo\"}"}}]}),
        ];
        assert_eq!(
            needs
                .render(&msgs, &[], false, "", &serde_json::Map::new())
                .unwrap(),
            "city=Oslo"
        );
        assert_eq!(
            serialises
                .render(&msgs, &[], false, "", &serde_json::Map::new())
                .unwrap(),
            "{\"city\": \"Oslo\"}"
        );
    }

    #[test]
    fn kwargs_reach_the_template() {
        let t = Template::compile("{{ enable_thinking | default(false) }}").unwrap();
        let mut kw = serde_json::Map::new();
        kw.insert("enable_thinking".into(), serde_json::Value::Bool(true));
        // pycompat prints booleans the way Python's Jinja does.
        assert_eq!(t.render(&[], &[], false, "", &kw).unwrap(), "True");
        assert_eq!(
            t.render(&[], &[], false, "", &serde_json::Map::new())
                .unwrap(),
            "False"
        );
    }
}
