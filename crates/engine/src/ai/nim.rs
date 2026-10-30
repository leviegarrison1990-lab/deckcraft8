use std::time::Duration;

use serde_json::{Value, json};

use super::plan::{Plan, PlanRequest};
use crate::cmd::bad;
use crate::{EngineError, Result};

const ENDPOINT: &str = "https://integrate.api.nvidia.com/v1/chat/completions";
// TODO(phase2): confirm against /v1/models once a key is available.
const DEFAULT_MODEL: &str = "meta/llama-3.1-8b-instruct";
const MAX_PROMPT: usize = 32 * 1024;
const MAX_REPORTED: usize = 500;
const TIMEOUT: Duration = Duration::from_secs(30);

const SYSTEM_PROMPT: &str = r#"You plan presentation slide decks. Return only a JSON object with this exact shape:
{"slides":[{"title":"...","bullets":["..."],"layout":null}]}
Rules:
- "title" is a short slide title.
- "bullets" is an array of 2-6 short points; use [] for a title-only slide.
- "layout" is null or one of: "title", "titleContent", "section", "twoContent", "picture".
- Do not add prose, markdown fences, comments, or extra keys.
Example:
{"slides":[{"title":"Why now","bullets":["Market shift","Customer pain","Our opening"],"layout":"titleContent"}]}"#;

pub(super) fn plan(req: PlanRequest<'_>) -> Result<Plan> {
    validate(&req)?;
    let key = api_key()?;
    let body = build_body(&req).to_string();
    let tls = ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build();
    let agent = ureq::Agent::config_builder().tls_config(tls).timeout_global(Some(TIMEOUT)).http_status_as_error(false).build().new_agent();
    let mut response = agent
        .post(ENDPOINT)
        .header("Authorization", format!("Bearer {key}"))
        .header("Accept", "application/json")
        .header("Content-Type", "application/json")
        .send(body)
        .map_err(|e| EngineError::Other(format!("NVIDIA request failed: {e}")))?;
    let status = response.status().as_u16();
    let raw = response.body_mut().read_to_string().map_err(|e| EngineError::Other(format!("NVIDIA response could not be read: {e}")))?;
    if !(200..300).contains(&status) {
        return Err(EngineError::Other(format!("NVIDIA HTTP {status}: {}", snippet(&raw))));
    }
    parse_response(&raw)
}

fn validate(req: &PlanRequest<'_>) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(bad("ai.plan", "`prompt` is empty"));
    }
    if req.prompt.len() > MAX_PROMPT {
        return Err(bad("ai.plan", format!("`prompt` is {} bytes; the limit is {MAX_PROMPT}", req.prompt.len())));
    }
    if req.slides == Some(0) {
        return Err(bad("ai.plan", "`slides` must be at least 1"));
    }
    Ok(())
}

fn api_key() -> Result<String> {
    let primary = std::env::var("DECKCRAFT_NVIDIA_API_KEY").ok();
    let fallback = std::env::var("NVIDIA_API_KEY").ok();
    pick_key(primary, fallback).ok_or_else(|| EngineError::Other("NVIDIA API key not set; set DECKCRAFT_NVIDIA_API_KEY or NVIDIA_API_KEY".into()))
}

fn pick_key(primary: Option<String>, fallback: Option<String>) -> Option<String> {
    primary.into_iter().chain(fallback).map(|key| key.trim().to_string()).find(|key| !key.is_empty())
}

fn build_body(req: &PlanRequest<'_>) -> Value {
    let mut ask = req.prompt.to_string();
    if let Some(n) = req.slides {
        ask.push_str("\n\nReturn exactly ");
        ask.push_str(&n.to_string());
        ask.push_str(" slides.");
    }
    if let Some(theme) = req.theme {
        ask.push_str("\nUse the theme/visual direction: ");
        ask.push_str(theme);
        ask.push('.');
    }
    json!({
        "model": req.model.unwrap_or(DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ask}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    })
}

fn parse_response(raw: &str) -> Result<Plan> {
    let envelope: Value =
        serde_json::from_str(raw).map_err(|e| EngineError::Other(format!("NVIDIA returned invalid JSON: {e}; {}", snippet(raw))))?;
    let content = envelope
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::Other(format!("NVIDIA response has no choices[0].message.content; {}", snippet(raw))))?;
    parse_content(content).map_err(EngineError::Other)
}

fn parse_content(content: &str) -> std::result::Result<Plan, String> {
    let text = strip_fence(content).trim();
    if let Ok(plan) = serde_json::from_str::<Plan>(text) {
        return Ok(plan);
    }
    let Some(start) = text.find('{') else {
        return Err(format!("model content is not JSON: {}", snippet(text)));
    };
    let Some(end) = text.rfind('}') else {
        return Err(format!("model content is not JSON: {}", snippet(text)));
    };
    if end <= start {
        return Err(format!("model content is not JSON: {}", snippet(text)));
    }
    serde_json::from_str::<Plan>(&text[start..=end]).map_err(|e| format!("model content is not a slide plan: {e}; {}", snippet(text)))
}

fn strip_fence(content: &str) -> &str {
    let trimmed = content.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.strip_prefix("\r\n").or_else(|| rest.strip_prefix('\n')).unwrap_or(rest);
    rest.trim_end().strip_suffix("```").unwrap_or(rest).trim()
}

fn snippet(text: &str) -> String {
    let mut out: String = text.chars().take(MAX_REPORTED).collect();
    if text.chars().count() > MAX_REPORTED {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use serde_json::json;

    #[test]
    fn key_prefers_the_deckcraft_name_then_the_nvidia_one() {
        assert_eq!(pick_key(Some("a".into()), Some("b".into())).as_deref(), Some("a"));
        assert_eq!(pick_key(Some("  ".into()), Some("b".into())).as_deref(), Some("b"));
        assert_eq!(pick_key(None, Some(" b ".into())).as_deref(), Some("b"));
        assert_eq!(pick_key(None, None), None);
    }

    #[test]
    fn prompt_is_capped_and_must_not_be_blank() {
        let empty = PlanRequest { prompt: "  ", slides: None, theme: None, model: None };
        assert!(matches!(validate(&empty), Err(EngineError::BadParams { .. })));
        let long = "x".repeat(MAX_PROMPT + 1);
        let too_long = PlanRequest { prompt: &long, slides: None, theme: None, model: None };
        assert!(matches!(validate(&too_long), Err(EngineError::BadParams { .. })));
        let zero = PlanRequest { prompt: "deck", slides: Some(0), theme: None, model: None };
        assert!(matches!(validate(&zero), Err(EngineError::BadParams { .. })));
    }

    #[test]
    fn body_uses_the_default_model_and_carries_the_prompt() {
        let req = PlanRequest { prompt: "a talk about rust", slides: Some(3), theme: Some("calm"), model: None };
        let body = build_body(&req);
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(
            body["messages"][1]["content"].as_str().unwrap(),
            "a talk about rust\n\nReturn exactly 3 slides.\nUse the theme/visual direction: calm."
        );
    }

    #[test]
    fn body_honours_an_explicit_model() {
        let req = PlanRequest { prompt: "x", slides: None, theme: None, model: Some("meta/llama-3.1-70b-instruct") };
        assert_eq!(build_body(&req)["model"], "meta/llama-3.1-70b-instruct");
    }

    #[test]
    fn parses_a_plain_envelope() {
        let content = json!({"slides":[{"title":"One","bullets":["a"],"layout":"title"}]}).to_string();
        let raw = json!({"choices":[{"message":{"content": content}}]}).to_string();
        let plan = parse_response(&raw).unwrap();
        assert_eq!(plan.slides.len(), 1);
        assert_eq!(plan.slides[0].title, "One");
        assert_eq!(plan.slides[0].layout.as_deref(), Some("title"));
    }

    #[test]
    fn parses_content_wrapped_in_a_markdown_fence() {
        let content = "```json\n{\"slides\":[{\"title\":\"Two\",\"bullets\":[]}]}\n```";
        let plan = parse_content(content).unwrap();
        assert_eq!(plan.slides[0].title, "Two");
    }

    #[test]
    fn tolerates_prose_around_the_json() {
        let content = "Here is the plan:\n{\"slides\":[{\"title\":\"Three\"}]}\nDone.";
        let plan = parse_content(content).unwrap();
        assert_eq!(plan.slides[0].title, "Three");
    }

    #[test]
    fn malformed_content_is_reported_with_its_start() {
        let err = parse_content("not json at all").unwrap_err();
        assert!(err.contains("not JSON"));
        assert!(err.contains("not json at all"));
    }

    #[test]
    fn a_missing_content_field_is_an_error() {
        let raw = json!({"choices":[{"message":{}}]}).to_string();
        assert!(parse_response(&raw).is_err());
    }

    #[test]
    #[ignore = "needs DECKCRAFT_NVIDIA_API_KEY or NVIDIA_API_KEY"]
    fn live_plan_reaches_nim() {
        if api_key().is_err() {
            return;
        }
        let plan = plan(PlanRequest { prompt: "Three slides on why small teams ship faster", slides: Some(3), theme: None, model: None }).unwrap();
        assert!(!plan.slides.is_empty());
        assert!(plan.slides.iter().all(|slide| !slide.title.trim().is_empty()));
    }
}
