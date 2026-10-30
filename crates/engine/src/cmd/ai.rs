use serde_json::Value;

use super::{CommandSpec, always, bad, cmd, opt_usize_param, str_param};
use crate::ai::{PlanRequest, plan as plan_with_ai};
use crate::{EngineError, Result, Session};

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        query "ai.plan",
        "Plan with AI",
        [],
        None,
        "{prompt, slides?, theme?, model?} → {slides: [{title, bullets, layout?}]}",
        always,
        plan_cmd
    )]
}

fn plan_cmd(_s: &mut Session, p: &Value) -> Result<Value> {
    let prompt = str_param(p, "prompt").ok_or_else(|| bad("ai.plan", "missing `prompt`"))?;
    let slides = opt_usize_param(p, "slides", "ai.plan")?;
    let theme = str_param(p, "theme");
    let model = str_param(p, "model");
    let result = plan_with_ai(PlanRequest { prompt, slides, theme, model })?;
    serde_json::to_value(result).map_err(|e| EngineError::Other(format!("could not encode the plan: {e}")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use serde_json::json;

    #[test]
    fn ai_plan_is_a_query_and_needs_a_prompt() {
        let spec = specs().into_iter().find(|s| s.id == "ai.plan").unwrap();
        assert!(!spec.journal);
        assert!(!spec.undoable);
        let mut session = Session::with_new();
        let err = (spec.run)(&mut session, &json!({})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }));
    }

    #[test]
    fn a_blank_prompt_is_rejected() {
        let mut session = Session::with_new();
        let err = plan_cmd(&mut session, &json!({"prompt": "  "})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }));
    }

    #[test]
    fn a_negative_slide_count_is_rejected() {
        let mut session = Session::with_new();
        let err = plan_cmd(&mut session, &json!({"prompt": "deck", "slides": -1})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }));
    }
}
