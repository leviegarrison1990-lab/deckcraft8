use serde_json::{Value, json};

use super::{CommandSpec, always, bad, cmd, has_doc, opt_usize_param, str_param};
use crate::ai::{PlanRequest, TemplatePlanRequest, plan as plan_with_ai, plan_template as plan_template_with_ai};
use crate::{EngineError, Result, Session};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            query "ai.plan",
            "Plan with AI",
            [],
            None,
            "{prompt, slides?, theme?, model?} → {slides: [{title, bullets, layout?}]}",
            always,
            plan_cmd
        ),
        cmd!(
            query "ai.templateScan",
            "Scan PPTX Template",
            [],
            None,
            "{pptx: base64, slides?: [int]} -> {slides: [SlideScan], warnings: [string]}",
            always,
            template_scan
        ),
        cmd!(
            query "ai.planTemplate",
            "Plan Template",
            [],
            None,
            "{content, slidesFromScan: [SlideScan], model?} -> {slides: [{index, keep_slot_ids: [string], fills?: [{slot_id, heading?, body?}]}]}",
            always,
            plan_template_cmd
        ),
        cmd!(
            "ai.applyTemplatePlan",
            "Apply Template Plan",
            [],
            None,
            "{plan: {slides: [{index, keep_slot_ids: [string], realign?: auto|none, prune_orphan_connectors?: bool = true}]}} -> {slides: [{index, deleted, moved, rows_realigned}]}",
            has_doc,
            apply_cmd
        ),
    ]
}

fn plan_cmd(_s: &mut Session, p: &Value) -> Result<Value> {
    let prompt = str_param(p, "prompt").ok_or_else(|| bad("ai.plan", "missing `prompt`"))?;
    let slides = opt_usize_param(p, "slides", "ai.plan")?;
    let theme = str_param(p, "theme");
    let model = str_param(p, "model");
    let result = plan_with_ai(PlanRequest { prompt, slides, theme, model })?;
    serde_json::to_value(result).map_err(|e| EngineError::Other(format!("could not encode the plan: {e}")))
}

fn plan_template_cmd(_s: &mut Session, p: &Value) -> Result<Value> {
    let content = str_param(p, "content").ok_or_else(|| bad("ai.planTemplate", "missing `content`"))?;
    let scans_val =
        p.get("slidesFromScan").or_else(|| p.get("slides_from_scan")).ok_or_else(|| bad("ai.planTemplate", "missing `slidesFromScan`"))?;
    let scans_val = scans_val.get("slides").unwrap_or(scans_val);
    let scans: Vec<crate::ai::SlideScan> =
        serde_json::from_value(scans_val.clone()).map_err(|e| bad("ai.planTemplate", format!("bad `slidesFromScan`: {e}")))?;
    let model = str_param(p, "model");
    let result = plan_template_with_ai(TemplatePlanRequest { content, slides_from_scan: &scans, model })?;
    serde_json::to_value(result).map_err(|e| EngineError::Other(format!("could not encode the template plan: {e}")))
}

fn template_scan(_s: &mut Session, p: &Value) -> Result<Value> {
    let b64 = str_param(p, "pptx").ok_or_else(|| bad("ai.templateScan", "missing `pptx`"))?;
    let bytes = crate::cmd::base64_decode(b64).ok_or_else(|| bad("ai.templateScan", "pptx is not valid base64"))?;
    let (doc, warnings) = deckcraft_pptx::import_checked(&bytes).map_err(|e| EngineError::Other(format!("PPTX import failed: {e}")))?;
    let scans = crate::ai::scan_presentation(&doc);
    let filter: Option<Vec<usize>> =
        p.get("slides").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect());
    let out: Vec<_> = scans.into_iter().filter(|s| filter.as_ref().is_none_or(|f| f.contains(&s.index))).collect();
    Ok(json!({"slides": out, "warnings": warnings}))
}

fn apply_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let plan_val = p.get("plan").cloned().ok_or_else(|| bad("ai.applyTemplatePlan", "missing `plan`"))?;
    let plan: crate::ai::ApplyPlan = serde_json::from_value(plan_val).map_err(|e| bad("ai.applyTemplatePlan", format!("bad plan: {e}")))?;
    s.edit(|doc, _sel| {
        let res = crate::ai::apply::apply_plan(doc, &plan)?;
        serde_json::to_value(res).map_err(|e| EngineError::Other(format!("serialize apply result: {e}")))
    })
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

    #[test]
    fn ai_plan_template_is_a_query_and_needs_input() {
        let spec = specs().into_iter().find(|s| s.id == "ai.planTemplate").unwrap();
        assert!(!spec.journal);
        assert!(!spec.undoable);
        let mut session = Session::with_new();
        assert!((spec.enabled)(&session).is_ok());
        for params in [json!({}), json!({"content": "# Slide 1: A"}), json!({"slidesFromScan": []})] {
            let err = (spec.run)(&mut session, &params).unwrap_err();
            assert!(matches!(err, EngineError::BadParams { .. }));
        }
    }

    #[test]
    fn plan_template_rejects_bad_scan_input_before_network() {
        let mut session = Session::with_new();
        let err = plan_template_cmd(&mut session, &json!({"content": "# Slide 1: A", "slidesFromScan": [{}]})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }));
        let err = plan_template_cmd(&mut session, &json!({"content": "# Slide 1: A", "slidesFromScan": []})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }));
    }

    use std::sync::Arc;

    use deckcraft_model::{ColorRef, Fill, LayoutId, Presentation, Rgba, Shape, ShapeId, ShapeKind, Slide, SlideId, Xfrm};

    fn card_deck(slides: usize) -> Presentation {
        let mut doc = Presentation::default();
        doc.slides.clear();
        let layout = doc.masters.first().and_then(|m| m.layouts.first()).map(|l| l.id).unwrap_or(LayoutId(0));
        for i in 0..slides {
            let base = 10 + i as u32 * 10;
            let card = Shape {
                id: ShapeId(base),
                xfrm: Some(Xfrm::new(100.0, 100.0, 500.0, 300.0)),
                kind: ShapeKind::Group {
                    children: vec![Shape {
                        id: ShapeId(base + 1),
                        xfrm: Some(Xfrm::new(0.0, 0.0, 500.0, 300.0)),
                        fill: Some(Fill::solid(ColorRef::rgb(Rgba::rgb(80, 60, 120)))),
                        ..Default::default()
                    }],
                    child: Xfrm::new(0.0, 0.0, 500.0, 300.0),
                },
                ..Default::default()
            };
            let id = SlideId(300 + i as u32);
            doc.slides.push(Arc::new(Slide { id, layout, shapes: vec![card], ..Default::default() }));
        }
        doc
    }

    fn card_pptx(slides: usize) -> String {
        let bytes = deckcraft_pptx::export(&card_deck(slides)).unwrap();
        crate::cmd::base64_encode(&bytes)
    }

    #[test]
    fn template_scan_command_returns_slides() {
        let value = template_scan(&mut Session::with_new(), &json!({"pptx": card_pptx(2)})).unwrap();
        let slides = value["slides"].as_array().unwrap();
        assert_eq!(slides.len(), 2);
        assert_eq!(slides[0]["candidate_slots"].as_array().unwrap().len(), 1);
        assert_eq!(slides[1]["candidate_slots"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn template_scan_rejects_bad_base64() {
        let err = template_scan(&mut Session::with_new(), &json!({"pptx": "!!!!"})).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { .. }));
    }

    #[test]
    fn template_scan_rejects_non_pptx() {
        let b64 = crate::cmd::base64_encode(b"not a pptx");
        let err = template_scan(&mut Session::with_new(), &json!({"pptx": b64})).unwrap_err();
        assert!(matches!(err, EngineError::Other(message) if message.starts_with("PPTX import failed:")));
    }

    #[test]
    fn template_scan_filters_slides() {
        let value = template_scan(&mut Session::with_new(), &json!({"pptx": card_pptx(2), "slides": [1]})).unwrap();
        let slides = value["slides"].as_array().unwrap();
        assert_eq!(slides.len(), 1);
        assert_eq!(slides[0]["index"], 1);
    }

    #[test]
    #[ignore = "needs DECKCRAFT_TEST_PPTX"]
    fn template_scan_mud_full() {
        let Some(path) = std::env::var_os("DECKCRAFT_TEST_PPTX") else { return };
        let bytes = std::fs::read(path).unwrap();
        let (doc, _) = deckcraft_pptx::import_checked(&bytes).unwrap();
        let scan = crate::ai::scan_slide(&doc, 6).unwrap();
        assert_eq!(scan.candidate_slots.len(), 7);
        assert_eq!(scan.title_shapes, [4]);
        assert_eq!(scan.visible_title.as_deref(), Some("Rammed Earth Construction Process"));
        assert_eq!(scan.invisible, [2, 5, 10, 13, 20, 23, 30, 33, 40, 43, 50, 53, 60, 63, 70, 73]);
        assert_eq!(scan.rows, [vec![7, 17, 27, 37], vec![47, 57, 67]]);
        assert_eq!(scan.decorations, [77, 79, 81, 83, 85, 87]);
        assert_eq!(scan.unclassified, [89]);
    }
}
