//! Apply a template plan: delete dropped slots, fill the survivors, then realign them within
//! their rows.
//!
//! DeckCraft has `edit.delete`, `shape.setBounds`, `arrange.align` and `arrange.distribute`,
//! but none compose here: `setBounds` applies one box to every target, `align`/`distribute`
//! need a selection and don't preserve per-card xfrms, and each command is its own undo
//! entry. Doing the whole plan inside one `Session::edit` keeps undo atomic and the formula
//! in one place.

use std::collections::HashSet;
use std::sync::Arc;

use deckcraft_model::resolve::{self, Ctx};
use deckcraft_model::text::{ParaProps, RunProps, TextBody};
use deckcraft_model::{Presentation, Shape, ShapeId, Slide};
use serde::{Deserialize, Serialize};

use super::plan::SlotFill;
use super::scan;
use crate::cmd::edit::{cleanup_deleted, remove_shapes};
use crate::{EngineError, Result};

const CONNECTOR_TOUCH_TOLERANCE: f64 = 5.0;
const CONNECTOR_MAX_AREA_FRAC: f64 = 0.02;

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ApplyPlan {
    pub slides: Vec<ApplySlide>,
}

#[derive(Debug, Deserialize)]
pub struct ApplySlide {
    pub index: usize,
    pub keep_slot_ids: Vec<String>,
    #[serde(default)]
    pub fills: Vec<SlotFill>,
    #[serde(default)]
    pub realign: RealignMode,
    #[serde(default = "default_true")]
    pub prune_orphan_connectors: bool,
}

#[derive(Debug, Default, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum RealignMode {
    #[default]
    Auto,
    None,
}

#[derive(Debug, Serialize)]
pub struct ApplyResult {
    pub slides: Vec<SlideResult>,
}

#[derive(Debug, Serialize)]
pub struct SlideResult {
    pub index: usize,
    pub deleted: usize,
    pub moved: usize,
    pub rows_realigned: usize,
    pub filled: usize,
}

struct SlotInfo {
    card_id: u32,
    bbox: [f64; 4],
    top_level: Vec<u32>,
}

struct StagedFill {
    heading: Option<(u32, String)>,
    body: Option<(u32, String)>,
}

struct ParaStyle {
    level: u8,
    props: ParaProps,
    def_props: RunProps,
    end_props: RunProps,
    run: Option<RunProps>,
}

struct StagedSlide {
    index: usize,
    realign: RealignMode,
    delete_ids: Vec<ShapeId>,
    kept: Vec<SlotInfo>,
    rows: Vec<Vec<u32>>,
    fills: Vec<StagedFill>,
}

pub fn apply_plan(doc: &mut Presentation, plan: &ApplyPlan) -> Result<ApplyResult> {
    let slide_w = doc.slide_size.width;
    let slide_h = doc.slide_size.height;

    // Stage: validate the whole plan before mutating anything, so a bad plan doesn't
    // leave a partial delete on the undo stack.
    let mut staged: Vec<StagedSlide> = Vec::with_capacity(plan.slides.len());
    let mut seen_slides = HashSet::new();
    for s in &plan.slides {
        if !seen_slides.insert(s.index) {
            return Err(bad(format!("slide {} appears more than once", s.index)));
        }
        if s.keep_slot_ids.is_empty() {
            return Err(bad(format!("slide {} has empty `keep_slot_ids`", s.index)));
        }
        let scan = scan::scan_slide(doc, s.index).ok_or_else(|| bad(format!("no slide {}", s.index)))?;
        let keep: HashSet<&str> = s.keep_slot_ids.iter().map(String::as_str).collect();
        for id in &keep {
            if !scan.candidate_slots.iter().any(|slot| slot.id == *id) {
                return Err(bad(format!("slide {} has no slot `{id}`", s.index)));
            }
        }
        let mut delete_ids: Vec<ShapeId> = Vec::new();
        let mut kept: Vec<SlotInfo> = Vec::new();
        for slot in &scan.candidate_slots {
            if keep.contains(slot.id.as_str()) {
                kept.push(SlotInfo { card_id: slot.background, bbox: slot.bbox, top_level: slot.top_level_members.clone() });
            } else {
                for id in &slot.children {
                    delete_ids.push(ShapeId(*id));
                }
            }
        }
        if s.prune_orphan_connectors {
            for id in connector_prune_targets(doc, s.index, &scan, &keep, slide_w, slide_h) {
                delete_ids.push(ShapeId(id));
            }
        }
        let fills = stage_fills(s.index, &scan, &keep, &delete_ids, &s.fills)?;
        staged.push(StagedSlide { index: s.index, realign: s.realign, delete_ids, kept, rows: scan.rows, fills });
    }

    // Apply: mutate each slide once, so the whole plan is one undo entry per Session::edit.
    let mut result = ApplyResult { slides: Vec::with_capacity(staged.len()) };
    for st in staged {
        let slide = match doc.slides.get_mut(st.index) {
            Some(s) => Arc::make_mut(s),
            None => return Err(EngineError::Other(format!("ai.applyTemplatePlan: slide {} disappeared mid-apply", st.index))),
        };

        let before = shape_ids(&slide.shapes);
        let deleted = remove_shapes(&mut slide.shapes, &st.delete_ids);
        if deleted > 0 {
            // `remove_shapes` drops group descendants and emptied groups too; clean up
            // everything that is actually gone, not just the ids named in the plan.
            let after = shape_ids(&slide.shapes);
            let gone: HashSet<ShapeId> = before.difference(&after).copied().collect();
            cleanup_deleted(slide, &gone);
        }

        let mut moved = 0usize;
        let mut rows_realigned = 0usize;
        if st.realign == RealignMode::Auto {
            for row in &st.rows {
                let mut row_slots: Vec<&SlotInfo> = st.kept.iter().filter(|k| row.contains(&k.card_id)).collect();
                if row_slots.len() < 2 {
                    continue;
                }
                let row_moved = realign_row(slide, slide_w, &mut row_slots);
                if row_moved > 0 {
                    moved += row_moved;
                    rows_realigned += 1;
                }
            }
        }

        let filled = st.fills.len();
        for fill in &st.fills {
            if let Some((id, text)) = &fill.heading {
                fill_text_shape(slide, ShapeId(*id), text)?;
            }
            if let Some((id, text)) = &fill.body {
                fill_text_shape(slide, ShapeId(*id), text)?;
            }
        }

        result.slides.push(SlideResult { index: st.index, deleted, moved, rows_realigned, filled });
    }
    Ok(result)
}

fn stage_fills(
    slide_index: usize,
    scan: &scan::SlideScan,
    keep: &HashSet<&str>,
    delete_ids: &[ShapeId],
    fills: &[SlotFill],
) -> Result<Vec<StagedFill>> {
    let deleting: HashSet<ShapeId> = delete_ids.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(fills.len());
    for fill in fills {
        if !seen.insert(fill.slot_id.as_str()) {
            return Err(bad(format!("slide {slide_index} fills slot `{}` more than once", fill.slot_id)));
        }
        if !keep.contains(fill.slot_id.as_str()) {
            return Err(bad(format!("slide {slide_index} fills slot `{}` but does not keep it", fill.slot_id)));
        }
        let slot = scan
            .candidate_slots
            .iter()
            .find(|slot| slot.id == fill.slot_id)
            .ok_or_else(|| bad(format!("slide {slide_index} has no slot `{}`", fill.slot_id)))?;
        if fill.heading.is_none() && fill.body.is_none() {
            return Err(bad(format!("slide {slide_index} fill for `{}` needs a heading or body", fill.slot_id)));
        }
        let heading = match (&fill.heading, slot.heading) {
            (Some(text), Some(id)) => Some((id, text.clone())),
            (Some(_), None) => return Err(bad(format!("slide {slide_index} slot `{}` has no heading", fill.slot_id))),
            (None, _) => None,
        };
        let body = match (&fill.body, slot.body) {
            (Some(text), Some(id)) => Some((id, text.clone())),
            (Some(_), None) => return Err(bad(format!("slide {slide_index} slot `{}` has no body", fill.slot_id))),
            (None, _) => None,
        };
        for (id, _) in heading.iter().chain(body.iter()) {
            if deleting.contains(&ShapeId(*id)) {
                return Err(bad(format!("slide {slide_index} fill for `{}` targets a dropped shape", fill.slot_id)));
            }
        }
        out.push(StagedFill { heading, body });
    }
    Ok(out)
}

fn fill_text_shape(slide: &mut Slide, id: ShapeId, text: &str) -> Result<()> {
    let Some(shape) = deckcraft_model::find_shape_mut(&mut slide.shapes, id) else {
        return Err(EngineError::Other(format!("ai.applyTemplatePlan: fill target {id} disappeared mid-apply")));
    };
    let Some(body) = shape.text.as_mut() else {
        return Err(EngineError::Other(format!("ai.applyTemplatePlan: fill target {id} lost its text body")));
    };
    fill_text_body(body, text);
    Ok(())
}

fn fill_text_body(target: &mut TextBody, text: &str) {
    let styles: Vec<ParaStyle> = target
        .paragraphs
        .iter()
        .map(|p| ParaStyle {
            level: p.level,
            props: p.props.clone(),
            def_props: p.def_props.clone(),
            end_props: p.end_props.clone(),
            run: p.runs.first().map(|r| r.props.clone()),
        })
        .collect();
    let fallback_run = styles.iter().find_map(|style| style.run.clone()).unwrap_or_default();
    let mut next = TextBody::from_text(text);
    if !styles.is_empty() {
        let last = styles.len() - 1;
        for (i, paragraph) in next.paragraphs.iter_mut().enumerate() {
            let style = &styles[i.min(last)];
            paragraph.level = style.level;
            paragraph.props = style.props.clone();
            paragraph.def_props = style.def_props.clone();
            paragraph.end_props = style.end_props.clone();
            let run = style.run.as_ref().unwrap_or(&fallback_run);
            for r in &mut paragraph.runs {
                r.props = run.clone();
            }
        }
    }
    target.paragraphs = next.paragraphs;
}

fn realign_row(slide: &mut Slide, slide_w: f64, row_slots: &mut Vec<&SlotInfo>) -> usize {
    if !row_slots.iter().all(|k| k.bbox.iter().all(|v| v.is_finite())) {
        return 0;
    }
    row_slots.sort_by(|a, b| a.bbox[0].partial_cmp(&b.bbox[0]).unwrap_or(std::cmp::Ordering::Equal));
    let total_w: f64 = row_slots.iter().map(|k| k.bbox[2]).sum();
    let gap = (slide_w - total_w) / (row_slots.len() + 1) as f64;
    if !gap.is_finite() || gap < 0.0 {
        return 0;
    }
    let target_y = row_slots.iter().map(|k| k.bbox[1]).sum::<f64>() / row_slots.len() as f64;
    let mut x = gap;
    let mut moved = 0usize;
    for slot in row_slots {
        let dx = x - slot.bbox[0];
        let dy = target_y - slot.bbox[1];
        if dx.abs() > 0.01 || dy.abs() > 0.01 {
            for &id in &slot.top_level {
                if let Some(sh) = slide.shape_mut(ShapeId(id))
                    && let Some(xfrm) = sh.xfrm.as_mut()
                {
                    xfrm.x += dx;
                    xfrm.y += dy;
                }
            }
            moved += 1;
        }
        x += slot.bbox[2] + gap;
    }
    moved
}

fn shape_ids(shapes: &[Shape]) -> HashSet<ShapeId> {
    let mut out = HashSet::new();
    deckcraft_model::walk(shapes, &mut |s, _| {
        out.insert(s.id);
    });
    out
}

/// Decorations that bridged at least two cards before the delete but bridge fewer than
/// two kept cards. A decoration that only ever touched one card is not a connector and
/// is left alone, so ordinary design elements are not collateral damage.
fn connector_prune_targets(
    doc: &Presentation,
    slide_index: usize,
    scan: &scan::SlideScan,
    kept: &HashSet<&str>,
    slide_w: f64,
    slide_h: f64,
) -> Vec<u32> {
    let slide_area = slide_w * slide_h;
    let Some(slide) = doc.slides.get(slide_index) else { return Vec::new() };
    let Some(ctx) = Ctx::for_slide(doc, slide) else { return Vec::new() };
    let all_cards: Vec<[f64; 4]> = scan.candidate_slots.iter().map(|s| s.bbox).collect();
    let kept_cards: Vec<[f64; 4]> = scan.candidate_slots.iter().filter(|s| kept.contains(s.id.as_str())).map(|s| s.bbox).collect();
    let mut out = Vec::new();
    for &dec_id in &scan.decorations {
        let Some(sh) = slide.shape(ShapeId(dec_id)) else { continue };
        let x = resolve::xfrm(&ctx, sh);
        let dec_box = [x.x, x.y, x.w, x.h];
        if (x.w * x.h).abs() > slide_area * CONNECTOR_MAX_AREA_FRAC {
            continue;
        }
        if all_cards.iter().filter(|c| intersects(dec_box, **c, CONNECTOR_TOUCH_TOLERANCE)).count() < 2 {
            continue;
        }
        if kept_cards.iter().filter(|c| intersects(dec_box, **c, CONNECTOR_TOUCH_TOLERANCE)).count() < 2 {
            out.push(dec_id);
        }
    }
    out
}

fn intersects(a: [f64; 4], b: [f64; 4], tolerance: f64) -> bool {
    a[0] - tolerance < b[0] + b[2] && a[0] + a[2] + tolerance > b[0] && a[1] - tolerance < b[1] + b[3] && a[1] + a[3] + tolerance > b[1]
}

fn bad(msg: String) -> EngineError {
    EngineError::BadParams { cmd: "ai.applyTemplatePlan".into(), msg }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::sync::Arc;

    use deckcraft_model::style::PictureFill;
    use deckcraft_model::text::Bullet;
    use deckcraft_model::{ColorRef, Fill, LayoutId, Paragraph, Rgba, Shape, ShapeId, ShapeKind, Slide, SlideId, TextBody, Xfrm};
    use serde_json::json;

    use super::*;
    use crate::Session;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Xfrm {
        Xfrm::new(x, y, w, h)
    }

    fn deck(w: f64, h: f64) -> Presentation {
        let mut doc = Presentation { slide_size: deckcraft_model::Size::new(w, h), ..Default::default() };
        doc.slides.clear();
        doc
    }

    fn add_slide(doc: &mut Presentation, shapes: Vec<Shape>) {
        let layout = doc.masters.first().and_then(|m| m.layouts.first()).map(|l| l.id).unwrap_or(LayoutId(0));
        let id = SlideId(200 + doc.slides.len() as u32);
        doc.slides.push(Arc::new(Slide { id, layout, shapes, ..Default::default() }));
    }

    fn card(id: u32, x: f64, y: f64, w: f64, h: f64) -> Shape {
        Shape { id: ShapeId(id), xfrm: Some(rect(x, y, w, h)), kind: ShapeKind::Picture { fill: PictureFill::default() }, ..Default::default() }
    }

    fn solid(id: u32, x: f64, y: f64, w: f64, h: f64) -> Shape {
        Shape { id: ShapeId(id), xfrm: Some(rect(x, y, w, h)), fill: Some(Fill::solid(ColorRef::rgb(Rgba::rgb(80, 60, 120)))), ..Default::default() }
    }

    fn text(id: u32, x: f64, y: f64, w: f64, h: f64, value: &str) -> Shape {
        Shape {
            id: ShapeId(id),
            xfrm: Some(rect(x, y, w, h)),
            text_box: true,
            fill: Some(Fill::None),
            text: Some(TextBody::from_text(value)),
            ..Default::default()
        }
    }

    fn group(id: u32, x: f64, y: f64, w: f64, h: f64, children: Vec<Shape>) -> Shape {
        Shape {
            id: ShapeId(id),
            xfrm: Some(rect(x, y, w, h)),
            kind: ShapeKind::Group { children, child: rect(0.0, 0.0, w, h) },
            ..Default::default()
        }
    }

    fn slot(shapes: &mut Vec<Shape>, base: u32, x: f64, y: f64, w: f64, h: f64, label: &str) {
        shapes.push(card(base, x, y, w, h));
        shapes.push(group(base + 1, x + 10.0, y + 10.0, 40.0, 40.0, vec![solid(base + 2, 0.0, 0.0, 40.0, 40.0)]));
        shapes.push(text(base + 3, x + 60.0, y + 10.0, w - 70.0, 60.0, label));
        shapes.push(text(base + 4, x + 10.0, y + 80.0, w - 20.0, h - 90.0, "Body"));
    }

    fn three_slots() -> Presentation {
        let mut doc = deck(1440.0, 810.0);
        let mut shapes = Vec::new();
        slot(&mut shapes, 10, 50.0, 100.0, 300.0, 220.0, "One");
        slot(&mut shapes, 20, 400.0, 100.0, 300.0, 220.0, "Two");
        slot(&mut shapes, 30, 750.0, 100.0, 300.0, 220.0, "Three");
        add_slide(&mut doc, shapes);
        doc
    }

    fn overflow_slots() -> Presentation {
        let mut doc = deck(1440.0, 810.0);
        let mut shapes = Vec::new();
        slot(&mut shapes, 10, 0.0, 100.0, 600.0, 220.0, "One");
        slot(&mut shapes, 20, 600.0, 100.0, 600.0, 220.0, "Two");
        slot(&mut shapes, 30, 1200.0, 100.0, 600.0, 220.0, "Three");
        add_slide(&mut doc, shapes);
        doc
    }

    fn two_cards_with(extra: Vec<Shape>) -> Presentation {
        let mut doc = deck(1440.0, 810.0);
        let mut shapes = vec![card(10, 50.0, 100.0, 300.0, 220.0), card(20, 380.0, 100.0, 300.0, 220.0)];
        shapes.extend(extra);
        add_slide(&mut doc, shapes);
        doc
    }

    fn arrow(id: u32, x: f64, y: f64, w: f64, h: f64) -> Shape {
        group(id, x, y, w, h, vec![solid(id + 1, 0.0, 0.0, w, h)])
    }

    fn two_slides() -> Presentation {
        let mut doc = deck(1440.0, 810.0);
        for s in 0..2u32 {
            let mut shapes = Vec::new();
            let base = 10 + s * 10;
            slot(&mut shapes, base, 50.0, 100.0, 300.0, 220.0, "One");
            slot(&mut shapes, base + 10, 400.0, 100.0, 300.0, 220.0, "Two");
            slot(&mut shapes, base + 20, 750.0, 100.0, 300.0, 220.0, "Three");
            add_slide(&mut doc, shapes);
        }
        doc
    }

    fn plan(keep: &[&str], realign: RealignMode) -> ApplyPlan {
        ApplyPlan {
            slides: vec![ApplySlide {
                index: 0,
                keep_slot_ids: keep.iter().map(|s| (*s).to_string()).collect(),
                fills: Vec::new(),
                realign,
                prune_orphan_connectors: true,
            }],
        }
    }

    fn plan_json(keep: &[&str], realign: Option<&str>) -> serde_json::Value {
        let mut slide = json!({"index": 0, "keep_slot_ids": keep});
        if let Some(realign) = realign {
            slide["realign"] = json!(realign);
        }
        json!({"plan": {"slides": [slide]}})
    }

    fn session(doc: Presentation) -> Session {
        let mut s = Session::with_new();
        {
            let st = s.doc_mut().unwrap();
            st.doc = Arc::new(doc);
            st.selection = Default::default();
            st.history.undo.clear();
            st.history.redo.clear();
        }
        s
    }

    fn undo_len(s: &Session) -> usize {
        s.doc().unwrap().history.undo.len()
    }

    fn shape(doc: &Presentation, id: u32) -> &Shape {
        doc.slides[0].shape(ShapeId(id)).unwrap()
    }

    #[test]
    fn apply_deletes_dropped_slots_keeps_children() {
        let mut doc = three_slots();
        let was = scan::scan_slide(&doc, 0).unwrap();
        let dropped_children = was.candidate_slots[1].children.clone();
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c2"], RealignMode::Auto)).unwrap();
        let r = &res.slides[0];
        assert_eq!(r.deleted, dropped_children.len());
        assert_eq!((r.moved, r.rows_realigned), (2, 1));
        for id in dropped_children {
            assert!(doc.slides[0].shape(ShapeId(id)).is_none(), "dropped child {id} survived");
        }
        assert!((shape(&doc, 10).xfrm.unwrap().x - 280.0).abs() < 1e-9);
        assert!((shape(&doc, 30).xfrm.unwrap().x - 860.0).abs() < 1e-9);
        assert!((shape(&doc, 11).xfrm.unwrap().x - 290.0).abs() < 1e-9);
        assert_eq!(shape(&doc, 12).xfrm.unwrap().x, 0.0, "group descendants keep child-space coords");
    }

    #[test]
    fn apply_realign_none_skips_movement() {
        let mut doc = three_slots();
        let before = shape(&doc, 10).xfrm.unwrap().x;
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c2"], RealignMode::None)).unwrap();
        assert_eq!((res.slides[0].moved, res.slides[0].rows_realigned), (0, 0));
        assert_eq!(shape(&doc, 10).xfrm.unwrap().x, before);
        assert_eq!(shape(&doc, 30).xfrm.unwrap().x, 750.0);
    }

    #[test]
    fn apply_single_undo_entry() {
        let mut s = session(two_slides());
        let before = undo_len(&s);
        s.execute(
            "ai.applyTemplatePlan",
            &json!({"plan": {"slides": [
                {"index": 0, "keep_slot_ids": ["s0-c0", "s0-c2"]},
                {"index": 1, "keep_slot_ids": ["s1-c0", "s1-c2"]}
            ]}}),
        )
        .unwrap();
        assert_eq!(undo_len(&s), before + 1, "one plan is one undo step");
        s.execute("edit.delete", &json!({"ids": [10]})).unwrap();
        assert_eq!(undo_len(&s), before + 2, "edit.delete adds its own step, not a nested one");
    }

    #[test]
    fn apply_undo_restores_bytes() {
        let mut s = session(three_slots());
        let before = serde_json::to_vec(&*s.doc().unwrap().doc).unwrap();
        s.execute("ai.applyTemplatePlan", &plan_json(&["s0-c0", "s0-c2"], None)).unwrap();
        s.execute("edit.undo", &json!({})).unwrap();
        let after = serde_json::to_vec(&*s.doc().unwrap().doc).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn apply_unknown_slot_id_rejects() {
        let mut s = session(three_slots());
        let before = serde_json::to_vec(&*s.doc().unwrap().doc).unwrap();
        let before_undo = undo_len(&s);
        let err = s.execute("ai.applyTemplatePlan", &plan_json(&["s0-c9"], None)).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { msg, .. } if msg.contains("s0-c9") && msg.contains('0')));
        assert_eq!(serde_json::to_vec(&*s.doc().unwrap().doc).unwrap(), before);
        assert_eq!(undo_len(&s), before_undo);
    }

    #[test]
    fn apply_empty_keep_slot_ids_rejects() {
        let mut s = session(three_slots());
        let before_undo = undo_len(&s);
        let err = s.execute("ai.applyTemplatePlan", &plan_json(&[], None)).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { msg, .. } if msg.contains("empty")));
        assert_eq!(undo_len(&s), before_undo);
    }

    #[test]
    fn apply_row_overflow_skipped() {
        let mut doc = overflow_slots();
        let before: Vec<f64> = [10, 20, 30].into_iter().map(|id| shape(&doc, id).xfrm.unwrap().x).collect();
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c1", "s0-c2"], RealignMode::Auto)).unwrap();
        assert_eq!((res.slides[0].moved, res.slides[0].rows_realigned), (0, 0));
        let after: Vec<f64> = [10, 20, 30].into_iter().map(|id| shape(&doc, id).xfrm.unwrap().x).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn apply_all_kept_still_realigns() {
        let mut doc = three_slots();
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c1", "s0-c2"], RealignMode::Auto)).unwrap();
        assert_eq!(res.slides[0].deleted, 0);
        assert_eq!((res.slides[0].moved, res.slides[0].rows_realigned), (3, 1));
    }

    #[test]
    fn apply_fills_heading_and_body() {
        let mut doc = three_slots();
        let mut p = plan(&["s0-c0"], RealignMode::None);
        p.slides[0].fills.push(SlotFill {
            slot_id: "s0-c0".into(),
            heading: Some("Why now".into()),
            body: Some("Market shift\nCustomer pain".into()),
        });
        let res = apply_plan(&mut doc, &p).unwrap();
        assert_eq!(res.slides[0].filled, 1);
        assert_eq!(shape(&doc, 13).text.as_ref().unwrap().text(), "Why now");
        assert_eq!(shape(&doc, 14).text.as_ref().unwrap().text(), "Market shift\nCustomer pain");
    }

    #[test]
    fn apply_fill_keeps_template_paragraph_formatting_for_extra_lines() {
        let mut doc = three_slots();
        {
            let slide = Arc::make_mut(&mut doc.slides[0]);
            let body = slide.shape_mut(ShapeId(14)).unwrap().text.as_mut().unwrap();
            body.paragraphs = vec![Paragraph::new("First"), Paragraph::new("Second")];
            body.paragraphs[0].level = 0;
            body.paragraphs[0].props.bullet = Some(Bullet::Char { char: "•".into() });
            body.paragraphs[0].props.margin_left = Some(18.0);
            body.paragraphs[0].runs[0].props.size = Some(18.0);
            body.paragraphs[0].runs[0].props.bold = Some(true);
            body.paragraphs[1].level = 1;
            body.paragraphs[1].props.bullet = Some(Bullet::Char { char: "–".into() });
            body.paragraphs[1].props.margin_left = Some(36.0);
            body.paragraphs[1].runs[0].props.size = Some(14.0);
            body.paragraphs[1].runs[0].props.bold = None;
        }
        let mut p = plan(&["s0-c0"], RealignMode::None);
        p.slides[0].fills.push(SlotFill { slot_id: "s0-c0".into(), heading: None, body: Some("A\nB\nC\nD".into()) });
        apply_plan(&mut doc, &p).unwrap();
        let body = shape(&doc, 14).text.as_ref().unwrap();
        assert_eq!(body.paragraphs.len(), 4);
        assert_eq!(body.paragraphs.iter().map(|p| p.level).collect::<Vec<_>>(), [0, 1, 1, 1]);
        assert_eq!(body.paragraphs[0].props.margin_left, Some(18.0));
        assert_eq!(body.paragraphs[0].props.bullet, Some(Bullet::Char { char: "•".into() }));
        assert_eq!(body.paragraphs[0].runs[0].props.size, Some(18.0));
        assert_eq!(body.paragraphs[0].runs[0].props.bold, Some(true));
        assert_eq!(body.paragraphs[2].props.margin_left, Some(36.0));
        assert_eq!(body.paragraphs[3].props.bullet, Some(Bullet::Char { char: "–".into() }));
        assert_eq!(body.paragraphs[2].runs[0].props.size, Some(14.0));
        assert_eq!(body.paragraphs[2].runs[0].props.bold, None);
        assert_eq!(body.paragraphs[2].text(), "C");
    }

    #[test]
    fn fill_rejects_a_slot_that_is_not_kept() {
        let mut s = session(three_slots());
        let before = serde_json::to_vec(&*s.doc().unwrap().doc).unwrap();
        let before_undo = undo_len(&s);
        let err = s
            .execute(
                "ai.applyTemplatePlan",
                &json!({"plan": {"slides": [{"index": 0, "keep_slot_ids": ["s0-c0"], "fills": [{"slot_id": "s0-c1", "heading": "Two"}]}]}}),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::BadParams { msg, .. } if msg.contains("does not keep")));
        assert_eq!(serde_json::to_vec(&*s.doc().unwrap().doc).unwrap(), before);
        assert_eq!(undo_len(&s), before_undo);
    }

    #[test]
    fn fill_rejects_an_empty_fill() {
        let mut s = session(three_slots());
        let err = s
            .execute(
                "ai.applyTemplatePlan",
                &json!({"plan": {"slides": [{"index": 0, "keep_slot_ids": ["s0-c0"], "fills": [{"slot_id": "s0-c0"}]}]}}),
            )
            .unwrap_err();
        assert!(matches!(err, EngineError::BadParams { msg, .. } if msg.contains("needs a heading or body")));
    }

    #[test]
    fn fill_rejects_a_missing_target_field() {
        let mut doc = deck(1440.0, 810.0);
        add_slide(&mut doc, vec![card(10, 50.0, 100.0, 300.0, 220.0)]);
        let mut p = plan(&["s0-c0"], RealignMode::None);
        p.slides[0].fills.push(SlotFill { slot_id: "s0-c0".into(), heading: Some("Title".into()), body: None });
        let err = apply_plan(&mut doc, &p).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { msg, .. } if msg.contains("no heading")));
    }

    #[test]
    fn fill_rejects_duplicate_slot_fills() {
        let mut doc = three_slots();
        let mut p = plan(&["s0-c0"], RealignMode::None);
        p.slides[0].fills.push(SlotFill { slot_id: "s0-c0".into(), heading: Some("A".into()), body: None });
        p.slides[0].fills.push(SlotFill { slot_id: "s0-c0".into(), heading: Some("B".into()), body: None });
        let err = apply_plan(&mut doc, &p).unwrap_err();
        assert!(matches!(err, EngineError::BadParams { msg, .. } if msg.contains("more than once")));
    }

    #[test]
    fn fill_and_delete_share_one_undo_entry() {
        let mut s = session(three_slots());
        let before = serde_json::to_vec(&*s.doc().unwrap().doc).unwrap();
        let before_undo = undo_len(&s);
        let res = s
            .execute(
                "ai.applyTemplatePlan",
                &json!({"plan": {"slides": [{"index": 0, "keep_slot_ids": ["s0-c0", "s0-c2"], "fills": [{"slot_id": "s0-c0", "heading": "Why now", "body": "A\nB"}]}]}}),
            )
            .unwrap();
        assert_eq!(res["slides"][0]["filled"], 1);
        assert_eq!(undo_len(&s), before_undo + 1);
        assert_eq!(s.doc().unwrap().doc.slides[0].shape(ShapeId(13)).unwrap().text.as_ref().unwrap().text(), "Why now");
        s.execute("edit.undo", &json!({})).unwrap();
        assert_eq!(serde_json::to_vec(&*s.doc().unwrap().doc).unwrap(), before);
    }

    #[test]
    fn connector_between_two_kept_is_preserved() {
        let mut doc = two_cards_with(vec![arrow(30, 345.0, 150.0, 40.0, 20.0)]);
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c1"], RealignMode::None)).unwrap();
        assert_eq!(res.slides[0].deleted, 0);
        assert!(doc.slides[0].shape(ShapeId(30)).is_some());
        assert!(doc.slides[0].shape(ShapeId(31)).is_some());
    }

    #[test]
    fn connector_with_one_dropped_is_removed() {
        let mut doc = two_cards_with(vec![arrow(30, 345.0, 150.0, 40.0, 20.0)]);
        let res = apply_plan(&mut doc, &plan(&["s0-c0"], RealignMode::None)).unwrap();
        assert_eq!(res.slides[0].deleted, 3, "dropped card 20, arrow group 30, arrow child 31");
        assert!(doc.slides[0].shape(ShapeId(20)).is_none());
        assert!(doc.slides[0].shape(ShapeId(30)).is_none());
        assert!(doc.slides[0].shape(ShapeId(31)).is_none());
    }

    #[test]
    fn connector_between_two_dropped_is_removed() {
        let doc = two_cards_with(vec![arrow(30, 345.0, 150.0, 40.0, 20.0)]);
        let scan = scan::scan_slide(&doc, 0).unwrap();
        let targets = connector_prune_targets(&doc, 0, &scan, &std::collections::HashSet::new(), 1440.0, 810.0);
        assert_eq!(targets, [30]);
    }

    #[test]
    fn decoration_touching_one_card_untouched() {
        let mut doc = two_cards_with(vec![arrow(40, 354.0, 150.0, 20.0, 20.0)]);
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c1"], RealignMode::None)).unwrap();
        assert_eq!(res.slides[0].deleted, 0);
        assert!(doc.slides[0].shape(ShapeId(40)).is_some());
    }

    #[test]
    fn decoration_touching_nothing_untouched() {
        let mut doc = two_cards_with(vec![arrow(40, 1000.0, 600.0, 20.0, 20.0)]);
        let res = apply_plan(&mut doc, &plan(&["s0-c0", "s0-c1"], RealignMode::None)).unwrap();
        assert_eq!(res.slides[0].deleted, 0);
        assert!(doc.slides[0].shape(ShapeId(40)).is_some());
    }

    #[test]
    fn prune_disabled_keeps_all_connectors() {
        let mut doc = two_cards_with(vec![arrow(30, 345.0, 150.0, 40.0, 20.0)]);
        let mut p = plan(&["s0-c0"], RealignMode::None);
        p.slides[0].prune_orphan_connectors = false;
        let res = apply_plan(&mut doc, &p).unwrap();
        assert_eq!(res.slides[0].deleted, 1);
        assert!(doc.slides[0].shape(ShapeId(30)).is_some());
    }

    #[test]
    fn prune_does_not_touch_non_connector_decorations() {
        let mut doc = two_cards_with(vec![arrow(30, 345.0, 150.0, 40.0, 20.0), arrow(40, 1000.0, 600.0, 20.0, 20.0)]);
        let res = apply_plan(&mut doc, &plan(&["s0-c0"], RealignMode::None)).unwrap();
        assert_eq!(res.slides[0].deleted, 3);
        assert!(doc.slides[0].shape(ShapeId(30)).is_none());
        assert!(doc.slides[0].shape(ShapeId(40)).is_some());
    }

    #[test]
    fn prune_is_idempotent() {
        let mut doc = two_cards_with(vec![arrow(30, 345.0, 150.0, 40.0, 20.0)]);
        let first = apply_plan(&mut doc, &plan(&["s0-c0"], RealignMode::Auto)).unwrap();
        assert_eq!(first.slides[0].deleted, 3);
        let second = apply_plan(&mut doc, &plan(&["s0-c0"], RealignMode::Auto)).unwrap();
        assert_eq!((second.slides[0].deleted, second.slides[0].moved), (0, 0));
    }

    #[test]
    #[ignore = "needs DECKCRAFT_TEST_PPTX"]
    fn apply_mud_slide6() {
        let Some(path) = std::env::var_os("DECKCRAFT_TEST_PPTX") else { return };
        let bytes = std::fs::read(path).unwrap();
        let (mut doc, _) = deckcraft_pptx::import_checked(&bytes).unwrap();
        let scan = scan::scan_slide(&doc, 6).unwrap();
        let keep: Vec<String> = scan.candidate_slots.iter().take(5).map(|s| s.id.clone()).collect();
        let dropped: usize = scan.candidate_slots.iter().skip(5).map(|s| s.children.len()).sum();
        let first = &scan.candidate_slots[0];
        let heading = first.heading.unwrap();
        let body = first.body.unwrap();
        let fills = vec![SlotFill {
            slot_id: first.id.clone(),
            heading: Some("Why now".into()),
            body: Some("Market shift\nCustomer pain\nThird point".into()),
        }];
        let plan = ApplyPlan {
            slides: vec![ApplySlide { index: 6, keep_slot_ids: keep, fills, realign: RealignMode::Auto, prune_orphan_connectors: true }],
        };
        let res = apply_plan(&mut doc, &plan).unwrap();
        assert_eq!(res.slides[0].deleted, dropped + 4, "two connector groups and their children");
        assert_eq!((res.slides[0].moved, res.slides[0].rows_realigned), (4, 1));
        assert_eq!(res.slides[0].filled, 1);
        for id in [85, 87] {
            assert!(doc.slides[6].shape(ShapeId(id)).is_none(), "orphan arrow {id} survived");
        }
        assert!(doc.slides[6].shape(ShapeId(83)).is_some(), "arrow 83 still bridges cards 37 and 47");
        assert_eq!(doc.slides[6].shape(ShapeId(heading)).unwrap().text.as_ref().unwrap().text(), "Why now");
        let filled_body = doc.slides[6].shape(ShapeId(body)).unwrap().text.as_ref().unwrap();
        assert_eq!(filled_body.text(), "Market shift\nCustomer pain\nThird point");
        assert_eq!(filled_body.paragraphs.len(), 3);
        println!("mud slide 6: {:?}", res.slides[0]);
    }

    #[test]
    fn apply_cleans_animations_of_deleted_shapes() {
        let mut doc = three_slots();
        let anim = deckcraft_model::anim::Animation { shape: ShapeId(22), effect: "fade".into(), trigger: Some(ShapeId(20)), ..Default::default() };
        let kept = deckcraft_model::anim::Animation { shape: ShapeId(10), effect: "fade".into(), trigger: Some(ShapeId(22)), ..Default::default() };
        Arc::make_mut(&mut doc.slides[0]).animations = vec![anim, kept];
        apply_plan(&mut doc, &plan(&["s0-c0", "s0-c2"], RealignMode::None)).unwrap();
        let animations = &doc.slides[0].animations;
        assert_eq!(animations.len(), 1);
        assert_eq!(animations[0].shape, ShapeId(10));
        assert_eq!(animations[0].trigger, None, "trigger pointing at a deleted shape is cleared");
    }
}
