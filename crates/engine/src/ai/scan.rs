//! Template scanning: find candidate "slots" (card + heading + body + badge + icons) on a
//! slide. Purely geometric — no AI, no rendering, no name heuristics. The output feeds Phase
//! 4's prompt; Phase 3b applies the plan (keep/delete + realign) via existing commands.

use std::collections::HashSet;

use deckcraft_model::resolve::{self, Ctx};
use deckcraft_model::{Fill, Presentation, Shape, ShapeKind};
use serde::{Deserialize, Serialize};

const CARD_MIN_AREA_FRAC: f64 = 0.05;
const TITLE_BAND_FRAC: f64 = 0.15;
const PAGE_NUM_AREA_FRAC: f64 = 0.005;
const PAGE_NUM_CORNER_FRAC: f64 = 0.15;
const ALPHA_EPS: f64 = 0.001;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateSlot {
    pub id: String,
    pub bbox: [f64; 4],
    pub background: u32,
    pub heading: Option<u32>,
    pub body: Option<u32>,
    pub badge: Option<u32>,
    pub icons: Vec<u32>,
    pub children: Vec<u32>,
    /// Members to move as a unit: the card plus its top-level shapes. Descendants are not
    /// included because moving them directly would write child-space coordinates.
    pub top_level_members: Vec<u32>,
    pub confidence: f32,
    pub flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlideScan {
    pub index: usize,
    pub layout: String,
    pub title: String,
    pub visible_title: Option<String>,
    pub title_shapes: Vec<u32>,
    pub candidate_slots: Vec<CandidateSlot>,
    pub rows: Vec<Vec<u32>>,
    pub decorations: Vec<u32>,
    pub unclassified: Vec<u32>,
    pub invisible: Vec<u32>,
}

pub fn scan_presentation(doc: &Presentation) -> Vec<SlideScan> {
    (0..doc.slides.len()).filter_map(|i| scan_slide(doc, i)).collect()
}

pub fn scan_slide(doc: &Presentation, index: usize) -> Option<SlideScan> {
    let slide = doc.slides.get(index)?;
    let ctx = Ctx::for_slide(doc, slide)?;
    let slide_w = doc.slide_size.width;
    let slide_h = doc.slide_size.height;
    let slide_area = (slide_w * slide_h).max(1.0);
    let title_band = slide_h * TITLE_BAND_FRAC;

    // 1. Invisible (recursive through visible groups; stop at invisible group).
    let mut invisible: Vec<u32> = Vec::new();
    walk_for_invisible(&ctx, &slide.shapes, &mut invisible);
    let invisible_set: HashSet<u32> = invisible.iter().copied().collect();

    // 2. Cards: visible picture-fill shapes > 5%, OR one-child solid groups > 5%.
    let mut cards: Vec<(u32, [f64; 4])> = Vec::new();
    for sh in &slide.shapes {
        if sh.hidden || invisible_set.contains(&sh.id.0) {
            continue;
        }
        let x = resolve::xfrm(&ctx, sh);
        let area = (x.w * x.h).abs();
        if area < slide_area * CARD_MIN_AREA_FRAC {
            continue;
        }
        let (fill, _ph) = resolve::fill(&ctx, sh);
        let is_pic = matches!(fill, Some(Fill::Picture(_))) || matches!(&sh.kind, ShapeKind::Picture { .. });
        let is_group_card = matches!(&sh.kind, ShapeKind::Group { children, .. }
            if children.len() == 1
                && matches!(resolve::fill(&ctx, &children[0]).0, Some(Fill::Solid { .. })));
        if is_pic || is_group_card {
            cards.push((sh.id.0, [x.x, x.y, x.w, x.h]));
        }
    }
    let card_ids: HashSet<u32> = cards.iter().map(|(id, _)| *id).collect();

    // 3. Rows: cluster by vertical overlap > 50% of smaller height. Document order preserved.
    let rows_pairs = cluster_rows(&cards);

    // 4. Title band (top 15%).
    let title_shapes: Vec<u32> = slide
        .shapes
        .iter()
        .filter(|sh| {
            if sh.hidden || invisible_set.contains(&sh.id.0) {
                return false;
            }
            let x = resolve::xfrm(&ctx, sh);
            x.y + x.h <= title_band + 0.5
        })
        .map(|sh| sh.id.0)
        .collect();
    let visible_title = collect_visible_text(&slide.shapes, &title_shapes, &invisible_set);

    // 5. Slots.
    let mut slots: Vec<CandidateSlot> = Vec::new();
    let mut claimed: HashSet<u32> = HashSet::new();
    for (ci, (card_id, bbox)) in cards.iter().enumerate() {
        let members = members_of(&ctx, &slide.shapes, *bbox, &card_ids);
        let mut children: Vec<u32> = vec![*card_id];
        let mut top_level_members: Vec<u32> = vec![*card_id];
        for m in members {
            if !children.contains(&m) {
                children.push(m);
            }
            if !top_level_members.contains(&m) {
                top_level_members.push(m);
            }
            children.extend(descendants_of(&slide.shapes, m));
        }
        let mut seen = HashSet::new();
        children.retain(|id| seen.insert(*id));
        claimed.extend(children.iter().copied());

        let (heading, body) = find_heading_body(&ctx, &slide.shapes, *bbox, &card_ids, &invisible_set);
        let badge = find_badge(&ctx, &slide.shapes, *bbox, &card_ids, &invisible_set);
        let icons = find_icons(&ctx, &slide.shapes, *bbox, &card_ids, &invisible_set);

        let mut flags = Vec::new();
        if heading.is_none() {
            flags.push("no_heading".into());
        }
        if body.is_none() {
            flags.push("no_body".into());
        }
        let confidence = match (heading.is_some(), body.is_some()) {
            (true, true) => 0.9,
            (true, false) | (false, true) => 0.7,
            (false, false) => 0.3,
        };

        slots.push(CandidateSlot {
            id: format!("s{index}-c{ci}"),
            bbox: *bbox,
            background: *card_id,
            heading,
            body,
            badge,
            icons,
            children,
            top_level_members,
            confidence,
            flags,
        });
    }

    // 6. Page-number-like, decorations, unclassified.
    let page_number_like: HashSet<u32> =
        slide.shapes.iter().filter(|sh| is_page_number_like(&ctx, sh, slide_w, slide_h, slide_area)).map(|sh| sh.id.0).collect();

    let title_set: HashSet<u32> = title_shapes.iter().copied().collect();
    let mut decorations: Vec<u32> = Vec::new();
    let mut unclassified: Vec<u32> = Vec::new();
    for sh in &slide.shapes {
        let id = sh.id.0;
        if sh.hidden || invisible_set.contains(&id) || card_ids.contains(&id) || title_set.contains(&id) || claimed.contains(&id) {
            continue;
        }
        if page_number_like.contains(&id) {
            unclassified.push(id);
        } else {
            decorations.push(id);
        }
    }

    Some(SlideScan {
        index,
        layout: doc.layout(slide.layout).map(|(_, l)| l.name.clone()).unwrap_or_default(),
        title: slide.title(),
        visible_title,
        title_shapes,
        candidate_slots: slots,
        rows: rows_pairs.iter().map(|r| r.iter().map(|(id, _)| *id).collect()).collect(),
        decorations,
        unclassified,
        invisible,
    })
}

// ---------- helpers ----------

fn walk_for_invisible(c: &Ctx, shapes: &[Shape], out: &mut Vec<u32>) {
    for sh in shapes {
        if is_invisible(c, sh) {
            out.push(sh.id.0);
            continue;
        }
        if let ShapeKind::Group { children, .. } = &sh.kind {
            walk_for_invisible(c, children, out);
        }
    }
}

fn is_invisible(c: &Ctx, sh: &Shape) -> bool {
    if sh.hidden {
        return true;
    }
    let (fill, ph) = resolve::fill(c, sh);
    if let Some(f) = &fill {
        if let Fill::Picture(pf) = f
            && pf.alpha.is_some_and(|a| a <= ALPHA_EPS)
        {
            return true;
        }
        if let Fill::Solid { color } = f
            && c.color(color, ph).a == 0
        {
            return true;
        }
    }
    if let ShapeKind::Picture { fill: pf } = &sh.kind
        && pf.alpha.is_some_and(|a| a <= ALPHA_EPS)
    {
        return true;
    }
    if let ShapeKind::Group { children, .. } = &sh.kind
        && !children.is_empty()
        && children.iter().all(|ch| is_invisible(c, ch))
    {
        return true;
    }
    false
}

fn cluster_rows(cards: &[(u32, [f64; 4])]) -> Vec<Vec<(u32, [f64; 4])>> {
    let mut rows: Vec<Vec<(u32, [f64; 4])>> = Vec::new();
    for &(id, b) in cards {
        let mut placed = false;
        for row in rows.iter_mut() {
            let any = row.iter().any(|(_, rb)| {
                let overlap = (b[1] + b[3]).min(rb[1] + rb[3]) - b[1].max(rb[1]);
                let smaller = b[3].min(rb[3]);
                smaller > 0.0 && overlap / smaller > 0.5
            });
            if any {
                row.push((id, b));
                placed = true;
                break;
            }
        }
        if !placed {
            rows.push(vec![(id, b)]);
        }
    }
    rows.sort_by(|a, b| {
        let ya = a.iter().fold(f64::INFINITY, |m, (_, r)| m.min(r[1]));
        let yb = b.iter().fold(f64::INFINITY, |m, (_, r)| m.min(r[1]));
        ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

fn members_of(c: &Ctx, shapes: &[Shape], bbox: [f64; 4], cards: &HashSet<u32>) -> Vec<u32> {
    let mut out = Vec::new();
    for sh in shapes {
        let id = sh.id.0;
        if cards.contains(&id) {
            continue;
        }
        let x = resolve::xfrm(c, sh);
        let cx = x.x + x.w / 2.0;
        let cy = x.y + x.h / 2.0;
        if cx >= bbox[0] && cx <= bbox[0] + bbox[2] && cy >= bbox[1] && cy <= bbox[1] + bbox[3] {
            out.push(id);
        }
    }
    out
}

fn descendants_of(shapes: &[Shape], id: u32) -> Vec<u32> {
    let mut out = Vec::new();
    fn walk(shapes: &[Shape], id: u32, out: &mut Vec<u32>) -> bool {
        for sh in shapes {
            if sh.id.0 == id {
                if let ShapeKind::Group { children, .. } = &sh.kind {
                    for ch in children {
                        out.push(ch.id.0);
                        walk(children, ch.id.0, out);
                    }
                }
                return true;
            }
            if let ShapeKind::Group { children, .. } = &sh.kind
                && walk(children, id, out)
            {
                return true;
            }
        }
        false
    }
    walk(shapes, id, &mut out);
    out
}

fn find_heading_body(c: &Ctx, shapes: &[Shape], bbox: [f64; 4], cards: &HashSet<u32>, inv: &HashSet<u32>) -> (Option<u32>, Option<u32>) {
    let (mut heading, mut body) = (None, None);
    let (cx_min, cy_min, w, h) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    for sh in shapes {
        let id = sh.id.0;
        if cards.contains(&id) || inv.contains(&id) {
            continue;
        }
        if sh.text.as_ref().is_none_or(|t| t.is_empty()) {
            continue;
        }
        let x = resolve::xfrm(c, sh);
        let cx = x.x + x.w / 2.0;
        let cy = x.y + x.h / 2.0;
        if cx < cx_min || cx > cx_min + w {
            continue;
        }
        if cy >= cy_min && cy <= cy_min + h / 2.0 && heading.is_none() {
            heading = Some(id);
        } else if cy > cy_min + h / 2.0 && cy <= cy_min + h && body.is_none() {
            body = Some(id);
        }
    }
    (heading, body)
}

fn find_badge(c: &Ctx, shapes: &[Shape], bbox: [f64; 4], cards: &HashSet<u32>, inv: &HashSet<u32>) -> Option<u32> {
    for sh in shapes {
        let id = sh.id.0;
        if cards.contains(&id) || inv.contains(&id) {
            continue;
        }
        let ShapeKind::Group { children, .. } = &sh.kind else { continue };
        if !has_visible_text(children, inv) {
            continue;
        }
        let x = resolve::xfrm(c, sh);
        let cx = x.x + x.w / 2.0;
        let cy = x.y + x.h / 2.0;
        let in_tl = cx >= bbox[0] && cx <= bbox[0] + bbox[2] / 2.0 && cy >= bbox[1] && cy <= bbox[1] + bbox[3] / 2.0;
        if in_tl {
            return Some(id);
        }
    }
    None
}

fn find_icons(c: &Ctx, shapes: &[Shape], bbox: [f64; 4], cards: &HashSet<u32>, inv: &HashSet<u32>) -> Vec<u32> {
    let mut out = Vec::new();
    for sh in shapes {
        let id = sh.id.0;
        if cards.contains(&id) || inv.contains(&id) {
            continue;
        }
        let ShapeKind::Group { children, .. } = &sh.kind else { continue };
        if children.len() != 1 || has_visible_text(children, inv) {
            continue;
        }
        let x = resolve::xfrm(c, sh);
        let cx = x.x + x.w / 2.0;
        let cy = x.y + x.h / 2.0;
        let in_tr = cx >= bbox[0] + bbox[2] / 2.0 && cx <= bbox[0] + bbox[2] && cy >= bbox[1] && cy <= bbox[1] + bbox[3] / 2.0;
        if in_tr {
            out.push(id);
        }
    }
    out
}

fn has_visible_text(shapes: &[Shape], inv: &HashSet<u32>) -> bool {
    for sh in shapes {
        if inv.contains(&sh.id.0) {
            continue;
        }
        if sh.text.as_ref().is_some_and(|t| !t.is_empty()) {
            return true;
        }
        if let ShapeKind::Group { children, .. } = &sh.kind
            && has_visible_text(children, inv)
        {
            return true;
        }
    }
    false
}

fn is_page_number_like(c: &Ctx, sh: &Shape, sw: f64, sh_h: f64, slide_area: f64) -> bool {
    let x = resolve::xfrm(c, sh);
    let area = (x.w * x.h).abs();
    if area >= slide_area * PAGE_NUM_AREA_FRAC {
        return false;
    }
    let cx = x.x + x.w / 2.0;
    let cy = x.y + x.h / 2.0;
    if cx < sw * (1.0 - PAGE_NUM_CORNER_FRAC) || cy < sh_h * (1.0 - PAGE_NUM_CORNER_FRAC) {
        return false;
    }
    let Some(t) = &sh.text else { return false };
    t.text().trim().parse::<u32>().is_ok()
}

fn collect_visible_text(shapes: &[Shape], ids: &[u32], inv: &HashSet<u32>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for id in ids {
        fn walk(shapes: &[Shape], target: u32, inv: &HashSet<u32>, parts: &mut Vec<String>) -> bool {
            for sh in shapes {
                if sh.id.0 == target {
                    if !inv.contains(&sh.id.0)
                        && let Some(t) = &sh.text
                    {
                        let s = t.text();
                        if !s.trim().is_empty() {
                            parts.push(s);
                        }
                    }
                    if let ShapeKind::Group { children, .. } = &sh.kind {
                        for ch in children {
                            walk(children, ch.id.0, inv, parts);
                        }
                    }
                    return true;
                }
                if let ShapeKind::Group { children, .. } = &sh.kind
                    && walk(children, target, inv, parts)
                {
                    return true;
                }
            }
            false
        }
        walk(shapes, *id, inv, &mut parts);
    }
    if parts.is_empty() { None } else { Some(parts.join(" ")) }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::sync::Arc;

    use deckcraft_model::style::PictureFill;
    use deckcraft_model::{ColorRef, Fill, LayoutId, Rgba, ShapeId, Size, Slide, SlideId, TextBody, Xfrm};

    use super::*;

    fn rect(x: f64, y: f64, w: f64, h: f64) -> Xfrm {
        Xfrm::new(x, y, w, h)
    }

    fn deck(w: f64, h: f64) -> Presentation {
        let mut doc = Presentation { slide_size: Size::new(w, h), ..Default::default() };
        doc.slides.clear();
        doc
    }

    fn add_slide(doc: &mut Presentation, shapes: Vec<Shape>) {
        let layout = doc.masters.first().and_then(|m| m.layouts.first()).map(|l| l.id).unwrap_or(LayoutId(0));
        let id = SlideId(200 + doc.slides.len() as u32);
        doc.slides.push(Arc::new(Slide { id, layout, shapes, ..Default::default() }));
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

    fn card(id: u32, x: f64, y: f64, w: f64, h: f64) -> Shape {
        Shape { id: ShapeId(id), xfrm: Some(rect(x, y, w, h)), kind: ShapeKind::Picture { fill: PictureFill::default() }, ..Default::default() }
    }

    fn solid(id: u32, x: f64, y: f64, w: f64, h: f64) -> Shape {
        Shape { id: ShapeId(id), xfrm: Some(rect(x, y, w, h)), fill: Some(Fill::solid(ColorRef::rgb(Rgba::rgb(80, 60, 120)))), ..Default::default() }
    }

    fn group(id: u32, x: f64, y: f64, w: f64, h: f64, children: Vec<Shape>) -> Shape {
        Shape {
            id: ShapeId(id),
            xfrm: Some(rect(x, y, w, h)),
            kind: ShapeKind::Group { children, child: rect(0.0, 0.0, w, h) },
            ..Default::default()
        }
    }

    fn picture(id: u32, x: f64, y: f64, w: f64, h: f64, alpha: f64) -> Shape {
        Shape {
            id: ShapeId(id),
            xfrm: Some(rect(x, y, w, h)),
            kind: ShapeKind::Picture { fill: PictureFill { alpha: Some(alpha), ..Default::default() } },
            ..Default::default()
        }
    }

    fn solid_group_card(id: u32, x: f64, y: f64, w: f64, h: f64) -> Shape {
        group(id, x, y, w, h, vec![solid(id + 1, 0.0, 0.0, w, h)])
    }

    fn add_card(shapes: &mut Vec<Shape>, base: u32, x: f64, y: f64, heading: &str, body: &str, number: u32) {
        shapes.push(card(base, x, y, 300.0, 220.0));
        let icon = rect(x + 15.0, y + 15.0, 54.0, 54.0);
        shapes.push(group(base + 1, icon.x, icon.y, icon.w, icon.h, vec![solid(base + 2, 0.0, 0.0, 54.0, 54.0)]));
        shapes.push(group(base + 3, icon.x, icon.y, icon.w, icon.h, vec![picture(base + 4, 0.0, 0.0, 54.0, 54.0, 0.0)]));
        shapes.push(group(
            base + 5,
            icon.x,
            icon.y,
            icon.w,
            icon.h,
            vec![picture(base + 6, 0.0, 0.0, 54.0, 54.0, 0.0), text(base + 7, 0.0, 0.0, 54.0, 54.0, &number.to_string())],
        ));
        shapes.push(text(base + 8, x + 70.0, y + 10.0, 220.0, 70.0, heading));
        shapes.push(text(base + 9, x + 20.0, y + 100.0, 260.0, 100.0, body));
    }

    fn slide6_like() -> Presentation {
        let mut doc = deck(1440.0, 810.0);
        let mut shapes = Vec::new();
        shapes.push(group(2, 20.0, 10.0, 1400.0, 60.0, vec![picture(3, 0.0, 0.0, 1400.0, 60.0, 0.0)]));
        shapes.push(group(
            4,
            20.0,
            10.0,
            1400.0,
            60.0,
            vec![picture(5, 0.0, 0.0, 1400.0, 60.0, 0.0), text(6, 0.0, 0.0, 1400.0, 60.0, "Rammed Earth Construction Process")],
        ));
        let cards =
            [(7, 30.0, 80.0), (17, 380.0, 80.0), (27, 730.0, 80.0), (37, 1080.0, 80.0), (47, 1080.0, 360.0), (57, 730.0, 360.0), (67, 380.0, 360.0)];
        for (i, (base, x, y)) in cards.iter().enumerate() {
            add_card(&mut shapes, *base, *x, *y, &format!("Heading {i}"), &format!("Body {i}"), i as u32 + 1);
        }
        for (id, x, y, w, h) in [
            (77, 352.0, 180.0, 20.0, 15.0),
            (79, 702.0, 180.0, 20.0, 15.0),
            (81, 1052.0, 180.0, 20.0, 15.0),
            (83, 350.0, 315.0, 20.0, 60.0),
            (85, 702.0, 460.0, 20.0, 15.0),
            (87, 352.0, 460.0, 20.0, 15.0),
        ] {
            shapes.push(group(id, x, y, w, h, vec![solid(id + 1, 0.0, 0.0, w, h)]));
        }
        shapes.push(text(89, 1400.0, 780.0, 20.0, 18.0, "7"));
        add_slide(&mut doc, shapes);
        doc
    }

    #[test]
    fn slide6_matches_expected_slots() {
        let doc = slide6_like();
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.title_shapes, [4]);
        assert_eq!(scan.visible_title.as_deref(), Some("Rammed Earth Construction Process"));
        assert_eq!(scan.invisible, [2, 5, 10, 13, 20, 23, 30, 33, 40, 43, 50, 53, 60, 63, 70, 73]);
        assert_eq!(scan.rows, [vec![7, 17, 27, 37], vec![47, 57, 67]]);
        assert_eq!(scan.decorations, [77, 79, 81, 83, 85, 87]);
        assert_eq!(scan.unclassified, [89]);
        assert_eq!(scan.candidate_slots.len(), 7);
        assert_eq!(scan.candidate_slots[0].children, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(scan.candidate_slots[0].top_level_members, [7, 8, 10, 12, 15, 16]);
        for (i, slot) in scan.candidate_slots.iter().enumerate() {
            let base = 7 + 10 * i as u32;
            assert_eq!(slot.id, format!("s0-c{i}"));
            assert_eq!(slot.background, base);
            assert_eq!(slot.heading, Some(base + 8));
            assert_eq!(slot.body, Some(base + 9));
            assert_eq!(slot.badge, Some(base + 5));
            assert!(slot.icons.is_empty());
            assert!(slot.flags.is_empty());
            assert!(slot.confidence >= 0.9);
            assert_eq!(slot.children, [base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7, base + 8, base + 9]);
        }
    }

    #[test]
    fn empty_slide_has_no_slots() {
        let mut doc = deck(960.0, 540.0);
        add_slide(&mut doc, vec![]);
        let scan = scan_slide(&doc, 0).unwrap();
        assert!(scan.candidate_slots.is_empty());
        assert!(scan.rows.is_empty());
        assert!(scan.title_shapes.is_empty());
        assert_eq!(scan.visible_title, None);
        assert!(scan.decorations.is_empty());
        assert!(scan.unclassified.is_empty());
        assert!(scan.invisible.is_empty());
    }

    #[test]
    fn single_card_with_no_text() {
        let mut doc = deck(960.0, 540.0);
        add_slide(&mut doc, vec![card(10, 100.0, 100.0, 400.0, 300.0)]);
        let scan = scan_slide(&doc, 0).unwrap();
        let slot = &scan.candidate_slots[0];
        assert_eq!(slot.background, 10);
        assert_eq!(slot.heading, None);
        assert_eq!(slot.body, None);
        assert_eq!(slot.confidence, 0.3);
        assert_eq!(slot.flags, ["no_heading", "no_body"]);
    }

    #[test]
    fn kpi_card_detects_heading_and_body() {
        let mut doc = deck(960.0, 540.0);
        add_slide(
            &mut doc,
            vec![card(10, 100.0, 100.0, 400.0, 300.0), text(11, 150.0, 120.0, 250.0, 60.0, "Revenue"), text(12, 120.0, 260.0, 300.0, 60.0, "42%")],
        );
        let scan = scan_slide(&doc, 0).unwrap();
        let slot = &scan.candidate_slots[0];
        assert_eq!(slot.heading, Some(11));
        assert_eq!(slot.body, Some(12));
        assert_eq!(slot.confidence, 0.9);
        assert!(slot.flags.is_empty());
    }

    #[test]
    fn kpi_card_solid_fill() {
        let mut doc = deck(960.0, 540.0);
        add_slide(&mut doc, vec![solid_group_card(10, 100.0, 100.0, 400.0, 300.0)]);
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.candidate_slots[0].background, 10);
    }

    #[test]
    fn invisible_group_listed() {
        let mut doc = deck(960.0, 540.0);
        add_slide(&mut doc, vec![group(10, 100.0, 100.0, 200.0, 100.0, vec![picture(11, 0.0, 0.0, 200.0, 100.0, 0.0)])]);
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.invisible, [10]);
        assert!(scan.candidate_slots.is_empty());
        assert!(scan.decorations.is_empty());
    }

    #[test]
    fn page_number_unclassified() {
        let mut doc = deck(960.0, 540.0);
        add_slide(&mut doc, vec![text(89, 900.0, 500.0, 20.0, 20.0, "7")]);
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.unclassified, [89]);
        assert!(scan.decorations.is_empty());
    }

    #[test]
    fn arrow_between_cards_is_decoration() {
        let mut doc = deck(960.0, 540.0);
        add_slide(
            &mut doc,
            vec![
                card(10, 50.0, 100.0, 300.0, 300.0),
                card(20, 500.0, 100.0, 300.0, 300.0),
                group(30, 400.0, 180.0, 50.0, 30.0, vec![solid(31, 0.0, 0.0, 50.0, 30.0)]),
            ],
        );
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.decorations, [30]);
    }

    #[test]
    fn two_child_group_with_text_is_badge() {
        let mut doc = deck(960.0, 540.0);
        add_slide(
            &mut doc,
            vec![
                card(10, 100.0, 100.0, 400.0, 300.0),
                group(20, 120.0, 120.0, 54.0, 54.0, vec![picture(21, 0.0, 0.0, 54.0, 54.0, 0.0), text(22, 0.0, 0.0, 54.0, 54.0, "1")]),
            ],
        );
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.candidate_slots[0].badge, Some(20));
    }

    #[test]
    fn overlapping_cards_same_row() {
        let mut doc = deck(960.0, 540.0);
        add_slide(&mut doc, vec![card(10, 50.0, 100.0, 300.0, 300.0), card(20, 400.0, 200.0, 300.0, 300.0)]);
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.rows, [vec![10, 20]]);
    }

    #[test]
    fn slot_children_include_group_descendants() {
        let mut doc = deck(960.0, 540.0);
        add_slide(
            &mut doc,
            vec![
                card(10, 100.0, 100.0, 400.0, 300.0),
                group(20, 120.0, 120.0, 54.0, 54.0, vec![picture(21, 0.0, 0.0, 54.0, 54.0, 0.0), text(22, 0.0, 0.0, 54.0, 54.0, "1")]),
            ],
        );
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.candidate_slots[0].children, [10, 20, 21, 22]);
        assert_eq!(scan.candidate_slots[0].top_level_members, [10, 20]);
    }

    #[test]
    fn title_band_excluded_from_decorations() {
        let mut doc = deck(960.0, 540.0);
        add_slide(
            &mut doc,
            vec![text(10, 40.0, 20.0, 300.0, 40.0, "Title"), group(20, 40.0, 120.0, 100.0, 30.0, vec![solid(21, 0.0, 0.0, 100.0, 30.0)])],
        );
        let scan = scan_slide(&doc, 0).unwrap();
        assert_eq!(scan.title_shapes, [10]);
        assert_eq!(scan.visible_title.as_deref(), Some("Title"));
        assert_eq!(scan.decorations, [20]);
    }
}
