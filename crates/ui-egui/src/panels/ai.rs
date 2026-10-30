use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use deckcraft_engine::ai::TemplatePlan;
#[cfg(not(target_arch = "wasm32"))]
use deckcraft_engine::ai::TemplatePlanRequest;
use deckcraft_model::Presentation;

use super::{SlideApp, Tokens, Ui, theme};

/// The AI PPT Maker pane's outline and in-flight planning job.
#[derive(Default)]
pub(crate) struct State {
    pub(crate) content: String,
    pub(crate) error: Option<String>,
    pub(crate) summary: Option<String>,
    job: Option<Job>,
}

impl State {
    pub(crate) fn busy(&self) -> bool {
        self.job.is_some()
    }
}

struct Job {
    uid: u64,
    doc: Arc<Presentation>,
    rx: Receiver<Result<TemplatePlan, String>>,
}

/// AI PPT Maker: describe the content, then fill the open presentation's card layout from it.
pub(super) fn ai(app: &mut SlideApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.label(
        egui::RichText::new("Describe what the deck should say. The open presentation supplies the layout, and AI fills its cards.")
            .color(t.text_dim),
    );
    ui.add_space(8.0);

    let busy = app.ai.busy();
    let label = ui.label(egui::RichText::new("Content").font(theme::bold(12.0)));
    ui.add_enabled(
        !busy,
        egui::TextEdit::multiline(&mut app.ai.content)
            .desired_width(f32::INFINITY)
            .desired_rows(10)
            .hint_text("# Slide 1: Why now\n- Market shift\n- Customer pain"),
    )
    .labelled_by(label.id);
    ui.add_space(4.0);
    ui.label(egui::RichText::new("An outline works best: one heading per slide with bullet points under it.").size(11.5).color(t.text_dim));
    ui.add_space(8.0);

    if busy {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(16.0));
            ui.label("Planning with AI...");
            if ui.button("Cancel").clicked() {
                app.cancel_ai_plan();
            }
        });
        ui.label(egui::RichText::new("This can take up to 30 seconds.").size(11.5).color(t.text_dim));
    } else {
        let ready = !app.ai.content.trim().is_empty() && app.session.active().is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("Generate")).clicked() {
                let ctx = ui.ctx().clone();
                app.start_ai_plan(&ctx);
            }
            if ui.button("Clear").clicked() {
                app.ai.content.clear();
                app.ai.error = None;
                app.ai.summary = None;
            }
        });
    }

    if let Some(e) = &app.ai.error {
        ui.add_space(8.0);
        ui.label(egui::RichText::new(e).color(egui::Color32::from_rgb(0xD8, 0x3F, 0x6B)));
    }
    if let Some(s) = &app.ai.summary {
        ui.add_space(8.0);
        ui.label(egui::RichText::new(s).color(egui::Color32::from_rgb(0x2E, 0xA3, 0x6F)));
    }

    ui.add_space(10.0);
    ui.label(
        egui::RichText::new("The open presentation is the layout template. AI fills its cards, removes unused shapes, and leaves the rest as it is.")
            .size(11.5)
            .color(t.text_faint),
    );
}

impl SlideApp {
    /// Start planning the open presentation's template against the pane's content.
    pub(crate) fn start_ai_plan(&mut self, ctx: &egui::Context) {
        let content = self.ai.content.trim().to_string();
        if content.is_empty() || self.ai.job.is_some() {
            return;
        }
        let Some(st) = self.session.active() else {
            self.ai.error = Some("Open a presentation to use as the layout template.".into());
            return;
        };
        let uid = st.uid;
        let doc = st.doc.clone();
        let planner = self.services.plan_template.clone();
        self.ai.error = None;
        self.ai.summary = None;

        #[cfg(target_arch = "wasm32")]
        {
            let r = match planner {
                Some(f) => f(content, doc.clone()),
                None => default_ai_plan(&content, &doc),
            };
            self.finish_ai_plan(uid, doc, r);
            ctx.request_repaint();
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let worker_ctx = ctx.clone();
            let worker_doc = doc.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            let spawned = std::thread::Builder::new().name("deckcraft-ai".into()).spawn(move || {
                let r = match planner {
                    Some(f) => f(content, worker_doc.clone()),
                    None => default_ai_plan(&content, &worker_doc),
                };
                let _ = tx.send(r);
                worker_ctx.request_repaint();
            });
            match spawned {
                Ok(_) => {
                    self.ai.job = Some(Job { uid, doc, rx });
                    self.set_status("Planning the deck with AI...");
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(e) => {
                    self.ai.error = Some(format!("Could not start AI planning: {e}"));
                    self.set_status("AI planning failed.");
                }
            }
        }
    }

    /// Stop waiting for the current plan; the presentation is left untouched.
    pub(crate) fn cancel_ai_plan(&mut self) {
        if self.ai.job.take().is_some() {
            self.ai.error = None;
            self.ai.summary = None;
            self.set_status("Stopped AI planning.");
        }
    }

    /// Apply a plan once its worker finishes, unless its document has moved on since.
    pub(crate) fn poll_ai_plan(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.ai.job else { return };
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        let r = match job.rx.try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("AI planning stopped unexpectedly.".into()),
        };
        let Some(job) = self.ai.job.take() else { return };
        self.finish_ai_plan(job.uid, job.doc, r);
    }

    fn finish_ai_plan(&mut self, uid: u64, doc: Arc<Presentation>, r: Result<TemplatePlan, String>) {
        let plan = match r {
            Ok(plan) => plan,
            Err(e) => {
                self.ai.error = Some(e);
                self.set_status("AI planning failed.");
                return;
            }
        };
        let unchanged = self.session.active().is_some_and(|st| st.uid == uid && Arc::ptr_eq(&st.doc, &doc));
        if !unchanged {
            self.ai.error = Some("The presentation changed while the AI was planning. Generate again.".into());
            self.set_status("The AI plan was discarded because the presentation changed.");
            return;
        }
        let plan = match serde_json::to_value(plan) {
            Ok(plan) => plan,
            Err(e) => {
                self.ai.error = Some(format!("Could not send the AI plan: {e}"));
                self.set_status("AI planning failed.");
                return;
            }
        };
        match self.session.execute("ai.applyTemplatePlan", &serde_json::json!({ "plan": plan })) {
            Ok(result) => {
                let summary = summarize(&result);
                self.ai.summary = Some(summary.clone());
                self.set_status(summary);
            }
            Err(e) => {
                self.ai.error = Some(format!("Could not apply the AI layout: {e}"));
                self.set_status("Could not apply the AI layout.");
            }
        }
    }
}

fn summarize(result: &serde_json::Value) -> String {
    let slides = result.get("slides").and_then(|v| v.as_array());
    let count = slides.map_or(0, Vec::len);
    let filled: u64 = slides.into_iter().flatten().filter_map(|s| s.get("filled").and_then(|v| v.as_u64())).sum();
    let deleted: u64 = slides.into_iter().flatten().filter_map(|s| s.get("deleted").and_then(|v| v.as_u64())).sum();
    format!("Applied to {count} slide{}: {filled} slot{} filled, {deleted} shape{} removed.", plural(count as u64), plural(filled), plural(deleted))
}

fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

#[cfg(not(target_arch = "wasm32"))]
fn default_ai_plan(content: &str, doc: &Presentation) -> Result<TemplatePlan, String> {
    let slides_from_scan = deckcraft_engine::ai::scan_presentation(doc);
    deckcraft_engine::ai::plan_template(TemplatePlanRequest { content, slides_from_scan: &slides_from_scan, model: None }).map_err(|e| e.to_string())
}

#[cfg(target_arch = "wasm32")]
fn default_ai_plan(_content: &str, _doc: &Presentation) -> Result<TemplatePlan, String> {
    Err("AI planning is not available in the browser".into())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::sync::Mutex;
    use std::sync::mpsc::channel;
    use std::thread::ThreadId;

    use egui_kittest::Harness;
    use egui_kittest::kittest::{NodeT, Queryable};

    use super::*;
    use crate::{AiPlanFn, Services};
    use deckcraft_engine::Session;
    use deckcraft_engine::ai::{SlotFill, TemplatePlanSlide};
    use deckcraft_model::{ColorRef, Fill, LayoutId, Rgba, Shape, ShapeId, ShapeKind, Slide, SlideId, TextBody, Xfrm};

    fn harness(app: SlideApp) -> Harness<'static, SlideApp> {
        Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_ui_state(
            |ui, app: &mut SlideApp| {
                app.logic(ui.ctx());
                app.ui(ui);
            },
            app,
        )
    }

    fn app_with_deck(services: Services) -> SlideApp {
        let mut session = Session::with_new();
        let doc = card_deck();
        let st = session.active_mut().expect("a new presentation");
        st.doc = Arc::new(doc);
        st.saved_doc = st.doc.clone();
        let mut app = SlideApp::new(session, services);
        app.ui.pane = Some("ai_ppt_maker".into());
        app
    }

    fn card_deck() -> Presentation {
        let mut doc = Presentation::default();
        doc.slides.clear();
        let layout = doc.masters.first().and_then(|m| m.layouts.first()).map(|l| l.id).unwrap_or(LayoutId(0));
        let card = Shape {
            id: ShapeId(10),
            xfrm: Some(Xfrm::new(100.0, 100.0, 500.0, 300.0)),
            kind: ShapeKind::Group {
                children: vec![Shape {
                    id: ShapeId(11),
                    xfrm: Some(Xfrm::new(0.0, 0.0, 500.0, 300.0)),
                    fill: Some(Fill::solid(ColorRef::rgb(Rgba::rgb(80, 60, 120)))),
                    ..Default::default()
                }],
                child: Xfrm::new(0.0, 0.0, 500.0, 300.0),
            },
            ..Default::default()
        };
        let heading = Shape {
            id: ShapeId(12),
            xfrm: Some(Xfrm::new(150.0, 140.0, 300.0, 60.0)),
            text: Some(TextBody::from_text("Old heading")),
            ..Default::default()
        };
        let body = Shape {
            id: ShapeId(13),
            xfrm: Some(Xfrm::new(150.0, 270.0, 300.0, 100.0)),
            text: Some(TextBody::from_text("Old body")),
            ..Default::default()
        };
        doc.slides.push(std::sync::Arc::new(Slide { id: SlideId(300), layout, shapes: vec![card, heading, body], ..Default::default() }));
        doc
    }

    fn plan_for(doc: &Presentation) -> Result<TemplatePlan, String> {
        let scans = deckcraft_engine::ai::scan_presentation(doc);
        let slide = scans.first().ok_or("no slide")?;
        let slot = slide.candidate_slots.first().ok_or("no slot")?;
        if slot.heading.is_none() || slot.body.is_none() {
            return Err(format!("the slot has no heading or body: {slot:?}"));
        }
        Ok(TemplatePlan {
            slides: vec![TemplatePlanSlide {
                index: 0,
                keep_slot_ids: vec![slot.id.clone()],
                fills: vec![SlotFill { slot_id: slot.id.clone(), heading: Some("Why now".into()), body: Some("Market shift\nCustomer pain".into()) }],
            }],
        })
    }

    fn shape_text(doc: &Presentation, id: u32) -> String {
        doc.slides[0].shapes.iter().find(|s| s.id.0 == id).and_then(|s| s.text.as_ref()).map(TextBody::text).unwrap_or_default()
    }

    fn wait_for_plan(h: &mut Harness<'static, SlideApp>) {
        for _ in 0..400 {
            h.run_steps(1);
            if !h.state().ai.busy() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("AI planning did not finish");
    }

    #[test]
    fn the_pane_shows_its_title_and_a_disabled_generate_button() {
        let mut app = SlideApp::new(Session::with_new(), Services::default());
        app.ui.pane = Some("ai_ppt_maker".into());
        let mut h = harness(app);
        h.run_steps(3);
        assert!(h.query_by_label("AI PPT Maker").is_some());
        let generate = h.get_by_role_and_label(egui::accesskit::Role::Button, "Generate");
        assert!(generate.accesskit_node().is_disabled());
    }

    #[test]
    fn generate_plans_on_a_worker_and_applies_the_plan() {
        let (release_tx, release_rx) = channel();
        let go = Arc::new(Mutex::new(release_rx));
        let seen = Arc::new(Mutex::new(None::<ThreadId>));
        let worker_seen = seen.clone();
        let worker_go = go.clone();
        let planner: Arc<AiPlanFn> = Arc::new(move |_content, doc| {
            *worker_seen.lock().expect("worker thread id") = Some(std::thread::current().id());
            worker_go.lock().expect("worker release").recv().expect("worker release");
            plan_for(&doc)
        });
        let mut app = app_with_deck(Services { plan_template: Some(planner), ..Default::default() });
        app.ai.content = "# Slide 1: Why now\n- Market shift\n- Customer pain".into();
        let undo_before = app.session.active().expect("active presentation").history.undo.len();
        let mut h = harness(app);
        h.run_steps(3);
        h.get_by_role_and_label(egui::accesskit::Role::Button, "Generate").click();
        h.run_steps(1);
        assert!(h.state().ai.busy(), "the worker should still be planning");
        assert_eq!(shape_text(h.state().session.active().expect("active presentation").doc.as_ref(), 12), "Old heading");

        release_tx.send(()).expect("release the worker");
        wait_for_plan(&mut h);

        let worker = seen.lock().expect("worker thread id").expect("the worker ran");
        assert_ne!(worker, std::thread::current().id(), "planning must not run on the UI thread");
        assert_eq!(h.state().ai.error, None);
        let summary = h.state().ai.summary.clone().expect("a summary");
        assert!(summary.contains("1 slide"), "{summary}");
        assert!(summary.contains("1 slot filled"), "{summary}");
        assert!(summary.contains("0 shapes removed"), "{summary}");
        let doc = h.state().session.active().expect("active presentation").doc.clone();
        assert_eq!(shape_text(&doc, 12), "Why now");
        assert_eq!(shape_text(&doc, 13), "Market shift\nCustomer pain");
        assert_eq!(h.state().session.active().expect("active presentation").history.undo.len(), undo_before + 1);

        h.state_mut().session.execute("edit.undo", &serde_json::json!({})).expect("undo the AI fill");
        let doc = h.state().session.active().expect("active presentation").doc.clone();
        assert_eq!(shape_text(&doc, 12), "Old heading");
    }

    #[test]
    fn generate_shows_a_planner_error_without_changing_the_deck() {
        let planner: Arc<AiPlanFn> = Arc::new(|_content, _doc| Err("NVIDIA_API_KEY is not set".into()));
        let mut app = app_with_deck(Services { plan_template: Some(planner), ..Default::default() });
        app.ai.content = "# Slide 1: Why now".into();
        let undo_before = app.session.active().expect("active presentation").history.undo.len();
        let mut h = harness(app);
        h.run_steps(3);
        h.get_by_role_and_label(egui::accesskit::Role::Button, "Generate").click();
        wait_for_plan(&mut h);

        let error = h.state().ai.error.clone().expect("an error");
        assert!(error.contains("NVIDIA_API_KEY"), "{error}");
        assert_eq!(h.state().ai.summary, None);
        let doc = h.state().session.active().expect("active presentation").doc.clone();
        assert_eq!(shape_text(&doc, 12), "Old heading");
        assert_eq!(h.state().session.active().expect("active presentation").history.undo.len(), undo_before);
    }

    #[test]
    fn a_changed_presentation_discards_the_plan() {
        let (release_tx, release_rx) = channel();
        let go = Arc::new(Mutex::new(release_rx));
        let planner: Arc<AiPlanFn> = {
            let worker_go = go.clone();
            Arc::new(move |_content, doc| {
                worker_go.lock().expect("worker release").recv().expect("worker release");
                plan_for(&doc)
            })
        };
        let mut app = app_with_deck(Services { plan_template: Some(planner), ..Default::default() });
        app.ai.content = "# Slide 1: Why now".into();
        let mut h = harness(app);
        h.run_steps(3);
        h.get_by_role_and_label(egui::accesskit::Role::Button, "Generate").click();
        h.run_steps(2);
        assert!(h.state().ai.busy());

        let st = h.state_mut().session.active_mut().expect("active presentation");
        st.doc = Arc::new((*st.doc).clone());
        release_tx.send(()).expect("release the worker");
        wait_for_plan(&mut h);

        let error = h.state().ai.error.clone().expect("a stale-plan error");
        assert!(error.contains("presentation changed"), "{error}");
        assert_eq!(h.state().ai.summary, None);
        let doc = h.state().session.active().expect("active presentation").doc.clone();
        assert_eq!(shape_text(&doc, 12), "Old heading");
    }

    #[test]
    fn cancel_stops_waiting_without_applying_the_plan() {
        let (release_tx, release_rx) = channel();
        let go = Arc::new(Mutex::new(release_rx));
        let planner: Arc<AiPlanFn> = {
            let worker_go = go.clone();
            Arc::new(move |_content, doc| {
                worker_go.lock().expect("worker release").recv().expect("worker release");
                plan_for(&doc)
            })
        };
        let mut app = app_with_deck(Services { plan_template: Some(planner), ..Default::default() });
        app.ai.content = "# Slide 1: Why now".into();
        let undo_before = app.session.active().expect("active presentation").history.undo.len();
        let mut h = harness(app);
        h.run_steps(3);
        h.get_by_role_and_label(egui::accesskit::Role::Button, "Generate").click();
        h.run_steps(2);
        assert!(h.state().ai.busy());
        h.get_by_label("Cancel").click();
        h.run_steps(1);
        assert!(!h.state().ai.busy(), "Cancel stops waiting");
        release_tx.send(()).expect("release the worker");
        std::thread::sleep(std::time::Duration::from_millis(50));
        h.run_steps(3);

        let doc = h.state().session.active().expect("active presentation").doc.clone();
        assert_eq!(shape_text(&doc, 12), "Old heading");
        assert_eq!(h.state().session.active().expect("active presentation").history.undo.len(), undo_before);
        assert_eq!(h.state().ai.error, None);
    }
}
