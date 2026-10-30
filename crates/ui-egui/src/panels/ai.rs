use super::{SlideApp, Tokens, Ui};

/// AI PPT Maker: prompt-driven deck generation. The flow lands here in Phase 4.
pub(super) fn ai(_app: &mut SlideApp, ui: &mut Ui) {
    let t = Tokens::get(ui.ctx());
    ui.label(egui::RichText::new("Turn a prompt into a slide deck. Generation is not connected yet.").color(t.text_dim));
    ui.add_space(8.0);
    ui.add_enabled(false, egui::Button::new("Generate"));
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use egui_kittest::Harness;
    use egui_kittest::kittest::{NodeT, Queryable};

    use crate::{Services, SlideApp};

    #[test]
    fn the_pane_shows_its_title_and_a_disabled_generate_button() {
        let mut app = SlideApp::new(deckcraft_engine::Session::with_new(), Services::default());
        app.ui.pane = Some("ai_ppt_maker".into());
        let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_ui_state(
            |ui, app: &mut SlideApp| {
                app.logic(ui.ctx());
                app.ui(ui);
            },
            app,
        );
        h.run_steps(3);
        assert!(h.query_by_label("AI PPT Maker").is_some());
        let generate = h.get_by_role_and_label(egui::accesskit::Role::Button, "Generate");
        assert!(generate.accesskit_node().is_disabled());
    }
}
