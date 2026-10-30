pub mod apply;
pub mod plan;
pub mod scan;

#[cfg(not(target_arch = "wasm32"))]
mod nim;

pub use apply::{ApplyPlan, ApplyResult, ApplySlide, RealignMode, SlideResult, apply_plan};
pub use plan::{OutlineSlide, Plan, PlanRequest, PlanSlide, SlotFill, TemplatePlan, TemplatePlanRequest, TemplatePlanSlide, parse_outline};
pub use scan::{CandidateSlot, SlideScan, scan_presentation, scan_slide};

pub fn plan(req: PlanRequest<'_>) -> crate::Result<Plan> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        nim::plan(req)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = req;
        Err(crate::EngineError::Other("AI planning is not available in the browser".into()))
    }
}

pub fn plan_template(req: TemplatePlanRequest<'_>) -> crate::Result<TemplatePlan> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        nim::plan_template(req)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = req;
        Err(crate::EngineError::Other("AI planning is not available in the browser".into()))
    }
}
