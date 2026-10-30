pub mod plan;
pub mod scan;

#[cfg(not(target_arch = "wasm32"))]
mod nim;

pub use plan::{Plan, PlanRequest, PlanSlide};
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
