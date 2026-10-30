pub mod plan;

#[cfg(not(target_arch = "wasm32"))]
mod nim;

pub use plan::{Plan, PlanRequest, PlanSlide};

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
