use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub slides: Vec<PlanSlide>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSlide {
    pub title: String,
    #[serde(default)]
    pub bullets: Vec<String>,
    #[serde(default)]
    pub layout: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanRequest<'a> {
    pub prompt: &'a str,
    pub slides: Option<usize>,
    pub theme: Option<&'a str>,
    pub model: Option<&'a str>,
}
