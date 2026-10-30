use serde::{Deserialize, Serialize};

use super::scan::SlideScan;

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

/// A content slide after local outline parsing. The model never segments the user's text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutlineSlide {
    pub title: String,
    #[serde(default)]
    pub bullets: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplatePlan {
    pub slides: Vec<TemplatePlanSlide>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplatePlanSlide {
    pub index: usize,
    pub keep_slot_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fills: Vec<SlotFill>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotFill {
    pub slot_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct TemplatePlanRequest<'a> {
    pub content: &'a str,
    pub slides_from_scan: &'a [SlideScan],
    pub model: Option<&'a str>,
}

/// Parse a loose markdown outline into content slides before the model sees it.
///
/// The canonical form is `# Slide 1: Title` followed by `- bullets`. We also accept
/// `#`/`##` headings, `Slide N:` lines with or without a leading `#`, and, when there
/// are no headings, blank-line-separated paragraphs (first line is the title).
pub fn parse_outline(content: &str) -> Vec<OutlineSlide> {
    let content = content.strip_prefix('\u{feff}').unwrap_or(content);
    if content.lines().any(|line| heading_text(line).is_some()) { parse_with_headings(content) } else { parse_paragraphs(content) }
}

fn parse_with_headings(content: &str) -> Vec<OutlineSlide> {
    let mut out = Vec::new();
    let mut current: Option<OutlineSlide> = None;
    for line in content.lines() {
        if line.trim().is_empty() || marker_only(line) {
            continue;
        }
        if let Some(title) = heading_text(line) {
            push_slide(&mut out, current.take());
            current = Some(OutlineSlide { title: title.to_string(), bullets: Vec::new() });
        } else if let Some(text) = bullet_text(line) {
            if !text.is_empty() {
                current.get_or_insert_with(|| OutlineSlide { title: String::new(), bullets: Vec::new() }).bullets.push(text.to_string());
            }
        } else if let Some(slide) = current.as_mut() {
            slide.bullets.push(line.trim().to_string());
        } else {
            current = Some(OutlineSlide { title: line.trim().to_string(), bullets: Vec::new() });
        }
    }
    push_slide(&mut out, current);
    out
}

fn parse_paragraphs(content: &str) -> Vec<OutlineSlide> {
    let mut out = Vec::new();
    let mut paragraph: Vec<&str> = Vec::new();
    for line in content.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !paragraph.is_empty() {
                push_paragraph(&mut out, &paragraph);
                paragraph.clear();
            }
        } else if !marker_only(line) {
            paragraph.push(line.trim());
        }
    }
    out
}

fn push_paragraph(out: &mut Vec<OutlineSlide>, lines: &[&str]) {
    let Some(first) = lines.first() else { return };
    let mut title = bullet_text(first).unwrap_or(first).trim().to_string();
    let mut bullets: Vec<String> = lines
        .iter()
        .skip(1)
        .filter_map(|line| {
            let text = bullet_text(line).unwrap_or(line).trim();
            (!text.is_empty()).then(|| text.to_string())
        })
        .collect();
    if title.is_empty() && !bullets.is_empty() {
        title = bullets.remove(0);
    }
    push_slide(out, Some(OutlineSlide { title, bullets }));
}

fn push_slide(out: &mut Vec<OutlineSlide>, slide: Option<OutlineSlide>) {
    let Some(mut slide) = slide else { return };
    if slide.title.is_empty() && !slide.bullets.is_empty() {
        slide.title = slide.bullets.remove(0);
    }
    if !slide.title.is_empty() || !slide.bullets.is_empty() {
        out.push(slide);
    }
}

fn heading_text(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let hashes = line.bytes().take_while(|b| *b == b'#').count();
    let rest = if hashes == 0 { line } else { line[hashes..].trim() };
    let title = if hashes == 0 { strip_slide_prefix(rest)? } else { strip_slide_prefix(rest).unwrap_or(rest) };
    let title = clean_heading(title);
    (!title.is_empty()).then_some(title)
}

fn strip_slide_prefix(s: &str) -> Option<&str> {
    let s = s.trim();
    let head = s.get(..5)?;
    if !head.eq_ignore_ascii_case("slide") {
        return None;
    }
    let rest = s[5..].trim_start();
    let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    let rest = rest[digits..].trim_start();
    let rest = rest.strip_prefix(':')?;
    Some(rest.trim())
}

fn clean_heading(title: &str) -> &str {
    let title = title.trim();
    let without = title.trim_end_matches('#');
    if without.len() < title.len() && without.chars().next_back().is_some_and(char::is_whitespace) { without.trim_end() } else { title }
}

fn marker_only(line: &str) -> bool {
    let line = line.trim();
    let hashes = line.bytes().take_while(|b| *b == b'#').count();
    if hashes > 0 && line[hashes..].trim().is_empty() {
        return true;
    }
    bullet_text(line).is_some_and(str::is_empty)
}

fn bullet_text(line: &str) -> Option<&str> {
    let line = line.trim_start();
    let first = line.chars().next()?;
    if matches!(first, '-' | '*' | '+' | '•' | '·') {
        let rest = &line[first.len_utf8()..];
        return (rest.is_empty() || starts_space(rest)).then_some(rest.trim_start());
    }
    let digits = line.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let rest = line[digits..].strip_prefix('.').or_else(|| line[digits..].strip_prefix(')'))?;
    (rest.is_empty() || starts_space(rest)).then_some(rest.trim_start())
}

fn starts_space(s: &str) -> bool {
    s.starts_with(' ') || s.starts_with('\t')
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn slide(title: &str, bullets: &[&str]) -> OutlineSlide {
        OutlineSlide { title: title.into(), bullets: bullets.iter().map(|s| (*s).into()).collect() }
    }

    #[test]
    fn parses_the_canonical_heading_and_bullets() {
        let content = "# Slide 1: Why now\n- Market shift\n- Customer pain\n\n## Slide 2: Next\n* Ship beta\n";
        assert_eq!(parse_outline(content), [slide("Why now", &["Market shift", "Customer pain"]), slide("Next", &["Ship beta"])]);
    }

    #[test]
    fn accepts_numbered_and_bare_slide_headings() {
        let content = "Slide 1: One\n- a\n# Two\n- b\n# Slide: Three\n- c\nslide 4: Four\n- d\n";
        assert_eq!(parse_outline(content), [slide("One", &["a"]), slide("Two", &["b"]), slide("Three", &["c"]), slide("Four", &["d"])]);
    }

    #[test]
    fn accepts_common_bullet_markers_and_loose_lines() {
        let content = "# Slide 1: Loose\n * one\n+ two\n1. three\n2) four\nplain line\n";
        assert_eq!(parse_outline(content), [slide("Loose", &["one", "two", "three", "four", "plain line"])]);
    }

    #[test]
    fn falls_back_to_blank_line_separated_paragraphs() {
        let content = "Why now\nMarket shift\nCustomer pain\n\nNext\n- Ship beta\n- Hire two AEs\n";
        assert_eq!(parse_outline(content), [slide("Why now", &["Market shift", "Customer pain"]), slide("Next", &["Ship beta", "Hire two AEs"])]);
    }

    #[test]
    fn strips_a_bom_and_closing_heading_hashes() {
        let content = "\u{feff}# Slide 1: Title #\r\n- point\r\n";
        assert_eq!(parse_outline(content), [slide("Title", &["point"])]);
    }

    #[test]
    fn empty_or_marker_only_content_yields_no_slides() {
        assert!(parse_outline("").is_empty());
        assert!(parse_outline("   \n\n#\n- \n").is_empty());
    }
}
