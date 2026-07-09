"""Unit tests for the pure-Python parts of the corpus pipeline."""

import re

import pytest

from app.preprocess import (
    PAPERS_DIR,
    attach_metadata,
    chunk_markdown,
    clean_markdown,
)

LONG = "This sentence is repeated to make the section long enough to be split. " * 30
SHORT = "One tight paragraph that easily fits inside a single chunk on its own. " * 2


def test_chunk_markdown_splits_at_headings():
    text = f"## Abstract\n{SHORT}\n\n## Method\n{SHORT}"
    chunks = chunk_markdown(text, max_chars=1200, overlap=150)
    assert len(chunks) == 2
    assert chunks[0].startswith("## Abstract")
    assert chunks[1].startswith("## Method")


def test_chunk_markdown_subsplits_long_sections_with_overlap():
    text = f"## Experiments\n{LONG}"
    chunks = chunk_markdown(text, max_chars=1200, overlap=150)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)
    assert chunks[0].startswith("## Experiments")


def test_chunk_markdown_drops_fragments_and_keeps_preamble():
    text = f"Preamble before any heading. {SHORT}\n\n## A\nshort"
    chunks = chunk_markdown(text, max_chars=1200, overlap=150)
    assert len(chunks) == 1                      # "## A\nshort" is under 120 chars -> dropped
    assert chunks[0].startswith("Preamble")


def test_infer_headings_marks_arxiv_section_titles():
    from app.preprocess import infer_headings

    text = ("Abstract\nWe propose a method.\n"
            "1 Introduction\nLarge models are large.\n"
            "3.1 Low-Rank Updates\nDetails here.\n"
            "1 I NTRODUCTION\nsmall-caps artifact still detected.\n"
            "We propose a decomposition method here.\n"
            "1 Buy two eggs\n2 Boil the eggs\nprose after the list.\n"
            "# datasets: 7 5 3 4\n"
            "Appendix A\nextra material")
    result = infer_headings(text)
    assert "## Abstract" in result
    assert "## 1 Introduction" in result
    assert "## 3.1 Low-Rank Updates" in result
    assert "## 1 I NTRODUCTION" in result          # letters-only matching
    assert "## Appendix A" in result
    assert "## We propose" not in result           # prose lines stay untouched
    assert "## 1 Buy two eggs" not in result       # consecutive numbered lines = a list
    assert "# datasets" not in result              # stray '#' table artifact neutralized


def test_clean_markdown_cuts_smallcaps_references():
    from app.preprocess import clean_markdown

    text = "Intro. " * 100 + "\nR EFERENCES\n[1] a citation that should disappear"
    cleaned = clean_markdown(text)
    assert "citation" not in cleaned


def test_attach_metadata_tracks_sections_and_required_fields():
    paper = {"source_id": "lora", "title": "LoRA", "topic": "fine-tuning",
             "source_url": "https://arxiv.org/abs/2106.09685"}
    records = attach_metadata(paper, [
        f"## Abstract\n{SHORT}",
        SHORT,                       # no heading: inherits the previous section
        f"## Method\n{SHORT}",
    ])
    assert [r["chunk_id"] for r in records] == ["lora#000", "lora#001", "lora#002"]
    assert [r["section"] for r in records] == ["Abstract", "Abstract", "Method"]
    for record in records:
        for field in ("source_id", "title", "section", "source_url", "chunk_index", "text"):
            assert record[field] is not None


def test_clean_markdown_cuts_references_and_control_chars():
    text = "Intro paragraph. " * 50 + "\n\n# References\n[1] Some citation.\n[2] Another."
    cleaned = clean_markdown("bad\x00char " + text)
    assert "References" not in cleaned
    assert "Some citation" not in cleaned
    assert "\x00" not in cleaned


@pytest.mark.skipif(not list(PAPERS_DIR.glob("*.pdf")), reason="no downloaded papers yet")
def test_pdf_extracts_to_usable_text():
    from app.preprocess import pdf_to_text

    pdf = sorted(PAPERS_DIR.glob("*.pdf"))[0]
    cleaned = clean_markdown(pdf_to_text(pdf))
    assert len(cleaned) > 5000                   # a real paper, not an error page
    assert "abstract" in cleaned.lower()
    # extraction must preserve word spacing (glued 25+ letter runs signal damage)
    glued = len(re.findall(r"[A-Za-z]{25,}", cleaned))
    assert glued < len(cleaned) / 5000
