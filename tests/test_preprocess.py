"""Unit tests for the pure-Python parts of the corpus pipeline."""

import pytest

from app.preprocess import (
    PAPERS_DIR,
    attach_metadata,
    clean_markdown,
    segment_markdown,
    split_llm_output,
)

LONG = "This sentence is repeated to make the chunk long enough to survive the fragment filter. " * 3


def test_split_llm_output_splits_on_blank_lines_and_drops_fragments():
    llm_output = f"## Abstract\n{LONG}\n\n\n{LONG}\n\n\n\ntoo short"
    chunks = split_llm_output(llm_output)
    assert len(chunks) == 2                      # fragment under 120 chars is dropped
    assert chunks[0].startswith("## Abstract")
    assert "\n\n\n" not in chunks[0]


def test_attach_metadata_tracks_sections_and_required_fields():
    paper = {"source_id": "lora", "title": "LoRA", "topic": "fine-tuning",
             "source_url": "https://arxiv.org/abs/2106.09685"}
    records = attach_metadata(paper, [
        f"## Abstract\n{LONG}",
        LONG,                        # no heading: inherits the previous section
        f"## Method\n{LONG}",
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


def test_segment_markdown_respects_size_and_keeps_all_text():
    paragraphs = [f"Paragraph number {i}. " * 20 for i in range(30)]
    document = "\n\n".join(paragraphs)
    segments = segment_markdown(document, max_chars=2000)
    assert len(segments) > 1
    assert all(len(s) <= 2500 for s in segments)
    assert "".join(segments).replace("\n", "") == document.replace("\n", "")


@pytest.mark.skipif(not list(PAPERS_DIR.glob("*.pdf")), reason="no downloaded papers yet")
def test_pdf_converts_to_usable_text():
    import re

    from app.preprocess import pdf_to_text

    pdf = sorted(PAPERS_DIR.glob("*.pdf"))[0]
    cleaned = clean_markdown(pdf_to_text(pdf))
    assert len(cleaned) > 5000                   # a real paper, not an error page
    assert "abstract" in cleaned.lower()
    # extraction must preserve word spacing (glued 25+ letter runs signal damage)
    glued = len(re.findall(r"[A-Za-z]{25,}", cleaned))
    assert glued < len(cleaned) / 5000
