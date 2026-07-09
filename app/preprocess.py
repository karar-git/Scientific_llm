"""Corpus pipeline: download arXiv PDFs -> Markdown (marker) -> heading-based chunks -> vector store.

Run everything:      python -m app.preprocess
Smoke run:           python -m app.preprocess --limit 2
Redo from scratch:   python -m app.preprocess --force
Redo some papers:    python -m app.preprocess --stage chunk --redo fid,qlora

Each stage is resumable: papers already downloaded, converted, or chunked are
skipped unless --force is given. Only the embedding stage needs an API key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

from app.models import PROJECT_ROOT, env, env_flag, get_vector_store, manifest_path, chunks_path, reset_chunk_cache

PAPERS_DIR = PROJECT_ROOT / "papers"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"

HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
HEADING_LINE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
MIN_CHUNK_CHARS = 120


def _letters(s: str) -> str:
    """Lowercase letters only - makes matching immune to pypdf's small-caps
    artifacts ('R EFERENCES', '1 I NTRODUCTION') and stray punctuation."""
    return re.sub(r"[^a-z]", "", s.lower())


def load_manifest(limit: int | None = None) -> list[dict]:
    papers = json.loads(manifest_path().read_text(encoding="utf-8"))
    for paper in papers:
        paper["source_url"] = f"https://arxiv.org/abs/{paper['arxiv_id']}"
    return papers[:limit] if limit else papers


# ---------------------------------------------------------------------------
# Stage 1: download PDFs
# ---------------------------------------------------------------------------

def download_papers(papers: list[dict], force: bool = False, attempts: int = 3) -> None:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with httpx.Client(follow_redirects=True, timeout=120, headers={"User-Agent": "swibit-rag-task/1.0"}) as client:
        for paper in papers:
            target = PAPERS_DIR / f"{paper['source_id']}.pdf"
            if target.exists() and not force:
                continue
            url = f"https://arxiv.org/pdf/{paper['arxiv_id']}"
            for attempt in range(1, attempts + 1):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    if not response.content.startswith(b"%PDF"):
                        raise RuntimeError(f"{url} did not return a PDF")
                    target.write_bytes(response.content)
                    print(f"downloaded {target.name} ({len(response.content) // 1024} KB)")
                    break
                except Exception as exc:
                    print(f"attempt {attempt}/{attempts} failed for {paper['source_id']}: {exc}")
            else:
                failed.append(paper["source_id"])
    if failed:
        sys.exit(f"could not download: {', '.join(failed)} - rerun to retry just these")


# ---------------------------------------------------------------------------
# Stage 2: PDF -> Markdown. marker (deep-learning layout analysis) produces
# real markdown with headings, which the chunker relies on; pypdf is the
# plain-text fallback when marker is not installed.
# ---------------------------------------------------------------------------

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def pdf_to_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin words hyphen-broken across lines
    # cut everything after the last references/bibliography line (scanned from the
    # end; letters-only match survives small-caps spacing like 'R EFERENCES')
    lines = text.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if len(lines[i]) < 40 and _letters(lines[i]) in {"references", "bibliography"}:
            if sum(len(l) + 1 for l in lines[:i]) > len(text) * 0.4:
                text = "\n".join(lines[:i])
            break
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


SECTION_KEYS = {
    "abstract", "introduction", "relatedwork", "background", "preliminaries", "method",
    "methods", "methodology", "approach", "experiment", "experiments", "experimentalsetup",
    "results", "discussion", "analysis", "evaluation", "ablation", "ablations",
    "ablationstudies", "conclusion", "conclusions", "limitations", "futurework",
    "broaderimpact", "broaderimpacts", "ethicsstatement", "acknowledgment",
    "acknowledgments", "acknowledgement", "acknowledgements",
}
# "3.1 Low-Rank Updates" and appendix-style "A Experimental Details"
NUMBERED_TITLE = re.compile(r"(\d{1,2}|[A-Z])(\.\d+)*\.?\s+[A-Z][^.!?]{2,60}")


def infer_headings(text: str) -> str:
    """pypdf gives plain text with no structure; turn likely section-title lines into
    Markdown headings so the heading-based chunker has real sections to split at.
    arXiv papers are very regular: 'Abstract', '1 Introduction', '3.1 Low-Rank Updates'.
    Letters-only matching survives small-caps artifacts ('1 I NTRODUCTION')."""
    # pypdf text has no real markdown - any line-leading '#' is a table artifact
    # ("# datasets: 7"); neutralize it so it can't masquerade as a heading
    lines = [re.sub(r"^[ \t]*#+[ \t]*", "", line) for line in text.split("\n")]
    stripped = [line.strip() for line in lines]

    def numbered_title(i: int) -> bool:
        s = stripped[i]
        return bool(3 <= len(s) <= 70 and len(s.split()) <= 7
                    and not s.endswith((".", ",", ":", ";", ")"))
                    and NUMBERED_TITLE.fullmatch(s))

    out = []
    for i, line in enumerate(lines):
        s = stripped[i]
        key = _letters(s)
        is_section_word = (3 <= len(s) <= 60
                           and not s.endswith((".", ",", ":", ";", ")"))
                           and (key in SECTION_KEYS or (key.startswith("appendix") and len(key) <= 12)))
        # consecutive numbered lines are a list (or prompt example), not section titles
        is_numbered = (numbered_title(i)
                       and not (i > 0 and numbered_title(i - 1))
                       and not (i + 1 < len(lines) and numbered_title(i + 1)))
        out.append(f"## {s}" if (is_section_word or is_numbered) else line)
    return "\n".join(out)


def convert_papers(papers: list[dict], force: bool = False) -> None:
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p in papers if force or not (MARKDOWN_DIR / f"{p['source_id']}.md").exists()]
    if not todo:
        return

    marker_converter = None
    if env_flag("USE_MARKER"):
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            marker_converter = PdfConverter(artifact_dict=create_model_dict())
        except ImportError:
            print("USE_MARKER=true but marker-pdf is not installed - falling back to pypdf")

    for paper in todo:
        pdf = PAPERS_DIR / f"{paper['source_id']}.pdf"
        target = MARKDOWN_DIR / f"{paper['source_id']}.md"
        if marker_converter is not None:
            text = clean_markdown(text_from_rendered(marker_converter(str(pdf)))[0])
        else:
            text = infer_headings(clean_markdown(pdf_to_text(pdf)))
        target.write_text(text, encoding="utf-8")
        print(f"converted {pdf.name} -> {target.name} ({target.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Stage 3: heading-based chunking. Split at markdown headings so each chunk is
# (part of) one real section of the paper; long sections are sub-split at
# paragraph boundaries with overlap. Deterministic and instant.
# ---------------------------------------------------------------------------

def chunk_markdown(text: str, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    max_chars = max_chars or int(env("CHUNK_SIZE", "1200"))
    overlap = overlap if overlap is not None else int(env("CHUNK_OVERLAP", "150"))

    pieces = HEADING_LINE.split(text)
    sections = [pieces[0]] if pieces[0].strip() else []
    for heading, body in zip(pieces[1::2], pieces[2::2]):
        sections.append(f"{heading}\n{body}")

    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        # sub-split the body only, then stamp the heading onto every sub-chunk
        # so each chunk still carries its section title
        first_line, _, body = section.partition("\n")
        heading = first_line if HEADING_LINE.match(first_line) else None
        if heading is None:
            body = section
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars - (len(heading) + 1 if heading else 0),
            chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "])
        for sub in splitter.split_text(body):
            chunks.append(f"{heading}\n{sub}" if heading else sub)
    return [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


def attach_metadata(paper: dict, chunk_texts: list[str]) -> list[dict]:
    """Turn raw chunk texts into records with the metadata the task requires."""
    records, section = [], "Front matter"
    for index, text in enumerate(chunk_texts):
        headings = HEADING.findall(text)
        if headings:
            section = headings[0]
        records.append({
            "chunk_id": f"{paper['source_id']}#{index:03d}",
            "source_id": paper["source_id"],
            "title": paper["title"],
            "section": section,
            "source_url": paper["source_url"],
            "topic": paper["topic"],
            "chunk_index": index,
            "text": text,
        })
        if headings:
            section = headings[-1]
    return records


def chunk_papers(papers: list[dict], force: bool = False, redo: set[str] | None = None) -> None:
    path = chunks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if path.exists() and not force:
        with open(path, encoding="utf-8") as fh:
            done = {json.loads(line)["source_id"] for line in fh if line.strip()}
    elif force:
        path.unlink(missing_ok=True)
    if redo and path.exists():
        # drop the redo papers' existing chunks so they get re-chunked fresh
        keep = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line)["source_id"] not in redo]
        path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        done -= redo

    with open(path, "a", encoding="utf-8") as out:
        for paper in papers:
            if paper["source_id"] in done:
                continue
            markdown = (MARKDOWN_DIR / f"{paper['source_id']}.md").read_text(encoding="utf-8")
            chunk_texts = chunk_markdown(markdown)
            for record in attach_metadata(paper, chunk_texts):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"chunked {paper['source_id']}: {len(chunk_texts)} chunks ({len(markdown) // 1024} KB source)")
    reset_chunk_cache()


# ---------------------------------------------------------------------------
# Stage 4: embed into Chroma
# ---------------------------------------------------------------------------

def embed_chunks(batch_size: int = 100, skip_if_current: bool = False) -> None:
    from app.models import load_chunks

    reset_chunk_cache()
    chunks, _ = load_chunks()
    if not chunks:
        sys.exit("no chunks to embed - run the chunking stage first")
    store = get_vector_store()
    if skip_if_current:
        existing = len(store.get(include=[])["ids"])
        if existing == len(chunks):
            print(f"vector index already current ({existing} chunks) - skipping embed")
            return
    store.reset_collection()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        store.add_texts(
            texts=[c["text"] for c in batch],
            ids=[c["chunk_id"] for c in batch],
            metadatas=[{k: c[k] for k in ("chunk_id", "source_id", "title", "section", "source_url", "topic", "chunk_index")} for c in batch],
        )
        print(f"embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper corpus end to end.")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N papers")
    parser.add_argument("--force", action="store_true", help="redo work even if outputs exist")
    parser.add_argument("--stage", choices=["download", "convert", "chunk", "embed"], default=None,
                        help="run a single stage instead of all four")
    parser.add_argument("--if-empty", action="store_true",
                        help="skip embedding when the vector index already matches chunks.jsonl (used at deploy boot)")
    parser.add_argument("--redo", default="",
                        help="comma-separated source_ids to re-chunk (e.g. --redo fid,qlora)")
    args = parser.parse_args()

    papers = load_manifest(args.limit)
    stages = [args.stage] if args.stage else ["download", "convert", "chunk", "embed"]
    if "download" in stages:
        download_papers(papers, args.force)
    if "convert" in stages:
        convert_papers(papers, args.force)
    if "chunk" in stages:
        chunk_papers(papers, args.force, {s.strip() for s in args.redo.split(",") if s.strip()})
    if "embed" in stages:
        embed_chunks(skip_if_current=args.if_empty)
    print("done")


if __name__ == "__main__":
    main()
