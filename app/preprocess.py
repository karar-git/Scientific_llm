"""Corpus pipeline: download arXiv PDFs -> Markdown -> LLM chunking -> metadata -> vector store.

Run everything:      python -m app.preprocess
Smoke run:           python -m app.preprocess --limit 2
Redo from scratch:   python -m app.preprocess --force

Each stage is resumable: papers already downloaded, converted, or chunked are
skipped unless --force is given. Only the chunking and embedding stages need
an OpenRouter API key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import httpx

from app.models import PROJECT_ROOT, get_chat_model, get_vector_store, manifest_path, chunks_path, reset_chunk_cache

PAPERS_DIR = PROJECT_ROOT / "papers"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"

# The LLM does the chunking: it re-emits the paper as retrieval-sized passages
# separated by blank lines, so splitting them back apart is trivial and costs
# no output-format tokens (unlike JSON).
CHUNKER_SYSTEM_PROMPT = """You are a document chunker for a RAG system. You receive part of a research paper that was extracted from a PDF to plain text.

Re-emit the text as retrieval chunks, following these rules exactly:
1. A chunk is one self-contained passage about a single idea — usually one paragraph (roughly 60-250 words). Never cut a sentence in half.
2. Copy the original wording verbatim. Do not summarize, rephrase, or add any commentary of your own.
3. Separate consecutive chunks with exactly TWO blank lines (press enter three times). Do not use any other separator and do not number the chunks.
4. When a chunk starts a new section of the paper (Abstract, Introduction, Method, Experiments, ...), make the section title the first line of that chunk, written as a Markdown heading: ## Section Name
5. Skip content that is useless for retrieval: page headers/footers, author lists and affiliations, reference/bibliography entries, acknowledgements, and garbled PDF artifacts (broken math, orphaned figure/table fragments).
6. Repair obvious PDF-extraction damage only: rejoin words hyphen-broken across lines and sentences split mid-line.

Output the chunks and nothing else."""

CHUNK_SEPARATOR = re.compile(r"\n[ \t]*\n[ \t]*\n+")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
REFERENCES = re.compile(r"^#{0,6}\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
MIN_CHUNK_CHARS = 120
SEGMENT_CHARS = 9000  # how much markdown we hand the chunker LLM per call


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
# Stage 2: PDF -> text (pypdf) + light Python cleanup
# (MarkItDown was tried first, but its pdfminer backend glued words together
# on these LaTeX-built PDFs; pypdf extracts them cleanly.)
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
    match = None
    for match in REFERENCES.finditer(text):  # keep the LAST references heading
        pass
    if match and match.start() > len(text) * 0.4:
        text = text[: match.start()]
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


def convert_papers(papers: list[dict], force: bool = False) -> None:
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        target = MARKDOWN_DIR / f"{paper['source_id']}.md"
        if target.exists() and not force:
            continue
        pdf = PAPERS_DIR / f"{paper['source_id']}.pdf"
        target.write_text(clean_markdown(pdf_to_text(pdf)), encoding="utf-8")
        print(f"converted {pdf.name} -> {target.name} ({target.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Stage 3: LLM chunking + metadata
# ---------------------------------------------------------------------------

def segment_markdown(text: str, max_chars: int = SEGMENT_CHARS) -> list[str]:
    """Split a long document at paragraph boundaries so each LLM call stays small."""
    segments, current, size = [], [], 0
    for paragraph in text.split("\n\n"):
        if size + len(paragraph) > max_chars and current:
            segments.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        segments.append("\n\n".join(current))
    return segments


def split_llm_output(text: str) -> list[str]:
    """Split the chunker LLM's output back into individual chunks."""
    parts = [part.strip() for part in CHUNK_SEPARATOR.split(text) if part.strip()]
    if len(parts) <= 1 and len(text) > 1500:
        # the model separated chunks with a single blank line instead of two:
        # a giant one-blob result is useless for retrieval, so fall back
        parts = [part.strip() for part in re.split(r"\n[ \t]*\n+", text) if part.strip()]
    return [part for part in parts if len(part) >= MIN_CHUNK_CHARS]


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

    llm = get_chat_model()
    with open(path, "a", encoding="utf-8") as out:
        for paper in papers:
            if paper["source_id"] in done:
                continue
            markdown = (MARKDOWN_DIR / f"{paper['source_id']}.md").read_text(encoding="utf-8")
            chunk_texts: list[str] = []
            for segment in segment_markdown(markdown):
                reply = llm.invoke([("system", CHUNKER_SYSTEM_PROMPT), ("human", segment)])
                chunk_texts.extend(split_llm_output(str(reply.content)))
            for record in attach_metadata(paper, chunk_texts):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"chunked {paper['source_id']}: {len(chunk_texts)} chunks ({len(markdown) // 1024} KB source)")
            if len(chunk_texts) < max(3, len(markdown) // 12000):
                print(f"  WARNING: suspiciously few chunks - consider: python -m app.preprocess --stage chunk --redo {paper['source_id']}")
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
