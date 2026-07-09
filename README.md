# Swibit Knowledge Assistant

A RAG API over ~30 arXiv research papers about large language models (retrieval-augmented
generation, fine-tuning, tool calling, reasoning, and self-correction). You send a
natural-language question, the service retrieves relevant passages from the indexed papers,
and answers **only** from those passages, with citations back to the exact chunks it used.

Built with LangGraph (agent orchestration) + LangChain (models, vector store), FastAPI,
Chroma, and models served through OpenRouter.

## How it works (short version)

The main LLM sees your question first. Small talk is answered directly, off-topic questions
are declined, and real questions trigger a `search_papers` tool that runs the RAG pipeline:
vector similarity + exact keyword search, neighbor-chunk expansion, and an LLM reranker that
picks the best passages and judges whether they are enough. If they are not, an optional web
search fallback (DuckDuckGo, up to 3 tries) fills the gap — web material is reported
separately and never mixed into corpus citations. Full details and diagram in [DESIGN.md](DESIGN.md).

## Setup

Requires Python 3.12 and an [OpenRouter](https://openrouter.ai/keys) API key — that is the
only key needed; both the chat models and the embeddings go through OpenRouter.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # then put your real OPENROUTER_API_KEY in .env
```

## Build the corpus (one-time indexing)

```bash
python -m app.preprocess
```

By default PDFs are converted with pypdf plus a heading-inference pass tuned to arXiv's
regular section structure — fast and dependency-free. Optionally, `pip install marker-pdf`
and set `USE_MARKER=true` to convert with marker instead (a deep-learning layout model:
best-quality Markdown, but torch + ~2 GB of models and much slower on CPU). marker is
deliberately not in `requirements.txt` so the deployed API image stays small.

This runs four resumable stages (already-finished work is skipped on re-run):

1. **download** — fetches the 30 paper PDFs listed in `data/papers_manifest.json` into `papers/`
2. **convert** — extracts each PDF to text (pypdf) and infers Markdown section headings
   from arXiv's regular title patterns, into `data/markdown/`; set `USE_MARKER=true` to
   use marker instead (see above).
3. **chunk** — heading-based structural chunking: split at section headings, sub-split long
   sections with overlap; metadata is attached in Python and everything is written to
   `data/chunks.jsonl`. Instant, no API calls.
4. **embed** — chunks are embedded and stored in a persistent Chroma collection (`.chroma/`)

Useful flags: `--limit 2` (only the first 2 papers, good for a smoke run), `--stage chunk`
(one stage only), `--force` (redo everything), `--redo id1,id2` (re-chunk specific papers).
Only the embed stage calls OpenRouter — the full 30-paper indexing costs about a cent.

## Run the API

```bash
uvicorn app.api:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs

### `GET /health`

```json
{"status": "ok", "chunks_indexed": 812}
```

### `POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d "{\"question\": \"How does LoRA reduce the number of trainable parameters?\"}"
```

```json
{
  "answer": "LoRA freezes the pretrained weights and injects trainable low-rank matrices ... [lora#004]",
  "status": "answered",
  "citations": [
    {
      "chunk_id": "lora#004",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "Method",
      "source_url": "https://arxiv.org/abs/2106.09685"
    }
  ],
  "web_sources": []
}
```

`status` is always one of:

| status | meaning |
|---|---|
| `answered` | grounded answer with citations (if `web_sources` is non-empty, part of the answer came from the web fallback and says so) |
| `small_talk` | greeting handled directly, no retrieval |
| `out_of_scope` | valid question, but unrelated to the indexed papers |
| `needs_clarification` | empty, gibberish, or too vague to search (e.g. "tell me more") |
| `insufficient_information` | retrieval found nothing good enough; the system refuses instead of guessing |

Questions in other languages (e.g. Arabic) are answered in that language — see **Bonus** below.

## Tests

```bash
python -m pytest
```

`tests/test_preprocess.py` and `tests/test_rag.py` are pure unit tests of the pipeline
logic (chunk splitting, metadata, keyword search, neighbor expansion) and need no API key.
`tests/test_api.py` is an integration suite that sends real questions through the live
pipeline — it requires the corpus to be indexed and an API key in `.env`, and skips with a
clear message until both exist. See [TESTING.md](TESTING.md) for manual cases.

## Configuration

All settings live in `.env` (see `.env.example`): model names, retrieval `RETRIEVAL_K` /
`RERANK_K`, Chroma location, and `WEB_SEARCH_ENABLED` to turn the web fallback off entirely.

**Model choice:** `OPENROUTER_MODEL` (default `deepseek/deepseek-v4-flash`) drives the
agent, reranker, and web judge; any OpenRouter model that supports tool calling works.
`OPENROUTER_EMBEDDING_MODEL` (default `openai/text-embedding-3-small`) is used for
indexing and query-time search — if you change it, re-run the embed stage, since vectors
from different models are not comparable.

## Deployment (Railway)

The repo is Railway-ready (`railway.json`, `Procfile`, `runtime.txt`):

1. Run indexing locally once (`python -m app.preprocess`) and commit `data/chunks.jsonl`.
2. Create a Railway service from the GitHub repo and set the environment variables from
   `.env.example` (at minimum `OPENROUTER_API_KEY`).
3. Done. Railway's filesystem is ephemeral, so the start command runs
   `python -m app.preprocess --stage embed --if-empty` before uvicorn: when a fresh
   container has no vector index, it rebuilds one from the committed `data/chunks.jsonl`
   (one embedding pass, roughly a cent) and skips instantly when the index is already
   current. Attach a Railway volume at `.chroma/` if you want the index to survive
   restarts without even that.

## Bonus: Arabic query handling

Arabic questions are accepted as-is: the agent rewrites them into a precise English search
query for retrieval over the English corpus, then writes the final answer in Arabic. Example
in [TESTING.md](TESTING.md).

## Known limitations

- No multi-turn chat (yet): each `/ask` is independent, so follow-ups like "tell me more"
  without a topic get `needs_clarification`. Multi-turn conversation was planned — the
  LangGraph state already accumulates messages, so it mainly needs a checkpointer plus a
  `session_id` on the request — but the task's time window (built alongside a full-time
  job) didn't allow implementing and testing it properly, so the API ships stateless.
  The exact upgrade path is in DESIGN.md.
- Chunking is structural (headings + size), not semantic. An earlier version chunked with
  an LLM — nice passages, but far too slow to rebuild (~an hour per full run), so it was
  replaced; the story is in DESIGN.md.
- Keyword search is exact word matching, not BM25 — good for technical terms like "LoRA",
  not a general lexical ranker.
- The web fallback depends on DuckDuckGo availability; when it fails the system degrades to
  a clean `insufficient_information` refusal.
