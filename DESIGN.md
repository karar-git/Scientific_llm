# Design

## Architecture

```
                          POST /ask
                              |
                              v
                    +------------------+
                    |   agent (LLM)    |  routes the question
                    +------------------+
                     |        |       \
        small talk / |        | search_papers(query, keywords)
        out-of-scope |        v
        / clarify    |  +-----------------------------------------+
                     |  | retrieve node                           |
                     |  |  1. vector top-k (Chroma, cosine)       |
                     |  |  2. keyword exact-match search          |
                     |  |  3. LLM rerank -> best 4 + sufficient?  |
                     |  |  4. if insufficient: web search x3      |
                     |  |  5. expand selected chunks with their   |
                     |  |     previous / next neighbors           |
                     |  +-----------------------------------------+
                     |        |
                     |        v
                     |  agent again -> grounded answer + SOURCES line
                     v        |
                    +------------------+
                    | finalize (Python)|  parse tags + SOURCES -> status, citations
                    +------------------+
```

Orchestration is a LangGraph `StateGraph` with two nodes (`agent`, `retrieve`) and a
conditional edge: if the agent's reply contains a tool call we run retrieval and loop back,
otherwise we finish. The RAG pipeline is deliberately a **tool** of the main LLM rather than
a fixed pipeline: that way "hi" never touches the vector store, and the agent — not a pile
of regex heuristics — decides whether a question is small talk, out of scope, too vague, or
a real corpus question. It also rewrites the user's wording into a clean English search
query, which is what makes vague and non-English inputs work.

## Corpus

30 arXiv papers (`data/papers_manifest.json`), all PDFs, on five topics I actually work
with: RAG (8), fine-tuning (6), tool calling (6), reasoning (6), and error recovery /
self-correction (4). They are public, stable (arXiv IDs don't move), and technical enough
that grounding really matters — the model must quote the paper, not its own memory of it.

## Ingestion pipeline (`app/preprocess.py`)

Four resumable stages; each skips work whose output already exists.

1. **Download** — PDFs from `arxiv.org/pdf/<id>`, validated by the `%PDF` magic bytes.
2. **Convert** — pypdf extracts each PDF to text. I tried Microsoft's MarkItDown first, but
   its pdfminer backend glued words together ("Animportantparadigm...") on all 30 of these
   LaTeX-built PDFs, which would have poisoned both embeddings and keyword search; pypdf
   extracts them with correct spacing. Python cleanup then removes control characters,
   replaces ligatures (ﬁ → fi), rejoins hyphen-broken words, normalizes line endings,
   collapses whitespace runs, and cuts everything after the final References heading.
3. **Chunk (LLM-based, the main design decision)** — a chunker prompt gives the whole
   document (in ~9k-char segments, split at paragraph boundaries) to the LLM, which re-emits
   it verbatim as self-contained passages of roughly 60–250 words, separated by blank lines.
   Why an LLM instead of a fixed-size splitter: PDF extraction produces messy layout, and a
   character-count splitter happily cuts sentences, tables, and ideas in half; a semantic
   chunker returns passages that each carry one complete idea, which is exactly what you
   want a retrieval hit to be. Two cost controls: the output format is plain text with
   blank-line separators (not JSON — no schema tokens wasted, trivially split in Python),
   and the LLM also drops boilerplate (headers/footers, author lists, reference entries) and
   repairs hyphen-broken words as it goes. It additionally normalizes section titles to
   Markdown headings, which is what makes the `section` metadata reliable.
4. **Embed** — chunks go into a persistent Chroma collection with cosine similarity.

Metadata is attached in **Python**, not by the LLM (zero tokens, deterministic): `source_id`
(manifest slug), `title`, `section` (tracked from the headings the chunker emits),
`source_url` (arXiv abstract page), `topic`, and `chunk_index` — the index is what makes
neighbor expansion a simple slice at query time. Everything is written to
`data/chunks.jsonl`, which doubles as the keyword-search index.

## Retrieval and generation (`app/rag.py`)

- **Embedding model**: `openai/text-embedding-3-small` served through OpenRouter
  ($0.02/M tokens — the whole corpus embeds for about a cent). Cheap, strong on English
  technical text; the corpus is small enough that a bigger embedder buys nothing
  measurable.
- **Vector search**: Chroma, cosine, `RETRIEVAL_K=10` candidates.
- **Hybrid keyword search**: exact word-boundary matching for the technical terms the agent
  extracts from the question ("LoRA", "ReAct", "tool error recovery"). Embeddings are weak
  on rare exact tokens — a hit for the literal string is a signal worth having.
- **Neighbor expansion**: every selected chunk is returned together with the previous and
  next chunk of the same paper (`chunk_index` ± 1). A single paragraph often states a claim
  whose setup or numbers live in the paragraph next door.
- **Reranking**: an LLM sees all candidates and returns structured output — the best
  `RERANK_K=4` chunk ids plus a boolean `sufficient`. This is both the reranker and the
  low-confidence detector: no similarity-threshold tuning, the model judges relevance
  directly.
- **Web fallback (optional, `WEB_SEARCH_ENABLED`)**: when the corpus is insufficient, up to
  3 DuckDuckGo searches run; after each, an LLM judge decides "enough?" and proposes a
  better query if not. Web results are passed to the agent clearly labeled as NOT from the
  corpus, are cited by URL in the answer text, and are returned in a separate `web_sources`
  field — they never appear as corpus citations.
- **LLM**: everything goes through OpenRouter's OpenAI-compatible API with a single key,
  which lets me pick models purely on price/quality. Default is
  `deepseek/deepseek-v4-flash` ($0.09 in / $0.18 out per M tokens): strong instruction
  following with tool-calling and structured-output support, and cheap enough that
  LLM-chunking the whole corpus costs well under a dollar. One model serves the agent,
  chunker, reranker, and web judge to keep the setup simple; swapping is a one-line env
  change (`OPENROUTER_MODEL`) — e.g. `anthropic/claude-haiku-4.5` if stricter format
  adherence is ever needed, or `google/gemini-3.1-flash-lite` as a middle ground.

### Hallucination control

Layered: (1) the agent is instructed to answer only from retrieved passages and to cite
chunk ids inline; (2) the reranker's `sufficient=false` forces an explicit
`insufficient_information` refusal path — the "answer" prompt never even runs on weak
context alone; (3) the final `SOURCES:` line is parsed in Python and mapped to real chunk
metadata, so citations can only point at chunks that actually exist; (4) web content is
segregated as described above.

## Input robustness (task 3.5)

| Case | Handling |
|---|---|
| Non-English input | The agent rewrites the question into an English search query, retrieves over the English corpus, and answers in the user's language. |
| Out-of-scope question | The agent declines without retrieval and tags the reply `[OUT_OF_SCOPE]`, which the API maps to `status=out_of_scope`. |
| Vague / malformed input | Empty strings short-circuit in Python to `needs_clarification`; gibberish or "tell me more" gets a `[CLARIFY]` reply asking what to look up. |
| Low retrieval confidence | The reranker returns `sufficient=false`; without a successful web fallback the API returns `insufficient_information` and no invented facts. |

The routing tags (`[SMALL_TALK]`, `[OUT_OF_SCOPE]`, `[CLARIFY]`) are a deliberate trick:
they let one LLM call do classification *and* produce the user-facing reply, and Python
strips the tag and derives the machine-readable status. No separate classifier model, no
language-detection library.

## Vector DB choice

Chroma, embedded and persisted to `.chroma/`. For 30 documents (~1k chunks) a client-server
vector DB is pure overhead; Chroma needs no extra process, persists to disk, supports cosine
+ metadata storage, and has a first-class LangChain integration.

## Project structure

```
app/preprocess.py   everything data: download -> markdown -> chunks -> embeddings
app/models.py       config + factories (LLM, embeddings, vector store, chunk store)
app/rag.py          the LangGraph agent, retrieval pipeline, answer/citation assembly
app/api.py          FastAPI schemas and endpoints
tests/              unit tests (pure logic) + live integration tests (real corpus, real LLM)
data/               manifest + generated chunks.jsonl (+ markdown cache, gitignored)
```

Four flat modules, on purpose. Each file is one complete concern you can read top to
bottom; there are no single-method classes or indirection layers. The only classes are
Pydantic schemas (structured LLM output and API contracts) and the two fakes, which must
subclass LangChain interfaces.

## Trade-offs and assumptions

- **LLM chunking** costs ~2× the corpus size in tokens and is not byte-deterministic between
  runs. Accepted: it's a one-time indexing cost on 30 papers, the stage is resumable per
  paper, and chunk quality is what the whole pipeline stands on.
- **Answer latency** is 3+ LLM calls for a corpus question (agent, rerank, answer). Fine for
  an assistant API; batch throughput was not a goal.
- **No conversation memory** — each request is independent. "Tell me more" therefore asks
  for clarification instead of silently retrieving something random, which I consider the
  correct behavior for a stateless API.
- **Keyword search is not BM25** — it's exact matching for rare technical terms, feeding the
  reranker. The reranker, not the raw score, makes the final call.
- The four robustness behaviors rely on the instruction-following of a strong model rather
  than hand-written detectors; with a much weaker model the tag protocol would need a
  fallback classifier.

## AI tool disclosure

I used AI coding assistants (OpenAI Codex and Claude Code) to help write and refactor the
implementation and documentation. The architecture and its decisions — RAG-as-a-tool
routing, LLM chunking with blank-line separators and Python-side metadata, hybrid
vector+keyword retrieval with neighbor expansion, LLM reranking as the confidence gate, and
the bounded web-search fallback — are my own design, and I reviewed and tested the code.
