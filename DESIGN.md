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
2. **Convert** — pypdf extracts each PDF to text, then a heading-inference pass turns
   likely section-title lines into Markdown headings, exploiting how regular arXiv papers
   are: bare keywords ("Abstract", "Conclusion" — matched on letters only, so pypdf's
   small-caps artifacts like "R EFERENCES" still match) and numbered titles ("3.1 Low-Rank
   Updates"), while rejecting consecutive numbered lines (those are lists or prompt
   examples, not sections). Tools I tried and rejected: Microsoft's MarkItDown glued words
   together ("Animportantparadigm...") on all 30 of these LaTeX-built PDFs; marker produces
   genuinely better Markdown via deep-learning layout analysis and is supported behind
   `USE_MARKER=true`, but it needs torch plus ~2 GB of models and is far slower on CPU, so
   the zero-dependency heuristic is the default. Python cleanup also removes control
   characters, replaces ligatures (ﬁ → fi), rejoins hyphen-broken words, normalizes line
   endings, collapses whitespace, and cuts everything after the final References heading.
3. **Chunk (heading-based)** — the Markdown is split at heading lines, so each chunk is
   (part of) one real section of the paper; sections longer than `CHUNK_SIZE` (1200 chars)
   are sub-split at paragraph/sentence boundaries with `CHUNK_OVERLAP` (150) using
   LangChain's `RecursiveCharacterTextSplitter`, and every sub-chunk gets its section
   heading stamped on top so it stays self-describing. Fragments under 120 chars are
   dropped. Honest history: my first implementation was LLM-based chunking — the model
   re-emitted each paper as self-contained passages. The chunk quality was good, but
   re-emitting a 2 MB corpus token-by-token took the better part of an hour per rebuild
   and the model occasionally broke the separator format, silently producing giant blobs.
   A deterministic structural splitter over marker's real headings is instant,
   reproducible, and gives the same section metadata for free, so I replaced it.
4. **Embed** — chunks go into a persistent Chroma collection with cosine similarity. This
   is now the only stage that calls an API.

Metadata is attached in **Python** (zero tokens, deterministic): `source_id` (manifest
slug), `title`, `section` (from the real Markdown headings), `source_url` (arXiv abstract
page), `topic`, and `chunk_index` — the index is what makes neighbor expansion a simple
slice at query time. Everything is written to `data/chunks.jsonl`, which doubles as the
keyword-search index.

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
  following with tool-calling and structured-output support. One model serves the agent,
  reranker, and web judge to keep the setup simple; swapping is a one-line env
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
Pydantic schemas (structured LLM output and API contracts).

## Trade-offs and assumptions

- **Heading inference is a heuristic** — it recovers the big regular sections
  (Introduction, Method, Experiments...) reliably on arXiv papers but can miss unusual
  titles; ~1% of chunks end up with the generic "Front matter" label. marker
  (`USE_MARKER=true`) removes that limitation at the cost of a torch install and ~2 GB of
  models; the deployed API never needs either.
- **Answer latency** is 3+ LLM calls for a corpus question (agent, rerank, answer). Fine for
  an assistant API; batch throughput was not a goal.
- **No conversation memory** — each request is independent. "Tell me more" therefore asks
  for clarification instead of silently retrieving something random, which I consider the
  correct behavior for a stateless API.
- **Keyword search is not BM25** — it's exact matching for rare technical terms, feeding the
  reranker. The reranker, not the raw score, makes the final call.
- **LLM rerank vs a dedicated reranker** — a cross-encoder reranker (e.g. Cohere Rerank 4
  Pro through OpenRouter's `/rerank` endpoint, or ZeroEntropy's zerank-2) would rank
  candidates more accurately, but it only outputs scores: I would still need a separate
  sufficiency judgment (a tuned score threshold, or another LLM call) to drive the
  insufficient-information path. With ~15 candidates per question, one structured LLM call
  does both jobs at once, so I kept it. Swapping a dedicated reranker in behind the same
  interface — scores for ordering, a cheap LLM check for sufficiency — is the first
  improvement I would make with more time.
- The four robustness behaviors rely on the instruction-following of a strong model rather
  than hand-written detectors; with a much weaker model the tag protocol would need a
  fallback classifier.

## AI tool disclosure

I used AI coding assistants (OpenAI Codex and Claude Code) to help write and refactor the
implementation and documentation. The architecture and its decisions — RAG-as-a-tool
routing, marker conversion with heading-based chunking and Python-side metadata, hybrid
vector+keyword retrieval with neighbor expansion, LLM reranking as the confidence gate, and
the bounded web-search fallback — are my own design, and I reviewed and tested the code.
