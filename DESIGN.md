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

I used **Claude (Claude Code)** as my AI-assisted development tool, as permitted by the
task. How the work was actually split:

1. An earlier AI-generated draft of this task existed and I scrapped it entirely — too many
   classes, too much indirection, unclear code. I then dictated the design brief below
   (before any of this code existed) and had Claude rebuild the project to it.
2. **The architecture and its decisions are mine**: RAG-as-a-tool routing, the arXiv paper
   corpus, LLM-side chunking (later replaced — see the honest history in the ingestion
   section), hybrid vector + keyword retrieval, neighbor expansion, the reranker doubling as
   the sufficiency gate, the bounded 3-try web-search fallback, the flat few-files layout,
   and FastAPI delivery. Claude wrote and refactored the code and documentation to that
   brief, under my direction.
3. I set up the environment and API keys, ran the indexing, and tested the system myself.
4. I inspected the code, edited parts of it by hand, and handed the rest of my notes and
   corrections back to Claude to apply.
5. I checked everything again end to end (the automated tests plus the manual cases in
   TESTING.md) and delivered once I was satisfied. I understand and can explain every part
   of the submitted implementation.

<details>
<summary><b>The original design brief I gave Claude</b> — a voice-note transcript, lightly cleaned (profanity removed, obvious speech-to-text errors fixed), included as evidence of which decisions were mine before any code was written</summary>

> Yes, so just delete everything — it's too complex and messy. It just created classes as
> much as it could, wasted a lot of stuff, and unclear code. So I will tell you the strategy
> I'm going to use if I build this. You will not build it the old way, because you will use
> LangChain, like the graph, instead of that stuff.
>
> I'm going to use the OpenRouter API and use GPT, Gemini, or whatever model I choose — it
> doesn't matter for that task. I will get the articles or whatever I want to do RAG about.
> I didn't like the previous topic; pick something related to me and more interesting. Let's
> choose around 30 to 40 articles or whatever, and let's start.
>
> I will use a method of: I will write a system prompt like "you are a RAG chunker" — you
> will get a whole document and you will turn it into chunks. It will return it as a JSON
> array of strings or whatever. So I will give it the whole document and it will divide it
> into appropriate chunks. That's the first thing.
>
> After the chunking, we will embed it. We will embed it with a good model here — it doesn't
> need to be the same model as the one we use to respond or to chunk. Just a good embedder;
> it's so cheap.
>
> After the embedding, the reranker. My reranker will be: get the best 10 or whatever, give
> them to the model, and it will rerank them and give me the most appropriate four for that
> question, or two, or whatever, and then feed that to the main LLM that will answer the
> question for me. And in the reranker we will not just take the best ones alone — we will
> take the best four and return every chunk of them with the previous and the next one. Like
> if I get chunk three, I will return chunks two, three, four. Then it will decide whether
> the information from the four it selected is enough to answer the question. If the
> information is not enough, it will call a tool called search, where it will search the web
> for the information, and loop on that: if it gets enough information it answers; if not,
> it has only three tries.
>
> In addition: the RAG system will be just a tool inside the main LLM. The prompt goes first
> to the main LLM, which doesn't have RAG yet but has a tool. So if the question is just
> "hi", we don't need the RAG at all — it answers immediately. If the question is related to
> the documents we have, or the topic the bot specializes in, it calls the RAG tool, which
> returns after the reranker and those steps to the LLM, and then it decides: if it is
> enough, it answers; if not, it does the search-tool loop.
>
> It will be delivered with FastAPI. I love simple stuff. Let's have all the files in one
> folder, all of them as PDFs — let's do it about research papers actually, around 30
> research papers on LLM fine-tuning and that stuff. We will use a tool to convert those
> PDFs to Markdown — I think there was a tool or library from Microsoft to convert PDFs to
> Markdown, as it is the best. After that we give it to the LLM to do the chunking. That
> needs a test, actually — a test in the test file for the PDF conversion.
>
> I love all related stuff being in one file. The preprocessing should be in one file — the
> preprocessing of all the PDFs and the stuff I need. All the model creation in one file,
> the main logic in one file, and maybe the API in one file. No need to create "oh, that's a
> class with one variable" — I don't like that approach. Do it clear like this, but use
> LangChain, like a graph, and structured output. In the old version, validation used some
> strange method — to detect if the input is Arabic or English it used some strange
> hand-written approach. The LLM can handle this; we will use smart models — they can return
> JSON and such. Did they even ask for Arabic validation? It's not necessary; let the model
> just answer directly. It depends: we will use structured output for the chunking so it
> returns the chunks we want after the Markdown step. And surely we will save the index of
> each chunk in the document so we can just slice it.
>
> What do they mean by semantic search? Look — we will do the RAG stuff, which is
> calculating similarity. In addition, the tool will use keyword search as well. Let's say
> there's a topic like "error recovery from tools" or those famous terms — like ReAct, which
> I think was a framework — we will search for those words together across all files, return
> the chunk that has them, plus the one before and the one after, surely. Then all of those
> go to the reranker, it reranks them and gives the best chunks to our LLM.
>
> Okay, and let's do what they ask: they say each stored unit should keep enough metadata —
> document ID, title, section, and a reference to where it came from. Just do it, and do the
> stuff they ask for: the DESIGN.md, the tests, the README, whatever they are asking for.
> The answers must be restricted to the information that comes from the RAG, because it is
> science-specific information and the AI will hallucinate a lot about it otherwise, and
> also when there is no information from the search.
>
> In addition to the English research papers you can use some Arabic research papers if
> there are any — but that is for the bonus, I think. Let the articles be about the hot
> topics right now: some research papers about RAG, some about fine-tuning, some about tool
> calling, about the thought process in LLMs, about error recovery, those kinds of things.
>
> About the chunking: when you get the paper, convert it to Markdown and give it to the LLM
> to chunk. Actually, no need for it to return JSON or be restricted by JSON — that would
> burn a lot of tokens. Let it work like this: we give it the Markdown, and for every chunk
> it inserts a double or triple empty line, and we split it with Python code. Then in the
> preprocessing we attach the metadata with Python code, because that takes no tokens:
> source_id, title, section, source_url — the stuff they ask for in data-processing section
> 3.3.
>
> For the retrieval, let it be a tool: we send the prompt to the LLM, it decides whether the
> prompt is just "hi" or whatever and responds, or whether it is related and it should call
> the RAG function. The RAG function does the similarity, the keyword search, gets the
> chunks with the one before and the one after, and puts them into the rerank. The rerank
> gives us the best chunks and feeds them back to the LLM — easy as that. The rerank decides
> if that information is enough or not. If it is enough, it returns it. If not, it does the
> web search — three times. If there is still nothing, it returns to the LLM that there is
> not enough information, and the LLM responds to the user that there is insufficient
> information.
>
> Decide the overlap for me. The chunk size doesn't matter much for us, as the LLM will
> chunk it — chunks should be around one comprehensive paragraph, like two, three, four
> lines, whatever makes sense.
>
> For input robustness (3.5): our LLM is smart, so it will handle non-English input; we
> don't need special handling. We should tell it in the system prompt that it is specialized
> in this corpus and shouldn't answer anything else. For vague or malformed inputs —
> gibberish, or "tell me more": we can't just embed "tell me more" and search for
> similarity, because that's useless and we will get bad results. So if that happens, let
> the LLM write a proper search query itself, embed that, and use it for the keyword search
> as well.
>
> And do the testing — you must do it. Put it in a separate folder, and our implementation
> in around four or five files: one for the preprocessing, one for the main logic, and so on
> — in an understandable way, not the a-lot-of-classes way. In the bonus we will have the
> Arabic pipeline working as it is, and we have the reranker as well.
>
> Just simple architecture files, and no file-for-each-class-and-variable — I don't like
> that approach.

</details>
