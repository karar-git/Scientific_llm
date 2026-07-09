# Testing

## Automated tests

```bash
python -m pytest
```

Two layers. `test_preprocess.py` and `test_rag.py` are pure unit tests of the pipeline
logic and always run; `test_api.py` sends real questions through the live pipeline
(indexed corpus + real LLM) and skips with a clear message until the corpus is built and
an API key is configured.

| Test | What it proves |
|---|---|
| `test_chunk_markdown_...` (x3) | chunks split at section headings; long sections sub-split under the size cap with the heading stamped on; fragments are dropped |
| `test_attach_metadata_...` | every chunk gets `source_id`, `title`, `section`, `source_url`; sections carry across heading-less chunks |
| `test_clean_markdown_...` | references section and control characters are removed |
| `test_pdf_extracts_to_usable_text` | a real downloaded paper extracts to clean, correctly-spaced text (pypdf fallback path) |
| `test_keyword_search_...` / `test_expand_neighbors_...` | exact-match search ranks correctly; neighbor expansion returns prev/next chunks and handles paper boundaries |
| `test_empty_question_...` (rag) | empty input short-circuits to `needs_clarification` without any model call |
| `test_health_...` (api, live) | the index is loaded and reported |
| `test_in_scope_question_...` (api, live) | a corpus question runs the full graph and returns `answered` + citations with all metadata fields |
| `test_small_talk_...`, `test_out_of_scope_...`, `test_empty_question_...`, `test_oversized_question_...` (api, live) | the documented statuses and API validation |

## Manual test cases

Run after building the real corpus (`python -m app.preprocess`) and starting the API.
Each case below lists the input and the observed output.

> All commands: `curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "..."}'`

### 1. Normal in-scope question

Input: `How does LoRA reduce the number of trainable parameters?`
Expected: `status=answered`, citations pointing at the LoRA paper.

Observed output:
```json
<PASTE REAL OUTPUT HERE>
```

### 2. Non-English input (Arabic — bonus)

Input: `ما هو الفرق بين LoRA و QLoRA؟`
Expected: `status=answered`, answer written in Arabic, citations from the LoRA/QLoRA papers
(the agent rewrites the question into an English search query internally).

Observed output:
```json
<PASTE REAL OUTPUT HERE>
```

### 3. Out-of-scope question

Input: `Who won the 2022 World Cup?`
Expected: `status=out_of_scope`, polite refusal naming what the corpus covers, no citations.

Observed output:
```json
<PASTE REAL OUTPUT HERE>
```

### 4. Vague / malformed input

Input A: `` (empty string) — Expected: `status=needs_clarification` (short-circuited in Python).
Input B: `tell me more` — Expected: `status=needs_clarification`, the assistant asks which topic.
Input C: `asdkjh qwe zzz` — Expected: `status=needs_clarification`.

Observed outputs:
```json
<PASTE REAL OUTPUT HERE>
```

### 5. Low retrieval confidence

Input: `What exact GPU cluster hardware did the QLoRA authors use for the Guanaco 13B run?`
(a detail unlikely to be in the retained chunks)
Expected: the reranker reports insufficient context. With `WEB_SEARCH_ENABLED=false`:
`status=insufficient_information` and no invented facts. With the fallback on: either an
answer that explicitly says the corpus was not enough and cites web URLs in `web_sources`,
or the same clean refusal if the web search also fails.

Observed output:
```json
<PASTE REAL OUTPUT HERE>
```
