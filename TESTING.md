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

Observed output (≈27s):
```json
{
  "answer": "LoRA (Low-Rank Adaptation) reduces the number of trainable parameters through a clever re-parametrization approach:\n\n1. **Frozen pre-trained weights**: Instead of updating the full pre-trained weight matrix \\(W_0 \\in \\mathbb{R}^{d \\times k}\\) during fine-tuning, LoRA **freezes** \\(W_0\\) so it receives no gradient updates [lora#014].\n\n2. **Low-rank decomposition matrices**: LoRA constrains the weight update \\(\\Delta W\\) by representing it as a product of two much smaller matrices: \\(\\Delta W = BA\\), where \\(B \\in \\mathbb{R}^{d \\times r}\\) and \\(A \\in \\mathbb{R}^{r \\times k}\\), and the rank \\(r \\ll \\min(d, k)\\) [lora#014]. Only these small matrices \\(A\\) and \\(B\\) contain trainable parameters.\n\n3. **Dramatic parameter reduction**: Because \\(r\\) is chosen to be very small (e.g., \\(r = 1\\) or \\(2\\) even when the full dimension \\(d\\) is as high as 12,288), the number of trainable parameters is drastically reduced. For GPT-3 175B, LoRA reduces trainable parameters by **10,000 times** compared to full fine-tuning [lora#001].\n\n4. **Efficiency in training**: Since only the small low-rank matrices are optimized, there is no need to calculate gradients or maintain optimizer states for the vast majority of parameters. This lowers GPU memory requirements by up to **3 times** when using adaptive optimizers like Adam [lora#005].\n\nDuring deployment, the matrices can be merged: \\(W = W_0 + BA\\) is computed and stored explicitly, so **no inference latency** is introduced compared to a fully fine-tuned model [lora#016].",
  "status": "answered",
  "citations": [
    {
      "chunk_id": "lora#001",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "ABSTRACT",
      "source_url": "https://arxiv.org/abs/2106.09685"
    },
    {
      "chunk_id": "lora#005",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "1 I NTRODUCTION",
      "source_url": "https://arxiv.org/abs/2106.09685"
    },
    {
      "chunk_id": "lora#014",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "4.1 L OW-R ANK -PARAMETRIZED UPDATE MATRICES",
      "source_url": "https://arxiv.org/abs/2106.09685"
    },
    {
      "chunk_id": "lora#016",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "4.1 L OW-R ANK -PARAMETRIZED UPDATE MATRICES",
      "source_url": "https://arxiv.org/abs/2106.09685"
    }
  ],
  "web_sources": []
}
```

(Some `section` labels carry pypdf small-caps artifacts, e.g. `1 I NTRODUCTION` — a known
cosmetic limitation of the heuristic PDF extraction; see DESIGN.md trade-offs.)

### 2. Non-English input (Arabic — bonus)

Input: `ما هو الفرق بين LoRA و QLoRA؟`
Expected: `status=answered`, answer written in Arabic, citations from the LoRA/QLoRA papers
(the agent rewrites the question into an English search query internally).

Observed output (≈80s — the agent ran two search rounds to cover both papers):
```json
{
  "answer": "## الفرق بين LoRA و QLoRA\n\n**LoRA (Low-Rank Adaptation)** هو أسلوب ضبط دقيق (fine-tuning) فعّال يقوم بتجميد أوزان النموذج المُدرَّب مسبقًا (pre-trained weights) وإضافة مصفوفات قابلة للتدريب ذات تحلل منخفض الرتبة (low-rank decomposition matrices) في كل طبقة من طبقات Transformer [lora#001]. بدلاً من تحديث جميع معلمات النموذج الضخمة، يتم تعلم مصفوفتين صغيرتين `B` و `A` بحيث يكون تحديث الوزن `ΔW = BA`، حيث أن الرتبة `r` أصغر بكثير من أبعاد المصفوفة الأصلية [lora#014]. هذا يقلّل عدد المعلمات القابلة للتدريب بمقدار 10,000 مرّة ومتطلبات ذاكرة GPU بمقدار 3 مرات مقارنة بالضبط الكامل [lora#001].\n\n**QLoRA (Quantized Low-Rank Adaptation)** هو تطوّر على LoRA يقلّل استخدام الذاكرة بشكل أكبر بكثير، مما يتيح ضبط نموذج بحجم 65 مليار معلمة على GPU واحد بذاكرة 48GB فقط [qlora#001]. الفروقات الرئيسية هي:\n\n| الميزة | LoRA | QLoRA |\n|-------|------|-------|\n| **دقة الأوزان المُجمَّدة** | 16-bit (أو 32-bit) | 4-bit (مكمّمة/quantized) |\n| **نوع البيانات** | Float-point عادي | NormalFloat (NF4) جديد [qlora#001] |\n| **تقنيات إضافية** | لا يوجد | **التكميم المزدوج (Double Quantization)** لتقليل حجم ثوابت التكميم، و **المُحسّنات المُقسّمة (Paged Optimizers)** لإدارة ارتفاعات الذاكرة [qlora#005] |\n| **متطلبات الذاكرة لنموذج 65B** | >780 GB (ضبط كامل) | <48 GB [qlora#003] |\n| **الأداء** | يضاهي الضبط الكامل | يحافظ على أداء 16-bit بدون تدهور [qlora#003] |\n\nباختصار، **QLoRA يبني على LoRA** بنفس فكرة المحوّلات منخفضة الرتبة، لكنّه يُضيف تكميم النموذج الأساسي إلى 4-bit باستخدام نوع بيانات NormalFloat الأمثل معلوماتيًا، بالإضافة إلى تقنيتي التكميم المزدوج والمُحسّنات المُقسّمة، مما يقلّل استهلاك الذاكرة بشكل جذري مع الحفاظ على الأداء [qlora#001][qlora#005].",
  "status": "answered",
  "citations": [
    {
      "chunk_id": "lora#001",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "ABSTRACT",
      "source_url": "https://arxiv.org/abs/2106.09685"
    },
    {
      "chunk_id": "lora#005",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "1 I NTRODUCTION",
      "source_url": "https://arxiv.org/abs/2106.09685"
    },
    {
      "chunk_id": "lora#014",
      "source_id": "lora",
      "title": "LoRA: Low-Rank Adaptation of Large Language Models",
      "section": "4.1 L OW-R ANK -PARAMETRIZED UPDATE MATRICES",
      "source_url": "https://arxiv.org/abs/2106.09685"
    },
    {
      "chunk_id": "qlora#001",
      "source_id": "qlora",
      "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
      "section": "Abstract",
      "source_url": "https://arxiv.org/abs/2305.14314"
    },
    {
      "chunk_id": "qlora#003",
      "source_id": "qlora",
      "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
      "section": "1 Introduction",
      "source_url": "https://arxiv.org/abs/2305.14314"
    },
    {
      "chunk_id": "qlora#005",
      "source_id": "qlora",
      "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
      "section": "1 Introduction",
      "source_url": "https://arxiv.org/abs/2305.14314"
    }
  ],
  "web_sources": []
}
```

### 3. Out-of-scope question

Input: `Who won the 2022 World Cup?`
Expected: `status=out_of_scope`, polite refusal naming what the corpus covers, no citations.

Observed output (≈4s — no retrieval is run):
```json
{
  "answer": "That question is about sports (football/soccer), which falls outside the scope of my indexed corpus. I cover research papers on large language models, specifically topics such as retrieval-augmented generation, fine-tuning (LoRA, QLoRA, RLHF, DPO), tool calling and agents, reasoning and chain-of-thought, and self-correction / error recovery. I cannot answer questions about sports, cooking, politics, or other unrelated subjects. If you have a question about LLM research, I'd be happy to help!",
  "status": "out_of_scope",
  "citations": [],
  "web_sources": []
}
```

### 4. Vague / malformed input

Input A: `` (empty string) — Expected: `status=needs_clarification` (short-circuited in Python).
Input B: `tell me more` — Expected: `status=needs_clarification`, the assistant asks which topic.
Input C: `asdkjh qwe zzz` — Expected: `status=needs_clarification`.

Observed output A (instant, no model call):
```json
{
  "answer": "Please send a non-empty question.",
  "status": "needs_clarification",
  "citations": [],
  "web_sources": []
}
```

Observed output B (≈7s):
```json
{
  "answer": "I'd be happy to help! You mentioned \"tell me more\" — could you let me know what specific topic you're interested in? The corpus covers:\n\n- **Retrieval-Augmented Generation (RAG)**\n- **Fine-tuning** (LoRA, QLoRA, RLHF, DPO)\n- **Tool calling and agents**\n- **Reasoning and chain-of-thought**\n- **Self-correction / error recovery**\n\nFeel free to ask a specific question about any of these areas!",
  "status": "needs_clarification",
  "citations": [],
  "web_sources": []
}
```

Observed output C (≈3s):
```json
{
  "answer": "I'm sorry, but your input doesn't seem to form a clear question. Could you please rephrase and let me know what you'd like to learn about? My corpus covers research papers on:\n\n- **Retrieval-Augmented Generation (RAG)**\n- **Fine-tuning methods** (LoRA, QLoRA, RLHF, DPO)\n- **Tool calling & agents**\n- **Reasoning & chain-of-thought**\n- **Self-correction & error recovery**\n\nJust ask me a specific question about any of these topics!",
  "status": "needs_clarification",
  "citations": [],
  "web_sources": []
}
```

### 5. Low retrieval confidence

Input: `What exact GPU cluster hardware did the QLoRA authors use for the Guanaco 13B run?`
(a detail unlikely to be in the retained chunks)
Expected: the reranker reports insufficient context. With `WEB_SEARCH_ENABLED=false`:
`status=insufficient_information` and no invented facts. With the fallback on: either an
answer that explicitly says the corpus was not enough and cites web URLs in `web_sources`,
or the same clean refusal if the web search also fails.

Observed output (≈150s with `WEB_SEARCH_ENABLED=true` — the reranker judged the corpus
insufficient, the web fallback ran its bounded 3 attempts without finding the exact
hardware, and the system refused cleanly instead of inventing a GPU model):
```json
{
  "answer": "The indexed papers in this corpus do not contain specific details about the exact GPU cluster hardware used for the Guanaco 13B run. The QLoRA paper mentions that Guanaco models were trained on a single GPU — for example, the abstract notes that Guanaco reached 99.3% of ChatGPT performance \"while only requiring 24 hours of finetuning on a single GPU\" [qlora#001], and elsewhere it mentions training \"in less than 12 hours on a single consumer GPU\" or \"using a single professional GPU\" [qlora#005]. However, the specific make and model of the GPU(s) used for the Guanaco 13B run are not stated in the indexed excerpts.",
  "status": "insufficient_information",
  "citations": [],
  "web_sources": []
}
```

Note: it correctly reports what the corpus *does* say ("a single GPU") while refusing to
name hardware the papers never specify — exactly the no-hallucination behavior required.
