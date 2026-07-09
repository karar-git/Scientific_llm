"""Unit tests for the retrieval building blocks (pure Python, no model calls)."""

from app.rag import answer_question, expand_neighbors, keyword_search


def record(source_id: str, index: int, text: str) -> dict:
    return {
        "chunk_id": f"{source_id}#{index:03d}",
        "source_id": source_id,
        "title": source_id.upper(),
        "section": "Method",
        "source_url": f"https://arxiv.org/abs/{source_id}",
        "topic": "test",
        "chunk_index": index,
        "text": text,
    }


CHUNKS = [
    record("lora", 0, "LoRA freezes the pretrained weights and injects low-rank matrices."),
    record("lora", 1, "The rank r is much smaller than the model dimension."),
    record("lora", 2, "LoRA reduces trainable parameters by 10,000 times."),
    record("react", 0, "ReAct interleaves reasoning traces with task-specific actions."),
]
BY_ID = {c["chunk_id"]: c for c in CHUNKS}


def test_keyword_search_ranks_by_exact_matches():
    hits = keyword_search(CHUNKS, ["LoRA"])
    assert hits and all(h["source_id"] == "lora" for h in hits)
    assert keyword_search(CHUNKS, ["reasoning traces"]) == [BY_ID["react#000"]]
    assert keyword_search(CHUNKS, ["nonexistent-term-xyz"]) == []
    assert keyword_search(CHUNKS, []) == []


def test_expand_neighbors_adds_previous_and_next_chunk():
    expanded = expand_neighbors([BY_ID["lora#001"]], BY_ID)
    assert [c["chunk_id"] for c in expanded] == ["lora#000", "lora#001", "lora#002"]
    # a chunk at the edge of a paper only gets the neighbors that exist
    expanded = expand_neighbors([BY_ID["react#000"]], BY_ID)
    assert [c["chunk_id"] for c in expanded] == ["react#000"]


def test_empty_question_short_circuits_without_any_model_call():
    result = answer_question("   ")
    assert result["status"] == "needs_clarification"
    assert result["citations"] == []
