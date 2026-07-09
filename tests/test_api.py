"""API integration tests against the real indexed corpus and live LLM.

Prerequisites: the corpus is indexed (python -m app.preprocess) and an API key
is set in .env. The whole module skips with a clear message until then.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.models import chunks_path

if not chunks_path().exists():
    pytest.skip("corpus not indexed - run: python -m app.preprocess", allow_module_level=True)
if not os.getenv("OPENROUTER_API_KEY"):
    pytest.skip("OPENROUTER_API_KEY not set in .env", allow_module_level=True)

from app.api import app

client = TestClient(app)


def test_health_reports_index_size():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0


def test_in_scope_question_is_answered_with_citations():
    response = client.post("/ask", json={"question": "How does LoRA reduce the number of trainable parameters?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"]
    assert body["citations"][0]["source_url"].startswith("https://arxiv.org/abs/")
    assert "SOURCES:" not in body["answer"]          # the citation line is parsed out


def test_small_talk_is_answered_without_retrieval():
    body = client.post("/ask", json={"question": "hi!"}).json()
    assert body["status"] == "small_talk"
    assert body["citations"] == []


def test_out_of_scope_question_is_refused():
    body = client.post("/ask", json={"question": "Who won the 2022 FIFA World Cup final?"}).json()
    assert body["status"] == "out_of_scope"
    assert body["citations"] == []


def test_empty_question_needs_clarification():
    body = client.post("/ask", json={"question": ""}).json()
    assert body["status"] == "needs_clarification"


def test_oversized_question_is_rejected():
    assert client.post("/ask", json={"question": "x" * 3000}).status_code == 422
