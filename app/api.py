"""HTTP API: POST /ask answers a question, GET /health reports index status.

Indexing runs as a setup step (python -m app.preprocess), so the API itself
stays read-only. Run with: uvicorn app.api:app --reload
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer_question
from app.models import load_chunks

logger = logging.getLogger("swibit.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Swibit Knowledge Assistant",
    description="RAG API over ~30 research papers about LLMs (RAG, fine-tuning, tool calling, reasoning, self-correction).",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(max_length=2000, description="Natural-language question. May be in any language.")


class Citation(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    section: str
    source_url: str


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "small_talk", "out_of_scope", "needs_clarification", "insufficient_information"]
    citations: list[Citation]
    web_sources: list[WebSource] = []


@app.get("/health")
def health() -> dict:
    chunks, _ = load_chunks()
    return {"status": "ok", "chunks_indexed": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    chunks, _ = load_chunks()
    if not chunks and request.question.strip():
        raise HTTPException(status_code=503, detail="The corpus is not indexed yet. Run: python -m app.preprocess")
    try:
        result = answer_question(request.question)
    except Exception:
        logger.exception("answering failed")
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")
    logger.info("question answered status=%s citations=%d", result["status"], len(result["citations"]))
    return AskResponse(**result)
