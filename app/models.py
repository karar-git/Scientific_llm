"""Configuration and model factories: chat LLM, embeddings, vector store, chunk store.

Everything reads the environment at call time so settings can be overridden
without import-order tricks.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def env_flag(name: str, default: bool = False) -> bool:
    return env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def chunks_path() -> Path:
    return PROJECT_ROOT / env("CHUNKS_PATH", "data/chunks.jsonl")


def manifest_path() -> Path:
    return PROJECT_ROOT / env("MANIFEST_PATH", "data/papers_manifest.json")


# ---------------------------------------------------------------------------
# Models — everything goes through OpenRouter (chat and embeddings)
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set - copy .env.example to .env and add your key")
    return key


def get_chat_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=env("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"),
        api_key=openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        timeout=150,
    )


def get_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=env("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        api_key=openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        # send raw strings: the tiktoken token-array preflight only works on OpenAI itself
        check_embedding_ctx_length=False,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=env("CHROMA_COLLECTION", "papers"),
        embedding_function=get_embeddings(),
        persist_directory=str(PROJECT_ROOT / env("CHROMA_DIR", ".chroma")),
        collection_metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Chunk store (the processed corpus, kept in memory for keyword search
# and neighbor expansion)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_chunks_cached(path: str) -> tuple[list[dict], dict[str, dict]]:
    chunks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    by_id = {c["chunk_id"]: c for c in chunks}
    return chunks, by_id


def load_chunks() -> tuple[list[dict], dict[str, dict]]:
    """Return (all chunks in corpus order, chunk_id -> chunk)."""
    path = chunks_path()
    if not path.exists():
        return [], {}
    return _load_chunks_cached(str(path))


def reset_chunk_cache() -> None:
    _load_chunks_cached.cache_clear()
