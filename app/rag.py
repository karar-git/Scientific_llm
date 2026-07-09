"""Question answering: a LangGraph agent whose only tool is the RAG pipeline.

Flow: the main LLM sees the user question first. Small talk, out-of-scope and
unanswerable-vague inputs are handled directly (tagged so the API can report a
status). Real questions trigger the search_papers tool, which runs vector
similarity + keyword search over the chunk corpus, expands each hit with its
neighboring chunks, and has a reranker LLM select the best passages and judge
whether they are sufficient. If they are not, an optional web-search fallback
(DuckDuckGo, up to 3 attempts) tries to fill the gap; web material is clearly
separated from corpus citations. The agent then writes the final grounded
answer and lists the chunk ids it used.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.models import env, env_flag, get_chat_model, get_vector_store, load_chunks

SYSTEM_PROMPT = """You are the research assistant for a corpus of ~30 research papers about large language models: retrieval-augmented generation, fine-tuning (LoRA, QLoRA, RLHF, DPO), tool calling and agents, reasoning and chain-of-thought, and self-correction / error recovery.

Routing rules — apply the FIRST one that matches:
1. Greeting or small talk (hi, thanks, how are you): reply briefly and invite a question about the corpus. Start your reply with the exact tag [SMALL_TALK]
2. A real question that is clearly UNRELATED to the corpus topics (sports, cooking, politics, ...): politely decline and say what the corpus covers. Start with the tag [OUT_OF_SCOPE]
3. Empty, gibberish, or too vague to search (e.g. "tell me more" with no topic): ask what exactly they want to know. Start with the tag [CLARIFY]
4. Anything else: you MUST call the search_papers tool. Write `query` as a self-contained English search question (rewrite vague or non-English questions into precise English). Put exact technical terms worth exact-matching (model names, method names, e.g. "LoRA", "ReAct") into `keywords`. Call the tool at most twice.

When you answer after a tool call:
- Use ONLY the retrieved passages. Never add facts from your own general knowledge.
- Answer in the same language the user asked in (e.g. Arabic question -> Arabic answer), even though passages are English.
- Cite passages inline with their chunk id in square brackets, e.g. [lora#004].
- If the tool reported web results and you use them, say clearly that the corpus was not enough and this part comes from a web search, citing the URL inline.
- If the retrieval status is insufficient, say plainly that the indexed papers do not contain enough information to answer. Do not guess.
- End your reply with one final line exactly like: SOURCES: lora#004, qlora#011 (the corpus chunk ids you actually used, or SOURCES: none)."""

RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You rerank retrieval results. Given a question and candidate passages, pick the passages that actually help answer it (best first, at most {max_selected}), and judge whether they are sufficient to answer the question fully. Only mark sufficient=false when key information is genuinely missing."),
    ("human", "Question: {question}\n\nCandidate passages:\n{candidates}"),
])

WEB_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You judge web search results. Decide whether the snippets contain enough information to answer the question. If not, propose one better search query to try instead."),
    ("human", "Question: {question}\n\nSearch results:\n{results}"),
])


class RerankResult(BaseModel):
    selected_chunk_ids: list[str] = Field(description="Chunk ids that help answer the question, best first.")
    sufficient: bool = Field(description="True if the selected passages fully answer the question.")
    reason: str = ""


class WebJudge(BaseModel):
    sufficient: bool
    better_query: str = ""


class RAGState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved: list[dict]        # chunks handed to the answering LLM
    selected_ids: list[str]      # what the reranker picked (before neighbor expansion)
    retrieval_status: str        # sufficient | web_fallback | insufficient
    web_sources: list[dict]


# ---------------------------------------------------------------------------
# Retrieval building blocks (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def keyword_search(chunks: list[dict], keywords: list[str], limit: int = 5) -> list[dict]:
    """Rank chunks by exact keyword occurrences (case-insensitive, word-boundary)."""
    patterns = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keywords if k.strip()]
    if not patterns:
        return []
    scored = []
    for chunk in chunks:
        score = sum(len(p.findall(chunk["text"])) for p in patterns)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: -pair[0])
    return [chunk for _, chunk in scored[:limit]]


def expand_neighbors(selected: list[dict], all_chunks: dict[str, dict]) -> list[dict]:
    """Return the selected chunks, each with its previous and next chunk from the same paper."""
    out, seen = [], set()
    for chunk in selected:
        for index in (chunk["chunk_index"] - 1, chunk["chunk_index"], chunk["chunk_index"] + 1):
            neighbor = all_chunks.get(f"{chunk['source_id']}#{index:03d}")
            if neighbor and neighbor["chunk_id"] not in seen:
                seen.add(neighbor["chunk_id"])
                out.append(neighbor)
    return out


def format_chunk(chunk: dict, max_chars: int | None = None) -> str:
    text = chunk["text"] if max_chars is None else chunk["text"][:max_chars]
    return f"[{chunk['chunk_id']}] {chunk['title']} — {chunk['section']}\n{text}"


def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS

        with DDGS() as search:
            hits = list(search.text(query, max_results=max_results))
        return [{"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")} for h in hits]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# The search_papers tool: hybrid retrieval -> rerank -> optional web fallback
# ---------------------------------------------------------------------------

def search_papers_impl(query: str, keywords: list[str]) -> dict:
    """Run the full retrieval pipeline. Returns the formatted context text plus
    everything the graph needs later to build citations."""
    chunks, by_id = load_chunks()
    llm = get_chat_model()

    if not chunks:
        return {"text": "RETRIEVAL STATUS: insufficient\nThe corpus index is empty. Run preprocessing first.",
                "retrieved": [], "selected_ids": [], "status": "insufficient", "web_sources": []}

    # 1. Candidate gathering: vector similarity + exact keyword matches.
    k = min(int(env("RETRIEVAL_K", "10")), len(chunks))
    vector_hits = [by_id[d.metadata["chunk_id"]] for d in get_vector_store().similarity_search(query, k=k) if d.metadata.get("chunk_id") in by_id]
    candidates: dict[str, dict] = {c["chunk_id"]: c for c in vector_hits}
    for chunk in keyword_search(chunks, keywords):
        candidates.setdefault(chunk["chunk_id"], chunk)

    # 2. Rerank: an LLM picks the best passages and judges sufficiency.
    listing = "\n\n".join(format_chunk(c, max_chars=700) for c in candidates.values())
    rerank = (RERANK_PROMPT | llm.with_structured_output(RerankResult)).invoke({
        "question": query, "candidates": listing, "max_selected": env("RERANK_K", "4"),
    })
    selected = [candidates[cid] for cid in rerank.selected_chunk_ids if cid in candidates][: int(env("RERANK_K", "4"))]
    status = "sufficient" if (selected and rerank.sufficient) else "insufficient"

    # 3. Web fallback: only when the corpus is not enough and the feature is on.
    web_sources: list[dict] = []
    if status == "insufficient" and env_flag("WEB_SEARCH_ENABLED", True):
        web_query = query
        for _ in range(3):
            results = web_search(web_query)
            if not results:
                break
            judged = (WEB_JUDGE_PROMPT | llm.with_structured_output(WebJudge)).invoke({
                "question": query,
                "results": "\n\n".join(f"[{r['url']}] {r['title']}\n{r['snippet']}" for r in results),
            })
            if judged.sufficient:
                web_sources, status = results, "web_fallback"
                break
            web_query = judged.better_query or web_query

    # 4. Context for the answering LLM: selected chunks plus their neighbors.
    retrieved = expand_neighbors(selected, by_id)
    parts = [f"RETRIEVAL STATUS: {status}"]
    if retrieved:
        parts.append("CORPUS PASSAGES:\n" + "\n\n".join(format_chunk(c) for c in retrieved))
    if web_sources:
        parts.append("WEB RESULTS (NOT from the indexed corpus):\n" + "\n\n".join(
            f"[{r['url']}] {r['title']}\n{r['snippet']}" for r in web_sources))
    if status == "insufficient":
        parts.append("The corpus does not contain enough information to answer this question.")

    return {
        "text": "\n\n".join(parts),
        "retrieved": retrieved,
        "selected_ids": [c["chunk_id"] for c in selected],
        "status": status,
        "web_sources": web_sources,
    }


@tool
def search_papers(query: str, keywords: list[str] = []) -> str:
    """Search the indexed research-paper corpus.

    query: a self-contained English search question.
    keywords: exact technical terms from the user's question worth exact-matching.
    """
    return search_papers_impl(query, keywords)["text"]


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def agent_node(state: RAGState) -> dict:
    llm = get_chat_model().bind_tools([search_papers])
    reply = llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [reply]}


def retrieve_node(state: RAGState) -> dict:
    # Runs search_papers_impl directly (instead of the plain tool wrapper) so the
    # retrieved chunks and status land in graph state for citation building.
    call = state["messages"][-1].tool_calls[0]
    result = search_papers_impl(call["args"].get("query", ""), call["args"].get("keywords") or [])
    return {
        "messages": [ToolMessage(result["text"], tool_call_id=call["id"])],
        "retrieved": result["retrieved"],
        "selected_ids": result["selected_ids"],
        "retrieval_status": result["status"],
        "web_sources": result["web_sources"],
    }


def route_after_agent(state: RAGState) -> str:
    last = state["messages"][-1]
    already_searched = sum(isinstance(m, ToolMessage) for m in state["messages"])
    if getattr(last, "tool_calls", None) and already_searched < 2:
        return "retrieve"
    return "end"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("agent", agent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"retrieve": "retrieve", "end": END})
    graph.add_edge("retrieve", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

TAGS = {"[SMALL_TALK]": "small_talk", "[OUT_OF_SCOPE]": "out_of_scope", "[CLARIFY]": "needs_clarification"}
SOURCES_LINE = re.compile(r"\n?\s*SOURCES:\s*(.*)\s*$", re.IGNORECASE)
CHUNK_ID = re.compile(r"\b([a-z0-9-]+#\d{3})\b")


def citation_for(chunk: dict) -> dict:
    return {key: chunk[key] for key in ("chunk_id", "source_id", "title", "section", "source_url")}


def answer_question(question: str) -> dict:
    question = (question or "").strip()
    if not question:
        return {"answer": "Please send a non-empty question.", "status": "needs_clarification",
                "citations": [], "web_sources": []}

    state = build_graph().invoke({"messages": [HumanMessage(question)]}, config={"recursion_limit": 12})
    last = state["messages"][-1]
    answer = str(last.content).strip() if isinstance(last, AIMessage) else ""

    for tag, status in TAGS.items():
        if answer.startswith(tag):
            return {"answer": answer[len(tag):].strip(), "status": status, "citations": [], "web_sources": []}

    _, by_id = load_chunks()
    cited_ids: list[str] = []
    match = SOURCES_LINE.search(answer)
    if match:
        cited_ids = [cid for cid in CHUNK_ID.findall(match.group(1)) if cid in by_id]
        answer = answer[: match.start()].strip()
    if not cited_ids:
        cited_ids = [cid for cid in state.get("selected_ids", []) if cid in by_id]

    status = state.get("retrieval_status", "")
    if status == "insufficient":
        return {"answer": answer or "The indexed papers do not contain enough information to answer that.",
                "status": "insufficient_information", "citations": [], "web_sources": []}

    return {
        "answer": answer,
        "status": "answered",
        "citations": [citation_for(by_id[cid]) for cid in cited_ids],
        "web_sources": state.get("web_sources", []),
    }
