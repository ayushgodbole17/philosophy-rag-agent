#!/usr/bin/env python3
import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, List, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, END

# ── Configuration ──
load_dotenv()  # expects OPENAI_API_KEY in your .env
CHROMA_DIR = "chroma_db/"
MODEL      = "gpt-4o"
TEMP       = 0.5
CONTEXT_K  = 3
MAX_ITER   = 2

# ── Helpers ──
def init_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        temperature=TEMP,
        openai_api_key=os.getenv("OPENAI_API_KEY")
        # no max_tokens → unlimited output
    )

def fetch_context_with_sources(retriever, query: str) -> List[Tuple[str, str]]:
    """Return list of (source_filename, snippet) for top-k docs."""
    docs = retriever.invoke(query)
    return [(doc.metadata.get("source", "unknown.pdf"), doc.page_content) for doc in docs]

def format_context(entries: List[Tuple[str, str]]) -> str:
    """Concatenate snippets with source tags for prompt context."""
    return "\n---\n".join(f"[{src}] {text}" for src, text in entries)

def print_sources(entries: List[Tuple[str, str]]):
    """Print which source files were referenced."""
    files = {src for src, _ in entries}
    print("⮕ Referenced:", ", ".join(files))

# ── Build RAG components ──
emb        = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore= Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
qa         = RetrievalQA.from_chain_type(
    llm=init_llm(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": CONTEXT_K})
)

# ── Thought‐Looper Setup ──
class LoopState(TypedDict):
    thesis: str
    argument: str
    critique: str
    iter: int
    max_iter: int

# instantiate three separate LLMs for clarity
gen_llm  = init_llm()
crit_llm = init_llm()
ref_llm  = init_llm()

def gen(state: LoopState) -> dict:
    entries = fetch_context_with_sources(qa.retriever, state["thesis"])
    print_sources(entries)
    ctx    = format_context(entries)
    prompt = (
        f"Context:\n{ctx}\n\n"
        "You're a precise philosopher—be concise.\n"
        f"Thesis:\n{state['thesis']}\nGenerate a clear, logical argument."
    )
    resp = gen_llm.invoke([SystemMessage(content=prompt)])
    return {"argument": resp.content}

def crit(state: LoopState) -> dict:
    entries = fetch_context_with_sources(qa.retriever, state["argument"])
    print_sources(entries)
    ctx    = format_context(entries)
    prompt = (
        f"Context:\n{ctx}\n\n"
        "You're sharply analytical—be concise.\n"
        f"Argument:\n{state['argument']}\nPoint out any flaws."
    )
    resp = crit_llm.invoke([SystemMessage(content=prompt)])
    return {"critique": resp.content}

def ref(state: LoopState) -> dict:
    entries = fetch_context_with_sources(qa.retriever, state["critique"])
    print_sources(entries)
    ctx    = format_context(entries)
    prompt = (
        f"Context:\n{ctx}\n\n"
        "Revise ruthlessly—be concise.\n"
        f"Original argument:\n{state['argument']}\n\n"
        f"Critique:\n{state['critique']}\nNow improve the argument."
    )
    resp = ref_llm.invoke([SystemMessage(content=prompt)])
    return {"argument": resp.content, "iter": state["iter"] + 1}

def decide(state: LoopState) -> str:
    if "no flaw" in state["critique"].lower():
        return END
    return "crit" if state["iter"] < state["max_iter"] else END

# build and compile the graph
graph = StateGraph(LoopState)
graph.add_node("gen", gen)
graph.add_node("crit", crit)
graph.add_node("ref", ref)
graph.set_entry_point("gen")
graph.add_edge("gen", "crit")
graph.add_edge("crit", "ref")
graph.add_conditional_edges("ref", decide, {"crit": "crit", END: END})
compiled = graph.compile()

# ── Interactive Chat Loop ──
async def main():
    print("=== Philosophy RAG Agent ===")
    while True:
        user_q = input("\nAsk a philosophical question (or type 'exit'): ").strip()
        if user_q.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # ensure RAG context exists
        entries = fetch_context_with_sources(qa.retriever, user_q)
        if not entries:
            print("🤖 Please ask only philosophy-related questions.")
            continue

        # run thought looper
        initial = {
            "thesis":   user_q,
            "argument": "",
            "critique": "",
            "iter":     0,
            "max_iter": MAX_ITER
        }
        final = compiled.invoke(initial)

        # display results
        print("\n⮕ Argument:\n", final["argument"])
        print("\n⮕ Critique:\n", final["critique"])

if __name__ == "__main__":
    asyncio.run(main())
