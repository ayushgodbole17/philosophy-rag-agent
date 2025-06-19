#!/usr/bin/env python3
import os, asyncio
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain.schema import SystemMessage
from langgraph.graph import StateGraph, END

# ── Load keys ──
load_dotenv()  # expects OPENAI_API_KEY in .env

# ── Load local Chroma vector store ─
CHROMA_DIR = "chroma_db/"
emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=emb
)

# ── Build RetrievalQA ─
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        #max_tokens=250,                     # ← hard cap on output tokens
        openai_api_key=os.getenv("OPENAI_API_KEY")
    ),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# ── Shared LLM for loop ─
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    #max_tokens=150,                         # ← same cap here
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ── State schema ─
class LoopState(TypedDict):
    thesis: str
    argument: str
    critique: str
    iter: int
    max_iter: int

# ── Node functions with concise prompts ─

def gen(state: LoopState) -> dict:
    docs = qa.retriever.invoke(state["thesis"])
    ctx  = "\n---\n".join(d.page_content for d in docs)
    prompt = (
        f"Context:\n{ctx}\n\n"
        "You're a precise philosopher—be concise (≤150 tokens).\n"
        f"Thesis:\n{state['thesis']}\nGenerate a clear, logical argument."
    )
    resp = llm.invoke([SystemMessage(content=prompt)])
    return {"argument": resp.content}

def crit(state: LoopState) -> dict:
    docs = qa.retriever.invoke(state["argument"])
    ctx  = "\n---\n".join(d.page_content for d in docs)
    prompt = (
        f"Context:\n{ctx}\n\n"
        "You're sharply analytical—be concise (≤150 tokens).\n"
        f"Argument:\n{state['argument']}\nPoint out any flaws."
    )
    resp = llm.invoke([SystemMessage(content=prompt)])
    return {"critique": resp.content}

def ref(state: LoopState) -> dict:
    docs = qa.retriever.invoke(state["critique"])
    ctx  = "\n---\n".join(d.page_content for d in docs)
    prompt = (
        f"Context:\n{ctx}\n\n"
        "Revise ruthlessly—be concise (≤250 tokens).\n"
        f"Original argument:\n{state['argument']}\n\n"
        f"Critique:\n{state['critique']}\nNow improve the argument."
    )
    resp = llm.invoke([SystemMessage(content=prompt)])
    return {"argument": resp.content, "iter": state["iter"] + 1}

def decide(state: LoopState) -> str:
    return "crit" if state["iter"] < state["max_iter"] else END

# ── Build & compile graph ─
graph = StateGraph(LoopState)
graph.add_node("gen", gen)
graph.add_node("crit", crit)
graph.add_node("ref", ref)
graph.set_entry_point("gen")
graph.add_edge("gen", "crit")
graph.add_edge("crit", "ref")
graph.add_conditional_edges("ref", decide, {"crit": "crit", END: END})
compiled = graph.compile()

# ── Runner with streaming ─
async def main():
    init = {
        "thesis": "What would Aristotle say about Universal Basic Income?",
        "argument": "",
        "critique": "",
        "iter": 0,
        "max_iter": 2
    }
    async for update in compiled.astream(init, stream_mode="updates"):
        print(update)
    final = compiled.invoke(init)
    print("\nFinal Argument:\n", final["argument"])
    print("\nFinal Critique:\n", final["critique"])

if __name__ == "__main__":
    asyncio.run(main())
