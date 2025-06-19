#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ── Load API key from .env ──
load_dotenv()  # ensures OPENAI_API_KEY is available

# ── Paths ──
BOOKS_DIR  = "data/"       # folder of PDFs
CHROMA_DIR = "chroma_db/"   # where Chroma persists

# ── Load & chunk PDFs ──
loader = PyPDFDirectoryLoader(BOOKS_DIR)  # imports from langchain-community to avoid deprecation :contentReference[oaicite:0]{index=0}
docs   = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# ── Embed & persist ──
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  # updated import per v0.2.0 :contentReference[oaicite:1]{index=1}
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)  # Chroma from langchain_community to suppress deprecation :contentReference[oaicite:2]{index=2}

db.persist()
print(f"✅ Built Chroma DB with {len(chunks)} chunks at '{CHROMA_DIR}'")
