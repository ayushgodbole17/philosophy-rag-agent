# Philosophy RAG Agent

A Retrieval-Augmented Generation (RAG) agent that:

- **Ingests** PDFs of philosophical works into a local Chroma vector store
- **Indexes** and **retrieves** relevant passages at query time
- **Runs** a three-step Thought‑Looper (generate → critique → refine) using LangGraph
- **Offers** an interactive CLI to ask philosophical questions and see arguments & critiques

---

## Files

- **build\_chroma.py**:\
  Reads `.pdf` files from `data/`, chunks them, embeds with OpenAI embeddings, and persists a Chroma database in `chroma_db/`.

- **meta\_agent.py**:\
  Loads the Chroma store, sets up a `RetrievalQA` chain and a LangGraph loop, then starts an interactive REPL.\
  For each question:

  1. Retrieves top-3 passages
  2. Prints source filenames
  3. Generates an argument, critique, and refined argument

---

## Prerequisites

- Python 3.9+
- A `.env` file with:
  ```bash
  OPENAI_API_KEY=sk-...
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Setup & Usage

1. **Build the vector store**:

   ```bash
   python build_chroma.py
   ```

   Chunks all PDFs in `data/` and persists embeddings in `chroma_db/`.

2. **Run the agent**:

   ```bash
   python meta_agent.py
   ```

   Follow the prompt to ask philosophical questions (or type `exit`).

---

## Project Structure

```plaintext
├── data/                # Input PDF files
├── build_chroma.py      # Builds the local Chroma DB
├── chroma_db/           # Persisted vector store
├── meta_agent.py        # Interactive Philosophy RAG agent
├── .env                 # OPENAI_API_KEY
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## License

MIT © Ayush Girish Godbole

