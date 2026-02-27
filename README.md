# Document RAG System

High-precision Retrieval-Augmented Generation system for financial/legal document Q&A, powered by OpenAI.

## Architecture

```
documents/ → doc_converter.py → markdown_output/ → ingest.py → vector_store/
                                                                     ↓
                                         query.py ← RAG Pipeline ← retrieval
                                                    (hybrid search + rerank + grounded generation)
```

### Precision Features

| Feature                    | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| **Hierarchical Chunking**  | Preserves document structure (headers, tables) as metadata |
| **Hybrid Retrieval**       | Combines semantic (embeddings) + keyword (BM25) search     |
| **Reciprocal Rank Fusion** | Merges ranked lists from both retrievers                   |
| **LLM Reranking**          | Uses GPT-4o-mini to filter irrelevant results              |
| **Citation Grounding**     | System prompt forces citations and prevents hallucination  |
| **Low Temperature**        | 0.1 for factual precision                                  |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Create the documents folder and add your files
mkdir -p documents
#    Copy your files into it (supported: PDF, DOCX, PPTX, HTML, XLSX)
cp /path/to/your/files documents/

# 4. Convert documents to markdown
python doc_converter.py

# 5. Ingest markdown into vector store
python ingest.py            # First time
python ingest.py --reset    # Re-ingest from scratch

# 6. Query the system
python query.py "What is the total revenue for 2024?"
python query.py --interactive  # Interactive mode (stateless)
python query.py --chat         # Continuous chat (keeps conversation history)
```

> **Important:** You must place your source documents in the `documents/` folder and run `python doc_converter.py` before ingestion. The converter produces markdown files in `markdown_output/`, which the ingestion pipeline reads.

## Query Modes

### Single question

```bash
python query.py "What is the projected CAGR?"
python query.py --no-rerank "question"     # Skip LLM reranking
python query.py --json "question"          # Raw JSON output
```

### Interactive mode (stateless)

Each question is independent — no memory of previous turns.

```bash
python query.py --interactive
```

### Continuous chat mode (with history)

Maintains conversation history so you can ask follow-up questions that reference earlier answers (e.g., "Can you elaborate on that?" or "What about the next year?").

```bash
python query.py --chat
```

**In-chat commands:**

| Command    | Description                |
| ---------- | -------------------------- |
| `/clear`   | Clear conversation history |
| `/history` | Show conversation history  |
| `quit`     | Exit the session           |

History is automatically capped at the last 20 turns to avoid token overflow.

## Evaluation

```bash
# Generate synthetic Q&A evaluation dataset
python run_eval.py --generate

# Run evaluation against the dataset
python run_eval.py --evaluate

# Both in one step
python run_eval.py --generate --evaluate

# Async mode (faster — concurrent LLM calls)
python run_eval.py --generate --evaluate --async

# Custom number of questions per document
python run_eval.py --generate --evaluate --num-questions 20
```

### Evaluation Metrics

**Retrieval Quality:**

- **Context Precision@5** — fraction of top-5 retrieved chunks that are relevant
- **Context Recall@5** — fraction of relevant chunks found in top-5
- **Hit Rate@5** — whether any relevant chunk appears in top-5
- **MRR** — reciprocal rank of first relevant chunk

**Answer Quality (LLM-judged):**

- **Faithfulness** — is every claim grounded in the retrieved context?
- **Answer Relevance** — does the answer actually address the question?
- **Correctness** — does the answer match the ground truth?

## Project Structure

```
├── config.py                 # Central configuration (all tunable params)
├── doc_converter.py          # Document → Markdown converter (docling)
├── ingest.py                 # Ingestion entry point
├── query.py                  # Query entry point (single/interactive/chat)
├── run_eval.py               # Evaluation entry point
├── requirements.txt
│
├── documents/                # ⬅ Put your source files here (PDF, DOCX, PPTX, etc.)
├── markdown_output/          # Converted markdown documents (auto-generated)
├── vector_store/             # ChromaDB persistence (auto-created)
├── eval_data/                # Evaluation datasets and reports
│
├── rag/                      # Core RAG modules
│   ├── chunker.py            # Hierarchical markdown chunker
│   ├── embeddings.py         # OpenAI embeddings wrapper
│   ├── vector_store.py       # ChromaDB persistent vector store
│   ├── retriever.py          # Hybrid retriever (semantic + BM25 + rerank)
│   ├── generator.py          # Citation-grounded answer generator
│   └── pipeline.py           # Pipeline orchestrator
│
└── eval/                     # Evaluation framework
    ├── generate_eval_data.py # Synthetic Q&A dataset generation
    ├── metrics.py            # Retrieval & answer quality metrics
    └── evaluator.py          # Full evaluation harness
```

## Configuration

All parameters are in `config.py`. Key tuning knobs:

| Parameter              | Default                | Description                       |
| ---------------------- | ---------------------- | --------------------------------- |
| `chunk_size`           | 1024                   | Characters per chunk              |
| `chunk_overlap`        | 200                    | Overlap between chunks            |
| `embedding_model`      | text-embedding-3-large | OpenAI embedding model            |
| `embedding_dimensions` | 1536                   | Embedding vector size             |
| `semantic_top_k`       | 20                     | Initial semantic retrieval count  |
| `bm25_top_k`           | 20                     | Initial BM25 retrieval count      |
| `rerank_top_k`         | 5                      | Final results after LLM reranking |
| `generator_model`      | gpt-4o                 | Model for answer generation       |
| `temperature`          | 0.1                    | Low for factual precision         |
