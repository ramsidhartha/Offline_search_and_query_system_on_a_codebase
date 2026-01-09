# 🧠 Structural Code Intelligence System

An offline RAG system with AST-aware chunking, hierarchical retrieval, and citation-grounded answers.

## ✨ Features

- **AST-Surgical Slicing**: Parses Python with `ast` module, JS/CPP with language-aware splitters
- **Parent-Child Retrieval**: Search fine-grained chunks, retrieve full function context
- **MMR Diversity**: Maximal Marginal Relevance (k=4, fetch_k=20) for diverse results
- **100% Offline**: Runs entirely locally with Ollama (gemma2:2b + nomic-embed-text)
- **Citation Grounded**: Every answer cites file paths and line numbers

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)

### Setup (One-time)

```bash
# 1. Clone this repository
git clone https://github.com/your-username/codebase-intelligence-system.git
cd codebase-intelligence-system

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull required Ollama models (~1.9GB total)
ollama pull nomic-embed-text
ollama pull gemma2:2b
```

### Usage

```bash
# Ingest a GitHub repository
python3 main.py ingest https://github.com/username/repo

# Ingest a local directory
python3 main.py ingest ./my-project

# Ask questions about the code
python3 main.py query "Where is the database connection initialized?"

# Check index status
python3 main.py status
```

## 📖 Example Output

```
❓ Query: How is the CNN model defined?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Thinking Trace:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 Searching vector space using MMR...
📄 Found relevant child snippet in custom_cnn.py...
⬆️ Elevating to Parent Context (build_custom_cnn)...
🤖 Generating response with gemma2:2b...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 Response:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The CNN model is defined in `build_custom_cnn` [custom_cnn.py:12-84].
It uses 3 convolutional blocks with MaxPooling2D...
```

## 🏗️ Architecture

```
Query → MMR Search (children) → Parent Elevation → LLM → Cited Answer
           ↓                         ↓
     ChromaDB (k=4)          Full function context
```

## 📁 Project Structure

```
├── main.py              # CLI interface with thinking trace
├── requirements.txt     # Python dependencies
├── core/
│   ├── ingestor.py      # AST parsing + parent-child chunking
│   └── rag_engine.py    # Vector search + hierarchical retrieval
├── temp_repo/           # Cloned repositories (auto-created)
└── chroma_db/           # Persistent vector storage (auto-created)
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Chunk size | 400 tokens | Child chunk size for search |
| MMR k | 4 | Number of results returned |
| MMR fetch_k | 20 | Candidates for diversity calculation |
| Embedding model | nomic-embed-text | 768-dimensional embeddings |
| LLM model | gemma2:2b | Local generation model |

## 📊 Performance

- **Ingestion**: ~10-30s depending on repo size
- **Query latency**: ~15s (local LLM inference)
- **Index persistence**: Survives restarts via ChromaDB

## 🔬 Research Foundations

This system implements:
- **MMR**: λ-weighted diversity-relevance tradeoff
- **HNSW**: Approximate nearest neighbor via hierarchical graphs
- **Parent-Child Retrieval**: Two-tier information hierarchy
- **AST as DAG**: Directed acyclic graph for code structure

## 📝 License

MIT
