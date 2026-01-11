# Structural Code Intelligence System - Architecture & Decisions

## System Data Flows

The system has two parallel data flows that meet at ChromaDB:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 INGESTION FLOW (runs once per repo)                 │
│                                                                     │
│  Codebase ──► AST Parse ──► Split ──► Embed ──► ChromaDB (write)   │
│     │                         │                      │              │
│  .py/.js     FunctionDef   Parent    nomic-embed   Parent Store    │
│  files       ClassDef      Child       768D        Child Store     │
└─────────────────────────────────────────────────────────────────────┘
                                           │
                                    [persistent storage]
                                           │
┌─────────────────────────────────────────────────────────────────────┐
│                  QUERY FLOW (runs per question)                     │
│                                                                     │
│  Question ──► Embed ──► MMR Search ──► Elevate ──► LLM ──► Answer  │
│     │          │            │            │          │         │     │
│  "How is    nomic-embed  ChromaDB     Parent    gemma2:2b  Cited   │
│  CNN built?"   768D      (k=4)        lookup              response │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Both flows are **separate** — you ingest once, query many times.

---

## 1. Architectural Decisions

### 1.1 AST-Surgical Slicing (vs Naive Chunking)

| Decision | Why |
|----------|-----|
| Use Python `ast` module | Exact line boundaries for functions/classes |
| Use `ast.iter_child_nodes()` | Top-level traversal avoids duplicates |
| Separate `MethodDef` from `FunctionDef` | Track `parent_class` for qualified names |
| Extract docstrings | Richer metadata for context headers |
| Track function calls | Dependency awareness in metadata |

**Tradeoff**: AST parsing fails on syntax errors → fallback to treating file as single entity.

---

### 1.2 Parent-Child Retrieval Hierarchy

| Decision | Why |
|----------|-----|
| 400-token children | Small enough for precise semantic matching |
| Full-entity parents | Complete context for LLM generation |
| Search children, retrieve parents | Best of both worlds |
| Cache parents in memory | Fast elevation without DB lookup |

**The Chunking Dilemma:**
- Small chunks → high search precision, but fragmented context
- Large chunks → complete context, but noisy embeddings
- **Solution**: Two-tier hierarchy separates search from retrieval

---

### 1.3 MMR (Maximal Marginal Relevance)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `k` | 4 | Return 4 diverse results |
| `fetch_k` | 20 | Fetch 20 candidates for diversity calculation |
| `lambda` | 0.5 (default) | Balanced relevance-diversity |

**Why MMR over top-k?**
- Top-k returns redundant snippets from same function
- MMR penalizes similarity to already-selected documents
- Result: Diverse context from different functions/files

---

### 1.4 Offline-First with Ollama

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embeddings | nomic-embed-text | 768D, runs locally, good for code |
| LLM | gemma2:2b | Smallest model, fits on any machine |
| Vector DB | ChromaDB | Persistent local storage, no server |

**Tradeoff**: ~15s query latency vs zero cloud dependency.

---

## 2. Problems Encountered & Solutions

### 2.1 Context Length Exceeded

**Problem**: nomic-embed-text has ~8192 token limit. Large classes exceeded this.

**Error**: `the input length exceeds the context length (status code: 400)`

**Solution**: 
```python
def _truncate_for_embedding(self, doc, max_chars=6000):
    if len(doc.page_content) <= max_chars:
        return doc
    truncated = doc.page_content[:max_chars] + "\n... [truncated]"
    return Document(page_content=truncated, metadata=doc.metadata)
```

**Key insight**: Cache full parent BEFORE truncation → retrieve complete context later.

---

### 2.2 Retrieval Misses (Dropout Rate Query)

**Problem**: Query about "dropout rate in custom CNN" returned wrong files.

**Root cause**: MMR diversified across files instead of concentrating on `custom_cnn.py`.

**Analysis**: Diversity is a tradeoff — sometimes you want concentration.

**Potential solutions** (not implemented):
- Hybrid search: Add keyword filter on filename
- Re-ranking: Second pass scoring by relevance
- Query expansion: Detect entity names and boost

---

### 2.3 Duplicate Parent Elevations

**Problem**: Same parent elevated multiple times from different child chunks.

**Solution**: Track `seen_parents` set, deduplicate before sending to LLM.

```python
seen_parents = set()
for child in child_results:
    parent = self._elevate_to_parent(child)
    if parent_id not in seen_parents:
        seen_parents.add(parent_id)
        parent_results.append(parent)
```

---

### 2.4 LangChain Deprecation Warning

**Problem**: `Chroma` class deprecated in LangChain 0.2.9.

**Solution**: Migrated from `langchain_community.vectorstores` to `langchain_chroma`.

---

## 3. Key Metrics

| Metric | Value |
|--------|-------|
| Flower-classification entities | 21 parents, 88 children |
| Self-referential entities | 51 parents, 181 children |
| Query accuracy | 6/7 (86%) on hard questions |
| Query latency | ~15s (local gemma2:2b) |
| Ingestion time | ~30s for small repos |

---

## 4. Research Foundations

| Concept | Paper/Source |
|---------|--------------|
| MMR | Carbonell & Goldstein, 1998 |
| HNSW | Malkov & Yashunin, 2016 |
| Transformers | Vaswani et al., 2017 |
| RAG | Lewis et al., 2020 |

---

