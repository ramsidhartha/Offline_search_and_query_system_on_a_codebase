"""
RAG Engine Module - Hierarchical Retrieval with Parent-Child Strategy

Handles vector storage, MMR search, parent document elevation, and LLM generation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


class ThinkingTrace:
    """Collects and formats the thinking trace for display."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps = []
    
    def add(self, emoji: str, message: str):
        """Add a step to the trace."""
        step = f"{emoji} {message}"
        self.steps.append(step)
        if self.verbose:
            print(step)
    
    def get_trace(self) -> str:
        """Return full trace as string."""
        return "\n".join(self.steps)


class RAGEngine:
    """
    Main RAG engine with parent-child retrieval strategy.
    
    Architecture:
    - Child Store: 400-token chunks for precise search
    - Parent Store: Full entities for complete context
    - MMR Search: Diverse retrieval (k=4, fetch_k=20)
    - Parent Elevation: Child match → lookup parent → full context to LLM
    """
    
    SYSTEM_PROMPT = """You are a code analysis assistant. Answer questions using ONLY the provided context.
Always cite your sources using the format [file_path:line_number].
Be precise and reference specific functions, classes, and line numbers when relevant.
If the context doesn't contain enough information to answer, say so clearly."""

    CONTEXT_TEMPLATE = """Context: [{entity_type}] '{entity_name}' in {file_path} (Lines {start_line}-{end_line})
{parent_info}{docstring_info}{callers_info}---
{content}
"""

    def __init__(
        self,
        base_dir: str = ".",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "gemma2:2b",
        verbose: bool = True
    ):
        self.base_dir = Path(base_dir)
        self.chroma_dir = self.base_dir / "chroma_db"
        self.verbose = verbose
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize LLM
        self.llm = ChatOllama(model=llm_model, temperature=0)
        
        # Vector stores (lazy initialization)
        self._parent_store: Optional[Chroma] = None
        self._child_store: Optional[Chroma] = None
        
        # Parent document cache for elevation
        self._parent_cache: Dict[str, Document] = {}
    
    @property
    def parent_store(self) -> Chroma:
        """Get or create parent document store."""
        if self._parent_store is None:
            self._parent_store = Chroma(
                collection_name="parent_documents",
                embedding_function=self.embeddings,
                persist_directory=str(self.chroma_dir / "parent")
            )
        return self._parent_store
    
    @property
    def child_store(self) -> Chroma:
        """Get or create child chunk store."""
        if self._child_store is None:
            self._child_store = Chroma(
                collection_name="child_documents",
                embedding_function=self.embeddings,
                persist_directory=str(self.chroma_dir / "child")
            )
        return self._child_store
    
    def _truncate_for_embedding(self, doc: Document, max_chars: int = 6000) -> Document:
        """
        Truncate document content for embedding while preserving metadata.
        nomic-embed-text has ~8192 token limit, so we truncate to ~6000 chars for safety.
        """
        if len(doc.page_content) <= max_chars:
            return doc
        
        # Truncate content but keep a note
        truncated_content = doc.page_content[:max_chars] + "\n... [truncated for embedding]"
        return Document(
            page_content=truncated_content,
            metadata=doc.metadata
        )
    
    def index_documents(
        self,
        parent_docs: List[Document],
        child_docs: List[Document]
    ) -> Dict[str, int]:
        """
        Index parent and child documents into ChromaDB.
        
        Returns:
            Dict with counts of indexed documents
        """
        trace = ThinkingTrace(self.verbose)
        
        # Clear existing stores for fresh indexing
        trace.add("[CLEAR]", "Clearing existing vector stores...")
        
        parent_dir = self.chroma_dir / "parent"
        child_dir = self.chroma_dir / "child"
        
        if parent_dir.exists():
            import shutil
            shutil.rmtree(parent_dir)
        if child_dir.exists():
            import shutil
            shutil.rmtree(child_dir)
        
        # Reset store references
        self._parent_store = None
        self._child_store = None
        
        # Build parent cache BEFORE truncation (keep full content for retrieval)
        for doc in parent_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                self._parent_cache[parent_id] = doc
        
        # Truncate parent docs for embedding (nomic-embed-text has ~8k token limit)
        trace.add("[INDEX]", f"Indexing {len(parent_docs)} parent documents...")
        
        if parent_docs:
            truncated_parents = [self._truncate_for_embedding(doc) for doc in parent_docs]
            self.parent_store.add_documents(truncated_parents)
        
        # Index child documents (already small enough)
        trace.add("[INDEX]", f"Indexing {len(child_docs)} child chunks...")
        
        if child_docs:
            self.child_store.add_documents(child_docs)
        
        trace.add("[DONE]", "Indexing complete!")
        
        return {
            "parent_count": len(parent_docs),
            "child_count": len(child_docs)
        }
    
    def _format_context(self, doc: Document) -> str:
        """Format a document with contextual header including docstring and parent info."""
        metadata = doc.metadata
        
        # Build parent class info
        parent_class = metadata.get("parent_class", "")
        parent_info = ""
        if parent_class:
            parent_info = f"Parent Class: {parent_class}\n"
        
        # Build docstring info
        docstring = metadata.get("docstring", "")
        docstring_info = ""
        if docstring:
            docstring_info = f"Description: {docstring}\n"
        
        # Build callers info
        calls = metadata.get("calls", "")
        callers_info = ""
        if calls:
            callers_info = f"Calls: {calls}\n"
        
        return self.CONTEXT_TEMPLATE.format(
            entity_type=metadata.get("entity_type", "Unknown"),
            entity_name=metadata.get("entity_name", "unknown"),
            file_path=metadata.get("file_path", "unknown"),
            start_line=metadata.get("start_line", 0),
            end_line=metadata.get("end_line", 0),
            parent_info=parent_info,
            docstring_info=docstring_info,
            callers_info=callers_info,
            content=doc.page_content
        )
    
    def _elevate_to_parent(self, child_doc: Document, trace: ThinkingTrace) -> Document:
        """Elevate a child chunk to its parent document."""
        parent_id = child_doc.metadata.get("parent_id")
        
        if parent_id and parent_id in self._parent_cache:
            parent = self._parent_cache[parent_id]
            entity_name = parent.metadata.get("entity_name", "unknown")
            trace.add("[ELEVATE]", f"Elevating to Parent Context ({entity_name})...")
            return parent
        
        # If parent not in cache, try to fetch from store
        if parent_id:
            results = self.parent_store.get(where={"parent_id": parent_id})
            if results and results.get("documents"):
                # Reconstruct document
                parent = Document(
                    page_content=results["documents"][0],
                    metadata=results["metadatas"][0] if results.get("metadatas") else {}
                )
                self._parent_cache[parent_id] = parent
                entity_name = parent.metadata.get("entity_name", "unknown")
                trace.add("[ELEVATE]", f"Elevating to Parent Context ({entity_name})...")
                return parent
        
        # Fallback: return child as-is
        return child_doc
    
    def search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20
    ) -> Tuple[List[Document], ThinkingTrace]:
        """
        Search for relevant code using MMR on children, then elevate to parents.
        
        Args:
            query: Natural language query
            k: Number of results to return
            fetch_k: Number of candidates to fetch for MMR
        
        Returns:
            Tuple of (relevant parent documents, thinking trace)
        """
        trace = ThinkingTrace(self.verbose)
        
        trace.add("[SEARCH]", "Searching vector space using MMR...")
        
        # MMR search on child chunks
        child_results = self.child_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k
        )
        
        # Elevate each child to its parent
        seen_parents = set()
        parent_results = []
        
        for child in child_results:
            file_path = child.metadata.get("file_path", "unknown")
            trace.add("[FOUND]", f"Found relevant child snippet in {file_path}...")
            
            parent = self._elevate_to_parent(child, trace)
            parent_id = parent.metadata.get("parent_id")
            
            # Deduplicate parents
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_results.append(parent)
        
        return parent_results, trace
    
    def query(self, question: str, k: int = 4, fetch_k: int = 20) -> str:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: Natural language question
            k: Number of context documents
            fetch_k: MMR fetch candidates
        
        Returns:
            Generated answer with citations
        """
        # Search for relevant context
        relevant_docs, trace = self.search(question, k=k, fetch_k=fetch_k)
        
        if not relevant_docs:
            return "No relevant code found in the indexed repository."
        
        # Format context with headers
        formatted_contexts = [self._format_context(doc) for doc in relevant_docs]
        context_str = "\n\n".join(formatted_contexts)
        
        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{context}\n\nQuestion: {question}\nAnswer:")
        ])
        
        # Generate response
        trace.add("[GENERATE]", "Generating response with gemma2:2b...")
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context_str,
            "question": question
        })
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get current index status."""
        try:
            parent_count = self.parent_store._collection.count()
        except:
            parent_count = 0
        
        try:
            child_count = self.child_store._collection.count()
        except:
            child_count = 0
        
        return {
            "parent_documents": parent_count,
            "child_chunks": child_count,
            "chroma_dir": str(self.chroma_dir),
            "embedding_model": "nomic-embed-text",
            "llm_model": "gemma2:2b"
        }
