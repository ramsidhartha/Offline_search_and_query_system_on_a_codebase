#!/usr/bin/env python3
"""
Structural Code Intelligence System - CLI Interface

A production-grade offline codebase intelligence system with AST-aware chunking
and hierarchical retrieval.

Usage:
    python main.py ingest <github_url_or_path>  - Clone/copy and index repository
    python main.py query "<question>"           - Ask questions about the code
    python main.py status                       - Show index status
"""

import sys
import argparse
from pathlib import Path

from core.ingestor import CodebaseIngestor
from core.rag_engine import RAGEngine


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║         🧠 Structural Code Intelligence System                   ║
║         AST-Aware • Offline • Citation-Grounded                  ║
╚═══════════════════════════════════════════════════════════════════╝
""")


def print_separator():
    """Print visual separator."""
    print("━" * 60)


def cmd_ingest(args):
    """Handle ingest command."""
    print_banner()
    print(f"📥 Target: {args.target}\n")
    
    base_dir = Path(__file__).parent
    
    # Initialize components
    ingestor = CodebaseIngestor(base_dir=str(base_dir))
    engine = RAGEngine(base_dir=str(base_dir))
    
    # Process repository
    print_separator()
    print("Phase 1: Repository Processing")
    print_separator()
    
    parent_docs, child_docs = ingestor.process_repository(args.target)
    
    # Index documents
    print()
    print_separator()
    print("Phase 2: Vector Indexing")
    print_separator()
    
    stats = engine.index_documents(parent_docs, child_docs)
    
    print()
    print_separator()
    print("✅ Ingestion Complete!")
    print_separator()
    print(f"   Parent Entities: {stats['parent_count']}")
    print(f"   Child Chunks:    {stats['child_count']}")
    print(f"   Storage:         {engine.chroma_dir}")
    print()


def cmd_query(args):
    """Handle query command."""
    print_banner()
    
    base_dir = Path(__file__).parent
    engine = RAGEngine(base_dir=str(base_dir))
    
    # Check if index exists
    status = engine.get_status()
    if status['parent_documents'] == 0:
        print("❌ No documents indexed. Run 'ingest' first.")
        return
    
    print(f"❓ Query: {args.question}\n")
    print_separator()
    print("Thinking Trace:")
    print_separator()
    
    # Execute query (thinking trace prints automatically)
    response = engine.query(args.question, k=4, fetch_k=20)
    
    print()
    print_separator()
    print("📝 Response:")
    print_separator()
    print()
    print(response)
    print()


def cmd_status(args):
    """Handle status command."""
    print_banner()
    
    base_dir = Path(__file__).parent
    engine = RAGEngine(base_dir=str(base_dir), verbose=False)
    
    status = engine.get_status()
    
    print_separator()
    print("📊 Index Status")
    print_separator()
    print(f"   Parent Entities:  {status['parent_documents']}")
    print(f"   Child Chunks:     {status['child_chunks']}")
    print(f"   Storage Path:     {status['chroma_dir']}")
    print(f"   Embedding Model:  {status['embedding_model']}")
    print(f"   LLM Model:        {status['llm_model']}")
    print()
    
    if status['parent_documents'] == 0:
        print("⚠️  No documents indexed. Run 'ingest <url>' to get started.")
    else:
        print("✅ System ready for queries.")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Structural Code Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Ingest a GitHub repository:
    python main.py ingest https://github.com/user/repo

  Ingest local directory:
    python main.py ingest ./my-project

  Query the indexed codebase:
    python main.py query "Where is the database connection initialized?"

  Check index status:
    python main.py status
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Clone and index a repository')
    ingest_parser.add_argument('target', help='GitHub URL or local directory path')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask questions about the code')
    query_parser.add_argument('question', help='Natural language question')
    query_parser.set_defaults(func=cmd_query)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show index status')
    status_parser.set_defaults(func=cmd_status)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
