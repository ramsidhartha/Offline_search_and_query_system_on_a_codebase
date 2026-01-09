"""
Core Ingestor Module - AST-Surgical Slicing

Handles repository cloning, AST-aware code parsing, and parent-child document creation.
"""

import ast
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import git
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)


class CodeEntity:
    """Represents a code entity (function, class) with its metadata."""
    
    def __init__(
        self,
        name: str,
        entity_type: str,
        content: str,
        file_path: str,
        start_line: int,
        end_line: int,
        calls: Optional[List[str]] = None,
        docstring: Optional[str] = None,
        parent_class: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.entity_type = entity_type
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.calls = calls or []
        self.docstring = docstring
        self.parent_class = parent_class
    
    @property
    def qualified_name(self) -> str:
        """Return fully qualified name (ClassName.method_name)."""
        if self.parent_class:
            return f"{self.parent_class}.{self.name}"
        return self.name
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert entity to metadata dictionary."""
        return {
            "parent_id": self.id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "entity_type": self.entity_type,
            "entity_name": self.qualified_name,
            "calls": ",".join(self.calls) if self.calls else "",
            "docstring": (self.docstring[:200] + "...") if self.docstring and len(self.docstring) > 200 else (self.docstring or ""),
            "parent_class": self.parent_class or ""
        }


class CallVisitor(ast.NodeVisitor):
    """AST visitor to extract function calls from a node."""
    
    def __init__(self):
        self.calls = []
    
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)


class PythonASTExtractor:
    """Extract code entities from Python files using AST with hierarchy support."""
    
    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.splitlines(keepends=True)
    
    def _get_source_segment(self, start_line: int, end_line: int) -> str:
        """Extract source code between line numbers (1-indexed)."""
        return "".join(self.lines[start_line - 1:end_line])
    
    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a function or class."""
        return ast.get_docstring(node)
    
    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from an AST node."""
        visitor = CallVisitor()
        visitor.visit(node)
        return list(set(visitor.calls))
    
    def _extract_from_class(self, class_node: ast.ClassDef) -> List[CodeEntity]:
        """Extract class and its methods as separate entities."""
        entities = []
        
        # Add the class itself
        class_entity = CodeEntity(
            name=class_node.name,
            entity_type="ClassDef",
            content=self._get_source_segment(class_node.lineno, class_node.end_lineno),
            file_path=self.file_path,
            start_line=class_node.lineno,
            end_line=class_node.end_lineno,
            calls=self._extract_calls(class_node),
            docstring=self._extract_docstring(class_node)
        )
        entities.append(class_entity)
        
        # Add methods with parent reference
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_entity = CodeEntity(
                    name=node.name,
                    entity_type="MethodDef",
                    content=self._get_source_segment(node.lineno, node.end_lineno),
                    file_path=self.file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    calls=self._extract_calls(node),
                    docstring=self._extract_docstring(node),
                    parent_class=class_node.name
                )
                entities.append(method_entity)
        
        return entities
    
    def extract_entities(self) -> List[CodeEntity]:
        """Parse Python file and extract all functions, classes, and methods."""
        entities = []
        
        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            # If parsing fails, treat entire file as one entity
            return [CodeEntity(
                name="module",
                entity_type="Module",
                content=self.content,
                file_path=self.file_path,
                start_line=1,
                end_line=len(self.lines),
                calls=[]
            )]
        
        # Process top-level nodes only (not walking entire tree)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class and its methods
                entities.extend(self._extract_from_class(node))
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Top-level function
                entity = CodeEntity(
                    name=node.name,
                    entity_type="FunctionDef",
                    content=self._get_source_segment(node.lineno, node.end_lineno),
                    file_path=self.file_path,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    calls=self._extract_calls(node),
                    docstring=self._extract_docstring(node)
                )
                entities.append(entity)
        
        # If no entities found, treat entire file as module
        if not entities:
            entities.append(CodeEntity(
                name="module",
                entity_type="Module",
                content=self.content,
                file_path=self.file_path,
                start_line=1,
                end_line=len(self.lines),
                calls=[]
            ))
        
        return entities


class LanguageAwareExtractor:
    """Extract entities from JS/CPP using LangChain language splitters."""
    
    LANGUAGE_MAP = {
        ".js": Language.JS,
        ".jsx": Language.JS,
        ".ts": Language.TS,
        ".cpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".hpp": Language.CPP,
        ".h": Language.CPP,
    }
    
    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.extension = Path(file_path).suffix.lower()
    
    def extract_entities(self) -> List[CodeEntity]:
        """Use language-aware splitting to create logical chunks."""
        language = self.LANGUAGE_MAP.get(self.extension)
        
        if language is None:
            return []
        
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=2000,
            chunk_overlap=200
        )
        
        chunks = splitter.split_text(self.content)
        entities = []
        
        current_line = 1
        for i, chunk in enumerate(chunks):
            # Estimate line numbers based on content
            chunk_lines = chunk.count('\n') + 1
            
            # Try to extract entity name from first line
            first_line = chunk.split('\n')[0].strip()
            entity_name = self._extract_entity_name(first_line) or f"chunk_{i}"
            entity_type = self._infer_entity_type(first_line)
            
            entity = CodeEntity(
                name=entity_name,
                entity_type=entity_type,
                content=chunk,
                file_path=self.file_path,
                start_line=current_line,
                end_line=current_line + chunk_lines - 1,
                calls=[]
            )
            entities.append(entity)
            
            # Update line counter (approximate due to overlap)
            current_line += chunk_lines - 10  # Account for overlap
        
        return entities
    
    def _extract_entity_name(self, line: str) -> Optional[str]:
        """Try to extract function/class name from a line."""
        import re
        
        # JS/TS patterns
        patterns = [
            r'function\s+(\w+)',
            r'class\s+(\w+)',
            r'const\s+(\w+)\s*=',
            r'(\w+)\s*:\s*function',
            r'async\s+function\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        
        return None
    
    def _infer_entity_type(self, line: str) -> str:
        """Infer entity type from first line."""
        if 'class ' in line:
            return "ClassDef"
        elif 'function ' in line or '=>' in line:
            return "FunctionDef"
        else:
            return "CodeBlock"


class CodebaseIngestor:
    """Main ingestor class for cloning and processing repositories."""
    
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.cpp', '.cc', '.cxx', '.hpp', '.h'}
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.temp_repo_dir = self.base_dir / "temp_repo"
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def clone_repository(self, url: str) -> Path:
        """Clone a GitHub repository to temp_repo directory."""
        # Clean existing repo
        if self.temp_repo_dir.exists():
            shutil.rmtree(self.temp_repo_dir)
        
        self.temp_repo_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Cloning repository: {url}")
        git.Repo.clone_from(url, self.temp_repo_dir)
        print(f"[OK] Repository cloned to: {self.temp_repo_dir}")
        
        return self.temp_repo_dir
    
    def ingest_local_directory(self, directory: str) -> Path:
        """Ingest a local directory (for self-referential testing)."""
        source = Path(directory).resolve()
        
        # If it's already the temp_repo, use it directly
        if source == self.temp_repo_dir.resolve():
            return self.temp_repo_dir
        
        # Clean and copy
        if self.temp_repo_dir.exists():
            shutil.rmtree(self.temp_repo_dir)
        
        shutil.copytree(source, self.temp_repo_dir, ignore=shutil.ignore_patterns(
            'temp_repo', 'chroma_db', '__pycache__', '*.pyc', '.git', 'node_modules'
        ))
        
        print(f"[OK] Local directory copied to: {self.temp_repo_dir}")
        return self.temp_repo_dir
    
    def discover_files(self) -> List[Path]:
        """Walk directory and find supported code files."""
        files = []
        
        for root, _, filenames in os.walk(self.temp_repo_dir):
            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(file_path)
        
        print(f"📁 Discovered {len(files)} code files")
        return files
    
    def extract_entities_from_file(self, file_path: Path) -> List[CodeEntity]:
        """Extract code entities from a single file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            print(f"[WARNING] Could not read {file_path}: {e}")
            return []
        
        relative_path = str(file_path.relative_to(self.temp_repo_dir))
        
        if file_path.suffix.lower() == '.py':
            extractor = PythonASTExtractor(relative_path, content)
        else:
            extractor = LanguageAwareExtractor(relative_path, content)
        
        return extractor.extract_entities()
    
    def create_parent_documents(self, entities: List[CodeEntity]) -> List[Document]:
        """Create parent documents from code entities."""
        documents = []
        
        for entity in entities:
            doc = Document(
                page_content=entity.content,
                metadata=entity.to_metadata()
            )
            documents.append(doc)
        
        return documents
    
    def create_child_documents(self, entities: List[CodeEntity]) -> List[Document]:
        """Create child chunks from code entities for fine-grained search."""
        documents = []
        
        for entity in entities:
            chunks = self.child_splitter.split_text(entity.content)
            entity_lines = entity.content.splitlines()
            total_lines = len(entity_lines)
            
            for i, chunk in enumerate(chunks):
                # Estimate line range for this chunk within the entity
                chunk_lines = chunk.count('\n') + 1
                approx_start_offset = int((i / max(len(chunks), 1)) * total_lines)
                chunk_start = entity.start_line + approx_start_offset
                chunk_end = min(chunk_start + chunk_lines - 1, entity.end_line)
                
                # Inherit parent metadata with chunk-specific overrides
                metadata = entity.to_metadata()
                metadata["chunk_index"] = i
                metadata["chunk_count"] = len(chunks)
                metadata["is_child"] = True
                metadata["chunk_start_line"] = chunk_start
                metadata["chunk_end_line"] = chunk_end
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)
        
        return documents
    
    def process_repository(self, url_or_path: str) -> Tuple[List[Document], List[Document]]:
        """
        Main entry point: Clone/copy repo and extract all documents.
        
        Returns:
            Tuple of (parent_documents, child_documents)
        """
        # Clone or copy
        if url_or_path.startswith(('http://', 'https://', 'git@')):
            self.clone_repository(url_or_path)
        else:
            self.ingest_local_directory(url_or_path)
        
        # Discover files
        files = self.discover_files()
        
        # Extract entities from all files
        all_entities = []
        for file_path in files:
            entities = self.extract_entities_from_file(file_path)
            all_entities.extend(entities)
            print(f"  - {file_path.name}: {len(entities)} entities")
        
        print(f"\nTotal entities extracted: {len(all_entities)}")
        
        # Create documents
        parent_docs = self.create_parent_documents(all_entities)
        child_docs = self.create_child_documents(all_entities)
        
        print(f"Parent documents: {len(parent_docs)}")
        print(f"Child documents: {len(child_docs)}")
        
        return parent_docs, child_docs


# Utility function for direct entity extraction
def extract_entities(file_path: str) -> List[Dict[str, Any]]:
    """Extract entities from a file and return as dictionaries."""
    path = Path(file_path)
    content = path.read_text(encoding='utf-8', errors='ignore')
    
    if path.suffix.lower() == '.py':
        extractor = PythonASTExtractor(str(path), content)
    else:
        extractor = LanguageAwareExtractor(str(path), content)
    
    entities = extractor.extract_entities()
    return [e.to_metadata() for e in entities]
