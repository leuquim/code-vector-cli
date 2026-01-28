"""AST-aware code chunking using tree-sitter"""

import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tree_sitter_languages import get_language, get_parser


class CodeChunk:
    """Represents a chunk of code with metadata"""

    def __init__(
        self,
        content: str,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        name: Optional[str] = None,
        parent: Optional[str] = None,
        language: Optional[str] = None,
        **extra_metadata
    ):
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type  # function, class, file, etc.
        self.name = name
        self.parent = parent
        self.language = language
        self.extra_metadata = extra_metadata
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        content_str = f"{self.file_path}:{self.start_line}:{self.content}"
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "content": self.content,
            "metadata": {
                "file_path": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "type": self.chunk_type,
                "name": self.name or "",
                "parent": self.parent or "",
                "language": self.language or "",
                "content_hash": self.hash,
                **self.extra_metadata
            }
        }


class ASTChunker:
    """AST-aware code chunker using tree-sitter"""

    # Language extensions mapping
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".cs": "c_sharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
    }

    # Node types to extract as chunks
    FUNCTION_NODES = {
        "function_definition", "function_declaration", "method_definition",
        "function_item", "function", "method", "constructor"
    }

    # Arrow functions and function expressions (JS/TS)
    ARROW_FUNCTION_NODES = {
        "arrow_function", "function_expression", "function"
    }

    # Variable declarations that may contain arrow functions
    VARIABLE_DECLARATION_NODES = {
        "lexical_declaration", "variable_declaration"
    }

    CLASS_NODES = {
        "class_definition", "class_declaration", "interface_declaration",
        "struct_item", "impl_item", "trait_item"
    }

    # Minimum lines for a function to be indexed (filter out tiny callbacks)
    MIN_FUNCTION_LINES = 5

    def __init__(self):
        self.parsers = {}
        self.languages = {}

    def get_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension"""
        suffix = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(suffix)

    def get_parser(self, language: str):
        """Get or create parser for language"""
        if language not in self.parsers:
            try:
                self.languages[language] = get_language(language)
                self.parsers[language] = get_parser(language)
            except Exception as e:
                return None
        return self.parsers[language]

    def chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Chunk a file into semantic units"""
        language = self.get_language(file_path)
        if not language:
            # Non-code file, return single chunk
            return [CodeChunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=content.count('\n') + 1,
                chunk_type="file",
                language="text"
            )]

        parser = self.get_parser(language)
        if not parser:
            # Parser not available, return single chunk
            return [CodeChunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=content.count('\n') + 1,
                chunk_type="file",
                language=language
            )]

        # Parse the file
        tree = parser.parse(bytes(content, "utf8"))
        chunks = []

        # Extract functions and classes
        self._extract_nodes(tree.root_node, content, file_path, language, chunks)

        total_lines = content.count('\n') + 1

        # Calculate how much of the file is covered by extracted chunks
        covered_lines = sum(c.end_line - c.start_line + 1 for c in chunks)
        coverage_ratio = covered_lines / total_lines if total_lines > 0 else 1.0

        # Add file-level chunk if:
        # 1. No chunks extracted, OR
        # 2. Large file (100+ lines) with low coverage (<50% of lines covered)
        # This ensures big React components and other code don't slip through
        if not chunks or (total_lines >= 100 and coverage_ratio < 0.5):
            chunks.append(CodeChunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                chunk_type="file",
                language=language
            ))

        return chunks

    def _extract_nodes(
        self,
        node,
        content: str,
        file_path: str,
        language: str,
        chunks: List[CodeChunk],
        parent: Optional[str] = None
    ):
        """Recursively extract function and class nodes"""
        node_type = node.type

        # Check if this is a function or class node
        if node_type in self.FUNCTION_NODES:
            name = self._get_node_name(node, content)
            chunk_content = self._get_node_text(node, content)
            start_line, end_line = self._get_node_lines(node)

            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type="function",
                name=name,
                parent=parent,
                language=language,
                complexity=self._estimate_complexity(node)
            ))

        # Handle variable declarations containing arrow functions (React components, etc.)
        elif node_type in self.VARIABLE_DECLARATION_NODES:
            arrow_func = self._find_arrow_function_in_declaration(node, content, file_path, language)
            if arrow_func:
                chunks.append(arrow_func)
            # Still recurse to find nested functions
            for child in node.children:
                self._extract_nodes(child, content, file_path, language, chunks, parent)
            return

        elif node_type in self.CLASS_NODES:
            name = self._get_node_name(node, content)
            chunk_content = self._get_node_text(node, content)
            start_line, end_line = self._get_node_lines(node)

            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type="class",
                name=name,
                parent=parent,
                language=language
            ))

            # Recurse into class to find methods
            for child in node.children:
                self._extract_nodes(child, content, file_path, language, chunks, parent=name)
            return

        # Recurse into children
        for child in node.children:
            self._extract_nodes(child, content, file_path, language, chunks, parent)

    def _find_arrow_function_in_declaration(
        self,
        node,
        content: str,
        file_path: str,
        language: str
    ) -> Optional[CodeChunk]:
        """
        Extract arrow function or function expression from variable declaration.
        Handles patterns like: const MyComponent = (props) => { ... }
        """
        for child in node.children:
            if child.type == "variable_declarator":
                name = None
                func_node = None

                for declarator_child in child.children:
                    if declarator_child.type == "identifier":
                        name = self._get_node_text(declarator_child, content)
                    elif declarator_child.type in self.ARROW_FUNCTION_NODES:
                        func_node = declarator_child

                if name and func_node:
                    start_line, end_line = self._get_node_lines(node)
                    num_lines = end_line - start_line + 1

                    # Only index functions with meaningful size (skip tiny callbacks)
                    if num_lines >= self.MIN_FUNCTION_LINES:
                        chunk_content = self._get_node_text(node, content)
                        return CodeChunk(
                            content=chunk_content,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_type="function",
                            name=name,
                            parent="",
                            language=language,
                            complexity=self._estimate_complexity(func_node)
                        )
        return None

    def _get_node_name(self, node, content: str) -> str:
        """Extract the name of a function or class"""
        for child in node.children:
            if child.type in ["identifier", "name", "property_identifier"]:
                return self._get_node_text(child, content)
        return "anonymous"

    def _get_node_text(self, node, content: str) -> str:
        """Get the text content of a node"""
        return content[node.start_byte:node.end_byte]

    def _get_node_lines(self, node) -> Tuple[int, int]:
        """Get start and end line numbers (1-indexed)"""
        return node.start_point[0] + 1, node.end_point[0] + 1

    def _estimate_complexity(self, node) -> int:
        """Estimate cyclomatic complexity"""
        complexity = 1
        keywords = {"if", "elif", "else", "for", "while", "case", "catch", "&&", "||"}

        def count_keywords(n):
            nonlocal complexity
            if n.type in keywords:
                complexity += 1
            for child in n.children:
                count_keywords(child)

        count_keywords(node)
        return complexity
