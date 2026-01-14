"""Main indexer for codebase"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import subprocess
from multiprocessing import Pool, cpu_count
from functools import partial

from .ast_chunker import ASTChunker, CodeChunk
from .embeddings import get_code_embedder, get_text_embedder
from .vector_store import VectorStore
from .metadata import ProjectMetadata


# Global worker function for multiprocessing (needs to be at module level)
# Global chunker instance for workers
_worker_chunker = None

def _process_file_worker(file_info: Tuple[Path, Path]) -> Optional[Tuple[List, List, List]]:
    """
    Worker function to process a single file in parallel
    Returns (function_chunks, class_chunks, file_chunks) or None on error
    """
    global _worker_chunker
    file_path, project_path = file_info

    try:
        content = file_path.read_text(errors='ignore')
        # Normalize path separators for cross-platform compatibility
        rel_path = str(file_path.relative_to(project_path)).replace('\\', '/')

        # Initialize chunker if not present (reuse across calls in same process)
        if _worker_chunker is None:
            _worker_chunker = ASTChunker()
            
        chunks = _worker_chunker.chunk_file(rel_path, content)

        # Separate by type
        function_chunks = [c for c in chunks if c.chunk_type == "function"]
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        file_chunks = [c for c in chunks if c.chunk_type == "file"]

        return (function_chunks, class_chunks, file_chunks)
    except Exception as e:
        return None


class CodebaseIndexer:
    """
    Unified indexer for any directory structure.

    Automatically detects:
    - Single git repo: indexes as-is
    - Multiple git repos: adds repo metadata to distinguish sources
    - Non-git directory: indexes all files respecting .gitignore
    """

    # File patterns to ignore
    DEFAULT_IGNORE_PATTERNS = {
        "node_modules", "venv", ".venv", "env", ".env",
        "dist", "build", ".git", ".svn", ".hg",
        "__pycache__", ".pytest_cache", ".mypy_cache",
        "vendor", "target", ".idea", ".vscode",
        "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll",
        ".DS_Store", "Thumbs.db"
    }

    # Documentation file extensions
    DOC_EXTENSIONS = {".md", ".rst", ".txt", ".adoc"}

    # Configuration file extensions
    CONFIG_EXTENSIONS = {
        ".json", ".yaml", ".yml", ".toml", ".ini",
        ".conf", ".config", ".env.example"
    }

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.workspace_root = self.project_path  # Save original for multi-repo mode
        self.vector_store = VectorStore(str(self.project_path))
        self.metadata = ProjectMetadata()
        self.ast_chunker = ASTChunker()
        self.code_embedder = get_code_embedder()
        self.text_embedder = get_text_embedder()

        # Auto-detect repository structure
        self.repos = self._discover_repos()
        self.is_multi_repo = len(self.repos) > 1

        # Log configuration
        embedder_type = type(self.code_embedder).__name__
        print(f"  Using embedder: {embedder_type}")

        if self.is_multi_repo:
            print(f"  Multi-repo mode: {len(self.repos)} repositories detected")
            for repo in self.repos:
                print(f"    - {repo['name']}")
        elif self.repos:
            print(f"  Single-repo mode: {self.repos[0]['name']}")
        else:
            print(f"  Non-git mode: indexing all files")

        # Load .gitignore patterns
        self.ignore_patterns = self._load_gitignore()
        self.ignore_patterns.update(self.DEFAULT_IGNORE_PATTERNS)

    def _discover_repos(self) -> List[Dict]:
        """
        Discover git repositories in the index root.

        Returns list of dicts with 'name' and 'path' keys.
        - If root is a git repo: returns single repo (self)
        - If root contains git repos: returns all subdirs with .git
        - Otherwise: returns empty list (non-git mode)
        """
        repos = []

        # Check if root itself is a git repo
        if (self.project_path / ".git").exists():
            repos.append({
                "name": self.project_path.name,
                "path": self.project_path,
                "is_root": True
            })
        else:
            # Look for git repos in subdirectories
            for item in self.project_path.iterdir():
                if item.is_dir() and (item / ".git").exists():
                    repos.append({
                        "name": item.name,
                        "path": item,
                        "is_root": False
                    })

        return sorted(repos, key=lambda r: r['name'])

    def _load_gitignore(self) -> Set[str]:
        """Load patterns from .gitignore"""
        patterns = set()
        gitignore_path = self.project_path / ".gitignore"

        if gitignore_path.exists():
            with open(gitignore_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.add(line)

        return patterns

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        path_str = str(path.relative_to(self.project_path)).replace('\\', '/')

        # Check against gitignore patterns using git check-ignore
        try:
            result = subprocess.run(
                ["git", "check-ignore", str(path)],
                cwd=self.project_path,
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                return True
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            # Git not available or timeout - fall back to pattern matching
            pass

        # Check against default patterns
        for pattern in self.ignore_patterns:
            if pattern in path_str or path.name == pattern:
                return True

        return False

    def _get_all_files(self) -> List[Path]:
        """Get all files to index using git ls-files"""
        files = []

        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        file_path = self.project_path / line
                        if file_path.exists() and file_path.is_file():
                            files.append(file_path)
                return files
        except Exception as e:
            print(f"Warning: git ls-files failed ({e}), falling back to rglob with gitignore")

        for path in self.project_path.rglob("*"):
            if path.is_file() and not self._should_ignore(path):
                files.append(path)
        return files

    def initialize(self):
        """Initialize vector database collections"""
        print("Initializing vector database collections...")
        self.vector_store.initialize_collections()

    def index_codebase(self, incremental: bool = False, repo_filter: str = None):
        """Index the entire codebase (handles both single and multi-repo)

        Args:
            incremental: Whether to do incremental indexing
            repo_filter: If set, only index the specified repo name in multi-repo mode
        """
        if self.is_multi_repo:
            # Multi-repo mode: index each repo separately with metadata
            self._index_multi_repo(incremental, repo_filter)
        else:
            # Single-repo or non-git mode: index everything directly
            if repo_filter:
                print(f"Warning: --repo filter ignored in single-repo mode")
            self._index_single(incremental)

    def _index_single(self, incremental: bool = False):
        """Index a single directory structure"""
        print(f"\nIndexing codebase: {self.project_path}")
        print(f"Incremental: {incremental}")

        files = self._get_all_files()
        print(f"Found {len(files)} files to process")

        # Group files by type
        code_files = []
        doc_files = []
        config_files = []

        for file_path in files:
            if file_path.suffix in self.DOC_EXTENSIONS:
                doc_files.append(file_path)
            elif file_path.suffix in self.CONFIG_EXTENSIONS:
                config_files.append(file_path)
            elif self.ast_chunker.get_language(str(file_path)):
                code_files.append(file_path)

        print(f"  Code files: {len(code_files)}")
        print(f"  Documentation files: {len(doc_files)}")
        print(f"  Config files: {len(config_files)}")

        # Index code files
        if code_files:
            print("\nIndexing code files...")
            self._index_code_files(code_files, incremental)

        # Index documentation
        if doc_files:
            print("\nIndexing documentation...")
            self._index_documentation(doc_files, incremental)

        # Index configuration
        if config_files:
            print("\nIndexing configuration files...")
            self._index_configuration(config_files, incremental)

        # Register project in metadata
        stats = self.vector_store.get_stats()
        collection_stats = {k: v["points_count"] for k, v in stats["collections"].items()}
        self.metadata.register_project(
            str(self.project_path),
            file_count=len(files),
            collection_stats=collection_stats
        )

        print("\n[OK] Indexing complete!")

    def _index_multi_repo(self, incremental: bool = False, repo_filter: str = None):
        """Index multiple repositories with repo metadata

        Args:
            incremental: Whether to do incremental indexing
            repo_filter: If set, only index repos matching this name
        """
        # Filter repos if specified
        repos_to_index = self.repos
        if repo_filter:
            repos_to_index = [r for r in self.repos if r['name'] == repo_filter]
            if not repos_to_index:
                available = ', '.join(r['name'] for r in self.repos)
                raise ValueError(f"Repository '{repo_filter}' not found. Available: {available}")
            print(f"\nIndexing specific repository: {repo_filter}")
        else:
            print(f"\nIndexing workspace with {len(self.repos)} repositories...")

        print("=" * 60)

        for i, repo_info in enumerate(repos_to_index, 1):
            repo_name = repo_info['name']
            repo_path = repo_info['path']

            total_repos = len(repos_to_index)
            print(f"\n[{i}/{total_repos}] Repository: {repo_name}")
            print("-" * 60)

            try:
                # Temporarily override project_path to index this repo
                original_path = self.project_path
                self.project_path = repo_path

                # Get files from this repo
                files = self._get_all_files()
                print(f"Found {len(files)} files to process")

                # Group files by type
                code_files = []
                doc_files = []
                config_files = []

                for file_path in files:
                    if file_path.suffix in self.DOC_EXTENSIONS:
                        doc_files.append(file_path)
                    elif file_path.suffix in self.CONFIG_EXTENSIONS:
                        config_files.append(file_path)
                    elif self.ast_chunker.get_language(str(file_path)):
                        code_files.append(file_path)

                print(f"  Code files: {len(code_files)}")
                print(f"  Documentation files: {len(doc_files)}")
                print(f"  Config files: {len(config_files)}")

                # Set repo metadata context for this indexing run
                self._current_repo_name = repo_name

                # Index code files
                if code_files:
                    print("\nIndexing code files...")
                    self._index_code_files(code_files, incremental)

                # Index documentation
                if doc_files:
                    print("\nIndexing documentation...")
                    self._index_documentation(doc_files, incremental)

                # Index configuration
                if config_files:
                    print("\nIndexing configuration files...")
                    self._index_configuration(config_files, incremental)

                # Restore original path
                self.project_path = original_path
                self._current_repo_name = None

                print(f"[OK] Indexing complete!")
                print(f"  [OK] Indexed {repo_name}")

            except Exception as e:
                print(f"  [FAIL] Error indexing {repo_name}: {e}")
                self.project_path = original_path
                self._current_repo_name = None
                continue

        print("\n" + "=" * 60)
        print("[OK] Workspace indexing complete!")
        print(f"\nIndexed {len(self.repos)} repositories into shared collections")

        # Register workspace project in metadata
        # Calculate total file count across all repos
        total_files = sum(
            len(list((repo_info['path']).rglob("*")))
            for repo_info in self.repos
            if repo_info['path'].exists()
        )
        stats = self.vector_store.get_stats()
        collection_stats = {k: v["points_count"] for k, v in stats["collections"].items()}
        self.metadata.register_project(
            str(self.project_path),
            file_count=total_files,
            collection_stats=collection_stats
        )
        print(f"\n[OK] Registered workspace in metadata: {self.project_path}")

    def _filter_changed_files(self, files: List[Path]) -> List[Path]:
        """Filter files to only those changed since last index"""
        import hashlib

        # Get last index time from metadata
        # Always use workspace root, even when temporarily indexing sub-repos
        last_indexed = self.metadata.get_last_indexed_time(str(self.workspace_root))

        if not last_indexed:
            # Never indexed before, return all files
            return files

        changed_files = []

        for file_path in files:
            try:
                # Check modification time first (fast)
                mtime = os.path.getmtime(file_path)
                if mtime > last_indexed:
                    # File modified since last index
                    changed_files.append(file_path)
                    continue

                # For files with same mtime, check content hash
                # (catches cases where file was restored to same mtime)
                content = file_path.read_text(errors='ignore')
                content_hash = hashlib.md5(content.encode()).hexdigest()

                # Query vector DB to see if this file exists with different hash
                # This is expensive, so only do it for files with matching mtime
                # For simplicity, we'll skip this check and rely on mtime

            except Exception:
                # If we can't check, include it to be safe
                changed_files.append(file_path)

        return changed_files

    def _index_code_files(self, files: List[Path], incremental: bool):
        """Index code files with parallel processing and batched embedding"""

        # Filter to only changed files if incremental
        if incremental:
            files = self._filter_changed_files(files)
            if not files:
                print("  No files changed since last index")
                return
            print(f"  Incremental mode: {len(files)} files changed")

        total_chunks = 0
        batch_size = 800  # Increased batch size for better throughput
        # Parsing is lightweight, can use more cores without competing
        # Use 50% of cores for parsing when using local embeddings
        # or more when using cloud embeddings (less CPU competition)
        num_workers = max(4, cpu_count() // 2)

        print(f"  Using {num_workers} worker processes for parsing")
        print(f"  Batch size: {batch_size} files")

        # Accumulators for batched embedding
        all_function_chunks = []
        all_class_chunks = []
        all_file_chunks = []

        # Process files in batches
        for batch_start in range(0, len(files), batch_size):
            batch_end = min(batch_start + batch_size, len(files))
            batch_files = files[batch_start:batch_end]

            # Progress reporting - show at start of each batch
            print(f"  Parsing: {batch_start}/{len(files)} ({batch_start*100//len(files) if len(files) > 0 else 0}%)")

            # Prepare file info tuples for workers
            file_infos = [(f, self.project_path) for f in batch_files]

            # Process batch in parallel
            with Pool(processes=num_workers) as pool:
                results = pool.map(_process_file_worker, file_infos, chunksize=10)

            # Collect chunks from all files in batch
            files_processed = 0
            progress_report_interval = 100  # Report every 100 files
            for result in results:
                files_processed += 1

                # Show progress every N files within the batch
                if files_processed % progress_report_interval == 0:
                    current_total = batch_start + files_processed
                    progress_pct = (current_total * 100) // len(files) if len(files) > 0 else 0
                    print(f"  Parsing: {current_total}/{len(files)} ({progress_pct}%)")

                # Update progress callback every 10 files
                if files_processed % 10 == 0 and hasattr(self, '_progress_callback'):
                    current_total = batch_start + files_processed
                    self._progress_callback(current_total, len(files))
                if result is not None:
                    func_chunks, cls_chunks, f_chunks = result
                    all_function_chunks.extend(func_chunks)
                    all_class_chunks.extend(cls_chunks)
                    all_file_chunks.extend(f_chunks)

            # Print embedding progress
            total_chunks_to_embed = len(all_function_chunks) + len(all_class_chunks) + len(all_file_chunks)
            print(f"  Embedding: {total_chunks_to_embed} chunks from batch...")

            # Store accumulated chunks (batch embedding happens in _store_code_chunks)
            if all_function_chunks:
                self._store_code_chunks(all_function_chunks, VectorStore.CODE_FUNCTIONS)
                total_chunks += len(all_function_chunks)
                all_function_chunks = []

            if all_class_chunks:
                self._store_code_chunks(all_class_chunks, VectorStore.CODE_CLASSES)
                total_chunks += len(all_class_chunks)
                all_class_chunks = []

            if all_file_chunks:
                self._store_code_chunks(all_file_chunks, VectorStore.CODE_FILES)
                total_chunks += len(all_file_chunks)
                all_file_chunks = []

            # Show completion of batch
            print(f"  [OK] Completed batch: {batch_end}/{len(files)} files processed")

        # Final progress update
        print(f"  Progress: {len(files)}/{len(files)}")
        if hasattr(self, '_progress_callback'):
            self._progress_callback(len(files), len(files))

        print(f"  Indexed {total_chunks} code chunks")

    def _store_code_chunks(self, chunks: List[CodeChunk], collection: str):
        """Generate embeddings and store code chunks"""
        if not chunks:
            return

        # Add repo metadata to chunks if in multi-repo mode
        if self.is_multi_repo and hasattr(self, '_current_repo_name') and self._current_repo_name:
            for chunk in chunks:
                chunk.extra_metadata["repo"] = self._current_repo_name
                # Make file path relative to workspace root
                chunk.file_path = f"{self._current_repo_name}/{chunk.file_path}"

        # Generate embeddings in batch
        contents = [chunk.content for chunk in chunks]
        embeddings = self.code_embedder.embed(contents)

        # Prepare points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_dict = chunk.to_dict()
            chunk_dict["vector"] = embedding
            points.append(chunk_dict)

        # Store in vector database
        self.vector_store.upsert_points(collection, points)

    def _index_documentation(self, files: List[Path], incremental: bool):
        """Index documentation files"""
        points = []

        for file_path in files:
            try:
                content = file_path.read_text(errors='ignore')
                # Normalize path separators for cross-platform compatibility
                rel_path = str(file_path.relative_to(self.project_path)).replace('\\', '/')

                # Add repo prefix if in multi-repo mode
                if self.is_multi_repo and hasattr(self, '_current_repo_name') and self._current_repo_name:
                    rel_path = f"{self._current_repo_name}/{rel_path}"

                # Generate embedding
                embedding = self.text_embedder.embed(content)[0]

                metadata = {
                    "file_path": rel_path,
                    "type": "documentation",
                    "content": content[:500],  # Store snippet
                    "start_line": 1,
                    "end_line": content.count('\n') + 1,
                    "name": file_path.name,
                    "parent": "",
                    "language": "markdown",
                    "content_hash": ""
                }

                # Add repo metadata if in multi-repo mode
                if self.is_multi_repo and hasattr(self, '_current_repo_name') and self._current_repo_name:
                    metadata["repo"] = self._current_repo_name

                points.append({
                    "vector": embedding,
                    "metadata": metadata
                })

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

        if points:
            self.vector_store.upsert_points(VectorStore.DOCUMENTATION, points)
            print(f"  Indexed {len(points)} documentation files")

    def _index_configuration(self, files: List[Path], incremental: bool):
        """Index configuration files"""
        points = []

        for file_path in files:
            try:
                content = file_path.read_text(errors='ignore')
                # Normalize path separators for cross-platform compatibility
                rel_path = str(file_path.relative_to(self.project_path)).replace('\\', '/')

                # Add repo prefix if in multi-repo mode
                if self.is_multi_repo and hasattr(self, '_current_repo_name') and self._current_repo_name:
                    rel_path = f"{self._current_repo_name}/{rel_path}"

                # Generate embedding
                embedding = self.text_embedder.embed(content)[0]

                metadata = {
                    "file_path": rel_path,
                    "type": "configuration",
                    "content": content[:500],
                    "start_line": 1,
                    "end_line": content.count('\n') + 1,
                    "name": file_path.name,
                    "parent": "",
                    "language": "config",
                    "content_hash": ""
                }

                # Add repo metadata if in multi-repo mode
                if self.is_multi_repo and hasattr(self, '_current_repo_name') and self._current_repo_name:
                    metadata["repo"] = self._current_repo_name

                points.append({
                    "vector": embedding,
                    "metadata": metadata
                })

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

        if points:
            self.vector_store.upsert_points(VectorStore.DOCUMENTATION, points)
            print(f"  Indexed {len(points)} configuration files")

    def reindex_file(self, file_path: str):
        """Reindex a single file (for incremental updates)"""
        path = Path(file_path)
        if not path.exists():
            # File deleted, remove from index
            for collection in VectorStore.ALL_COLLECTIONS:
                self.vector_store.delete_by_file(collection, str(path))
            return

        # Determine file type and index accordingly
        if path.suffix in self.DOC_EXTENSIONS or path.suffix in self.CONFIG_EXTENSIONS:
            self._index_documentation([path], incremental=True)
        elif self.ast_chunker.get_language(str(path)):
            self._index_code_files([path], incremental=True)
