"""Workspace indexer for multi-repo projects"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import subprocess

from .indexer import CodebaseIndexer
from .vector_store import VectorStore
from .ast_chunker import CodeChunk


class WorkspaceIndexer:
    """Indexes multiple git repositories as a unified workspace"""

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path).resolve()

        # Discover all git repos in workspace
        self.repos = self._discover_repos()

        if not self.repos:
            raise ValueError(f"No git repositories found in {workspace_path}")

        # Create shared vector store using workspace path
        self.vector_store = VectorStore(str(self.workspace_path))

        # Progress tracking file and structure
        self.progress_file = self.workspace_path / ".indexing-progress.json"
        self._initialize_progress_tracking()

        print(f"Workspace: {self.workspace_path}")
        print(f"Found {len(self.repos)} repositories:")
        for repo in self.repos:
            print(f"  - {repo.name}")

    def _discover_repos(self) -> List[Path]:
        """
        Find all git repositories in workspace

        Returns only direct subdirectories with .git
        Does NOT recurse (avoids submodules)
        """
        repos = []

        if not self.workspace_path.is_dir():
            return repos

        for item in self.workspace_path.iterdir():
            if item.is_dir() and (item / ".git").exists():
                repos.append(item)

        return sorted(repos, key=lambda p: p.name)

    def _get_repo_stats(self, repo_path: Path) -> Dict:
        """Get statistics about a repository using git ls-files"""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return {
                    "file_count": 0,
                    "primary_language": "unknown",
                    "size": 0
                }

            tracked_files = [line for line in result.stdout.strip().split('\n') if line]
            file_count = len(tracked_files)

            extensions = {}
            total_size = 0

            for file_rel_path in tracked_files:
                file_path = repo_path / file_rel_path
                if file_path.exists() and file_path.is_file():
                    if file_path.suffix:
                        ext = file_path.suffix
                        extensions[ext] = extensions.get(ext, 0) + 1

                    try:
                        total_size += file_path.stat().st_size
                    except:
                        pass

            primary_lang = max(extensions.items(), key=lambda x: x[1])[0] if extensions else "unknown"

            return {
                "file_count": file_count,
                "primary_language": primary_lang,
                "size": total_size // 1024
            }
        except Exception as e:
            return {
                "file_count": 0,
                "primary_language": "unknown",
                "size": 0
            }

    def _initialize_progress_tracking(self):
        """Initialize progress tracking structure with all repositories"""
        try:
            # Get stats for all repos to show total files
            repo_stats = {}
            for repo_path in self.repos:
                stats = self._get_repo_stats(repo_path)
                repo_stats[repo_path.name] = {
                    "status": "pending",
                    "files_processed": 0,
                    "total_files": stats.get("file_count", 0),
                    "percent": 0.0
                }

            self.progress_data = {
                "workspace": {
                    "total_repos": len(self.repos),
                    "completed_repos": 0,
                    "current_repo": None,
                    "overall_percent": 0.0
                },
                "repositories": repo_stats
            }

            # Write initial state
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            pass

    def _update_progress(self, repo_name: str, current: int, total: int):
        """Update progress file with current indexing status"""
        try:
            # Update repository progress
            if repo_name in self.progress_data["repositories"]:
                percent = round((current / total * 100), 1) if total > 0 else 0
                self.progress_data["repositories"][repo_name].update({
                    "files_processed": current,
                    "total_files": total,
                    "percent": percent
                })

            # Update workspace-level progress
            self.progress_data["workspace"]["current_repo"] = repo_name

            # Calculate overall completion
            total_files_all_repos = sum(
                r["total_files"] for r in self.progress_data["repositories"].values()
            )
            processed_files_all_repos = sum(
                r["files_processed"] for r in self.progress_data["repositories"].values()
            )

            if total_files_all_repos > 0:
                overall_percent = round((processed_files_all_repos / total_files_all_repos * 100), 1)
                self.progress_data["workspace"]["overall_percent"] = overall_percent

            # Write updated state
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            pass

    def _mark_repo_status(self, repo_name: str, status: str):
        """Mark a repository as pending/in_progress/completed"""
        try:
            if repo_name in self.progress_data["repositories"]:
                self.progress_data["repositories"][repo_name]["status"] = status

                # Update completed count
                completed = sum(
                    1 for r in self.progress_data["repositories"].values()
                    if r["status"] == "completed"
                )
                self.progress_data["workspace"]["completed_repos"] = completed

                # Write updated state
                with open(self.progress_file, 'w') as f:
                    json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            pass

    def initialize(self):
        """Initialize shared vector database collections"""
        print("\nInitializing workspace vector database...")
        self.vector_store.initialize_collections()
        print("[OK] Collections created")

    def index_workspace(self, incremental: bool = False):
        """
        Index all repositories in workspace

        Each repo is indexed with repo metadata attached to every vector
        """
        print(f"\nIndexing workspace with {len(self.repos)} repositories...")
        print("=" * 60)

        total_files = 0
        total_chunks = 0

        for i, repo_path in enumerate(self.repos, 1):
            print(f"\n[{i}/{len(self.repos)}] Repository: {repo_path.name}")
            print("-" * 60)

            # Get repo stats
            stats = self._get_repo_stats(repo_path)
            print(f"  Files: ~{stats['file_count']}")
            print(f"  Language: {stats['primary_language']}")
            print(f"  Size: ~{stats['size']} KB")

            # Index this repository
            try:
                # Mark repo as in progress
                self._mark_repo_status(repo_path.name, "in_progress")

                indexer = CodebaseIndexer(str(repo_path))

                # Add progress callback
                indexer._progress_callback = lambda current, total: self._update_progress(
                    repo_path.name, current, total
                )

                # Monkeypatch to add repo metadata
                original_store_chunks = indexer._store_code_chunks

                def store_chunks_with_repo_metadata(chunks: List[CodeChunk], collection: str):
                    """Add repo metadata to all chunks before storing"""
                    for chunk in chunks:
                        chunk.extra_metadata["repo"] = repo_path.name
                        chunk.extra_metadata["repo_path"] = str(repo_path)

                    # Use shared vector store
                    original_store = indexer.vector_store
                    indexer.vector_store = self.vector_store
                    original_store_chunks(chunks, collection)
                    indexer.vector_store = original_store

                indexer._store_code_chunks = store_chunks_with_repo_metadata

                # Also patch documentation/config indexing
                original_index_docs = indexer._index_documentation

                def index_docs_with_repo_metadata(files, inc):
                    """Add repo metadata to doc points"""
                    original_upsert = indexer.vector_store.upsert_points

                    def upsert_with_metadata(collection, points):
                        for point in points:
                            point["metadata"]["repo"] = repo_path.name
                            point["metadata"]["repo_path"] = str(repo_path)

                        # Use shared vector store
                        self.vector_store.upsert_points(collection, points)

                    indexer.vector_store.upsert_points = upsert_with_metadata
                    original_index_docs(files, inc)
                    indexer.vector_store.upsert_points = original_upsert

                indexer._index_documentation = index_docs_with_repo_metadata
                indexer._index_configuration = index_docs_with_repo_metadata

                # Index the repository
                indexer.index_codebase(incremental=incremental)

                # Mark repo as completed
                self._mark_repo_status(repo_path.name, "completed")

                print(f"  [OK] Indexed {repo_path.name}")

            except Exception as e:
                print(f"  [FAIL] Error indexing {repo_path.name}: {e}")
                # Mark as failed/pending on error
                self._mark_repo_status(repo_path.name, "error")
                continue

        print("\n" + "=" * 60)
        print("[OK] Workspace indexing complete!")
        print(f"\nIndexed {len(self.repos)} repositories into shared collections")
        print(f"Project ID: {self.vector_store.project_id}")

        # Clean up progress file
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
        except:
            pass

    def reindex_repo(self, repo_name: str):
        """Reindex a specific repository in the workspace"""
        repo_path = self.workspace_path / repo_name

        if not repo_path.exists() or not (repo_path / ".git").exists():
            raise ValueError(f"Repository '{repo_name}' not found in workspace")

        print(f"Reindexing repository: {repo_name}")

        # Delete old vectors for this repo
        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        for collection in VectorStore.ALL_COLLECTIONS:
            collection_name = self.vector_store._collection_name(collection)
            try:
                # Delete points where repo == repo_name
                self.vector_store.client.delete(
                    collection_name=collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="repo",
                                    match=MatchValue(value=repo_name)
                                )
                            ]
                        )
                    )
                )
            except Exception as e:
                print(f"  Warning: Could not delete old vectors from {collection}: {e}")

        # Reindex (full index for this repo)
        indexer = WorkspaceIndexer(str(self.workspace_path))

        # Temporarily set repos to just this one
        indexer.repos = [repo_path]
        indexer.index_workspace(incremental=False)

    def get_stats(self) -> Dict:
        """Get workspace statistics"""
        stats = {
            "workspace_path": str(self.workspace_path),
            "repo_count": len(self.repos),
            "repos": []
        }

        for repo in self.repos:
            repo_stats = self._get_repo_stats(repo)
            repo_stats["name"] = repo.name
            repo_stats["path"] = str(repo)
            stats["repos"].append(repo_stats)

        return stats
