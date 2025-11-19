"""Project metadata tracking"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib


class ProjectMetadata:
    """Manages project metadata registry"""

    REGISTRY_PATH = Path.home() / ".local/share/code-vector-db/indexes/project-registry.json"

    def __init__(self):
        self.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load the project registry from disk"""
        if self.REGISTRY_PATH.exists():
            try:
                with open(self.REGISTRY_PATH) as f:
                    self.registry = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load registry (corrupted?): {e}")
                print(f"  Creating new registry at {self.REGISTRY_PATH}")
                self.registry = {}
        else:
            self.registry = {}
        return self.registry

    def _save_registry(self):
        """Save the project registry to disk"""
        with open(self.REGISTRY_PATH, 'w') as f:
            json.dump(self.registry, f, indent=2)

    @staticmethod
    def get_project_id(project_path: str) -> str:
        """Calculate project ID from path"""
        abs_path = os.path.abspath(project_path)
        return hashlib.md5(abs_path.encode()).hexdigest()[:12]

    def register_project(
        self,
        project_path: str,
        file_count: int = 0,
        collection_stats: Optional[Dict[str, int]] = None
    ):
        """Register or update a project in the registry"""
        project_id = self.get_project_id(project_path)
        abs_path = os.path.abspath(project_path)

        now = datetime.utcnow().isoformat() + "Z"

        if project_id in self.registry:
            # Update existing entry
            self.registry[project_id].update({
                "path": abs_path,
                "file_count": file_count,
                "last_updated": now,
                "collection_stats": collection_stats or self.registry[project_id].get("collection_stats", {})
            })
        else:
            # Create new entry
            self.registry[project_id] = {
                "path": abs_path,
                "indexed_at": now,
                "last_updated": now,
                "file_count": file_count,
                "collection_stats": collection_stats or {}
            }

        self._save_registry()

    def get_project_info(self, project_path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific project"""
        project_id = self.get_project_id(project_path)
        return self.registry.get(project_id)

    def list_all_projects(self) -> List[Dict[str, Any]]:
        """List all registered projects"""
        projects = []
        for project_id, info in self.registry.items():
            projects.append({
                "project_id": project_id,
                **info
            })
        return projects

    def unregister_project(self, project_path: str):
        """Remove a project from the registry"""
        project_id = self.get_project_id(project_path)
        if project_id in self.registry:
            del self.registry[project_id]
            self._save_registry()

    def get_last_indexed_time(self, project_path: str) -> Optional[float]:
        """Get timestamp of last indexing for a project

        Returns:
            Unix timestamp (seconds since epoch) or None if never indexed
        """
        info = self.get_project_info(project_path)
        if not info:
            return None

        # Get last_updated timestamp (ISO format)
        timestamp_str = info.get("last_updated")
        if not timestamp_str:
            return None

        try:
            # Parse ISO format: "2025-11-18T22:30:00.123456Z"
            # The 'Z' indicates UTC timezone
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'

            dt = datetime.fromisoformat(timestamp_str)

            # Convert to Unix timestamp
            return dt.timestamp()
        except Exception as e:
            return None

    def get_last_indexed_commit(self, project_path: str, repo_name: str = "") -> Optional[str]:
        """Get the last indexed commit hash for a repository

        Args:
            project_path: Project or workspace path
            repo_name: Repository name (for multi-repo workspaces)

        Returns:
            Commit hash or None if never indexed
        """
        info = self.get_project_info(project_path)
        if not info:
            return None

        git_commits = info.get("git_commits", {})
        return git_commits.get(repo_name or "main")

    def set_last_indexed_commit(self, project_path: str, repo_name: str, commit_hash: str):
        """Store the last indexed commit hash for a repository

        Args:
            project_path: Project or workspace path
            repo_name: Repository name
            commit_hash: Latest commit hash indexed
        """
        project_id = self.get_project_id(project_path)
        if project_id not in self.registry:
            return

        if "git_commits" not in self.registry[project_id]:
            self.registry[project_id]["git_commits"] = {}

        self.registry[project_id]["git_commits"][repo_name or "main"] = commit_hash
        self._save_registry()

    def cleanup_missing_projects(self) -> List[str]:
        """Remove projects whose paths no longer exist"""
        removed = []
        for project_id, info in list(self.registry.items()):
            path = info.get("path")
            if path and not Path(path).exists():
                del self.registry[project_id]
                removed.append(f"{project_id} ({path})")

        if removed:
            self._save_registry()

        return removed
