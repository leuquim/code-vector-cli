#!/usr/bin/env python3
"""CLI for code vector database operations"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Load .env file early (before any other imports that might use env vars)
try:
    from dotenv import load_dotenv
    for env_path in [
        Path.home() / '.code-vector-db.env',
        Path.cwd() / '.env',
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

from code_vector_db.indexer import CodebaseIndexer
from code_vector_db.query import QueryInterface
from code_vector_db.metadata import ProjectMetadata


def _add_path_arg(subparser):
    """Add --path argument to a subparser"""
    subparser.add_argument(
        "--path",
        default=".",
        help="Project directory (auto-detects single-repo or multi-repo structure)"
    )


def cmd_init(args):
    """Initialize vector database for project and index codebase"""
    indexer = CodebaseIndexer(args.project_path)
    indexer.initialize()
    print(f"\n[OK] Initialized vector database for: {args.project_path}")
    print("\nIndexing codebase...")
    indexer.index_codebase(incremental=False)


def cmd_index(args):
    """Index the codebase"""
    indexer = CodebaseIndexer(args.project_path)
    # Ensure collections exist before upserting. Idempotent when they already
    # exist, but required on a fresh/deleted project — otherwise upserts fail
    # with "Collection doesn't exist".
    indexer.initialize()
    repo_filter = getattr(args, 'repo', None)
    indexer.index_codebase(incremental=args.incremental, repo_filter=repo_filter)


def cmd_reindex_file(args):
    """Reindex a single file"""
    indexer = CodebaseIndexer(args.project_path)
    indexer.reindex_file(args.file)
    print(f"[OK] Reindexed: {args.file}")


def _read_code_snippet(project_path, file_path, start_line, end_line, context_lines=3, max_lines=50):
    """Read code snippet from file with optional context lines"""
    try:
        # In workspace mode, file_path already includes the repo subdirectory
        # e.g., "builder/processors/build.js" or "cms/include/sendProgress.js"
        # Normalize path separators for cross-platform compatibility
        normalized_path = file_path.replace('\\', '/')
        full_path = Path(project_path) / normalized_path

        if not full_path.exists():
            return None

        with open(full_path, 'r', errors='ignore') as f:
            lines = f.readlines()

        # Add context lines
        start = max(0, start_line - 1 - context_lines)
        end = min(len(lines), end_line + context_lines)

        snippet_lines = lines[start:end]

        # Truncate if too long
        if len(snippet_lines) > max_lines:
            kept_start = max_lines - 10
            omitted = len(snippet_lines) - max_lines
            snippet_lines = (
                snippet_lines[:kept_start] +
                [f"   ... ({omitted} lines omitted) ...\n"] +
                snippet_lines[-10:]
            )

        # Add line numbers and indent
        result = []
        for i, line in enumerate(snippet_lines):
            if '... (' in line and 'omitted' in line:
                result.append(line)
            else:
                actual_line = start + i + 1
                result.append(f"   {line.rstrip()}")

        return ''.join(result)
    except Exception:
        return None


def _format_search_result(result, index, args, base_path=None):
    """Format a single search result for display"""
    repo_prefix = f"[{result.metadata.get('repo', '')}] " if result.metadata.get('repo') else ""
    line_range = f"{result.start_line}-{result.end_line}" if result.end_line > result.start_line else str(result.start_line)
    lang_suffix = f" [{result.language}]" if result.language else ""

    # When a reranker ordered the list, show its score (the one driving the
    # order) alongside the raw vector similarity, so the ordering reads correctly.
    if getattr(result, "rerank_score", None) is not None:
        score_str = f"rank {result.rerank_score:.3f} | sim {result.score:.3f}"
    else:
        score_str = f"{result.score:.3f}"

    print(f"{index}. [{score_str}] {repo_prefix}{result.file_path}:{line_range}{lang_suffix}")

    if result.name:
        print(f"   {result.type}: {result.name}")
    elif result.type:
        print(f"   type: {result.type}")

    if hasattr(args, 'show_parent') and args.show_parent and result.parent:
        print(f"   parent: {result.parent}")

    if hasattr(args, 'show_content') and args.show_content:
        if base_path is None:
            base_path = getattr(args, 'workspace_path', args.project_path)
        snippet = _read_code_snippet(
            base_path,
            result.file_path,
            result.start_line,
            result.end_line,
            context_lines=getattr(args, 'context_lines', 3)
        )
        if snippet:
            print(snippet)
        print()


def cmd_search_hybrid(args):
    """Hybrid search combining semantic + BM25 keyword matching"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.search_hybrid(
        args.query,
        limit=args.limit,
        threshold=args.threshold,
        bm25_weight=args.bm25_weight,
        semantic_weight=args.semantic_weight,
        repo=getattr(args, 'repo', None)
    )

    if not results:
        print(f"\nNo results found for: '{args.query}'")
        print(f"Threshold: {args.threshold}")
        print("Try lowering threshold with -t 0.1 or -t 0.0 for more results")
        return

    if getattr(args, 'json', False):
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    weights_info = f"(semantic: {args.semantic_weight:.1f}, keyword: {args.bm25_weight:.1f})"
    print(f"\nFound {len(results)} results {weights_info}:\n")
    for i, result in enumerate(results, 1):
        _format_search_result(result, i, args)


def cmd_search(args):
    """Search code"""
    t_start = time.time()

    query_interface = QueryInterface(args.project_path)
    t_init = time.time()

    results = query_interface.search_code(
        args.query,
        limit=args.limit,
        threshold=args.threshold,
        repo=getattr(args, 'repo', None)
    )
    t_search = time.time()

    if not results:
        print(f"\nNo results found for: '{args.query}'")
        print(f"Threshold: {args.threshold}")
        print("Try lowering threshold with -t 0.1 or -t 0.0 for more results")
        return

    if getattr(args, 'json', False):
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    print(f"\nFound {len(results)} results (threshold: {args.threshold}):\n")
    for i, result in enumerate(results, 1):
        _format_search_result(result, i, args)

    # Print timing breakdown
    init_ms = (t_init - t_start) * 1000
    search_ms = (t_search - t_init) * 1000
    total_ms = (t_search - t_start) * 1000
    print(f"\n[Timing] Init: {init_ms:.0f}ms | Search: {search_ms:.0f}ms | Total: {total_ms:.0f}ms")


def cmd_similar(args):
    """Find similar code - accepts file path or semantic query"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.find_similar(
        args.query,
        limit=args.limit,
        threshold=args.threshold,
        repo=getattr(args, 'repo', None)
    )

    # Detect if it was a file path
    is_file = Path(args.query).exists() and Path(args.query).is_file()
    query_type = "file" if is_file else "query"

    if not results:
        print(f"\nNo similar code found for {query_type}: '{args.query}'")
        print(f"Threshold: {args.threshold}")
        print("Try lowering threshold with -t 0.3 or -t 0.0 for more results")
        return

    if getattr(args, 'json', False):
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    print(f"\nSimilar to {query_type}: '{args.query}'")
    print(f"Found {len(results)} results (threshold: {args.threshold}):\n")
    for i, result in enumerate(results, 1):
        # Check if this is a workspace (has repo metadata)
        repo_prefix = f"[{result.metadata.get('repo', '')}] " if result.metadata.get('repo') else ""

        # Show complete line range
        line_range = f"{result.start_line}-{result.end_line}" if result.end_line > result.start_line else str(result.start_line)

        # Show language if available
        lang_suffix = f" [{result.language}]" if result.language else ""

        if getattr(result, "rerank_score", None) is not None:
            score_str = f"rank {result.rerank_score:.3f} | sim {result.score:.3f}"
        else:
            score_str = f"{result.score:.3f}"

        print(f"{i}. [{score_str}] {repo_prefix}{result.file_path}:{line_range}{lang_suffix}")

        # Show type and name
        if result.name:
            print(f"   {result.type}: {result.name}")
        elif result.type:
            print(f"   type: {result.type}")

        # Show code snippet if requested
        if hasattr(args, 'show_content') and args.show_content:
            # Use workspace_path if available, otherwise project_path
            base_path = getattr(args, 'workspace_path', args.project_path)
            snippet = _read_code_snippet(
                base_path,
                result.file_path,
                result.start_line,
                result.end_line,
                context_lines=getattr(args, 'context_lines', 3)
            )
            if snippet:
                print(f"\n{snippet}")

        print()


def cmd_context(args):
    """Get context for a task"""
    query_interface = QueryInterface(args.project_path)

    context_files = query_interface.get_context_for_task(
        args.task,
        max_files=args.limit,
        threshold=args.threshold,
        repo=getattr(args, 'repo', None)
    )

    if not context_files:
        print(f"\nNo relevant context found for: '{args.task}'")
        print(f"Threshold: {args.threshold}")
        print("Try lowering threshold with -t 0.2 or -t 0.0 for more results")
        return

    if args.json:
        print(json.dumps(context_files, indent=2))
    else:
        print(f"\nRelevant files for: '{args.task}' (threshold: {args.threshold})\n")
        for i, file_info in enumerate(context_files, 1):
            print(f"{i}. [{file_info['score']:.3f}] {file_info['file_path']}")
            print(f"   Reason: {file_info['reason']}")
            if file_info['lines']:
                print(f"   Lines: {file_info['lines']}")
            print()


def cmd_impact(args):
    """Analyze impact - accepts file path or semantic query"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.analyze_impact(
        args.query,
        depth=2,
        threshold=args.threshold,
        repo=getattr(args, 'repo', None)
    )

    if getattr(args, 'json', False):
        output = {
            "query": args.query,
            "query_type": results.get("query_type", "unknown"),
            "direct": [r.to_dict() for r in results["direct"]],
            "indirect": [r.to_dict() for r in results["indirect"]],
        }
        print(json.dumps(output, indent=2))
        return

    query_type = results.get("query_type", "unknown")
    print(f"\nImpact analysis for {query_type}: '{args.query}'")
    print(f"Threshold: {args.threshold}\n")

    if results["direct"]:
        print(f"Direct impacts ({len(results['direct'])}):")
        for result in results["direct"][:10]:
            print(f"  [{result.score:.3f}] {result.file_path}:{result.start_line}")
            if result.name:
                print(f"    {result.type}: {result.name}")
        print()
    else:
        print(f"No direct impacts found (threshold: {args.threshold})")
        print("Try lowering threshold with -t 0.3 or -t 0.0 for more results\n")

    if results["indirect"]:
        print(f"Indirect impacts ({len(results['indirect'])}):")
        for result in results["indirect"][:10]:
            print(f"  [{result.score:.3f}] {result.file_path}:{result.start_line}")
            if result.name:
                print(f"    {result.type}: {result.name}")
    elif results["direct"]:
        print(f"No indirect impacts found (threshold: {args.threshold})")


def cmd_search_docs(args):
    """Search documentation"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.search_documentation(
        args.query,
        limit=args.limit,
        threshold=args.threshold
    )

    if not results:
        print("No documentation found")
        return

    if getattr(args, 'json', False):
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    print(f"\nFound {len(results)} documentation results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.score:.3f}] {result.file_path}")
        print()


def cmd_search_conversations(args):
    """Search conversation history"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.search_conversations(
        args.query,
        limit=args.limit,
        threshold=args.threshold
    )

    if not results:
        print("No conversations found")
        return

    print(f"\nFound {len(results)} conversation results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.score:.3f}] Session: {result.metadata.get('session_id', 'unknown')[:12]}")
        print(f"   {result.metadata.get('timestamp', '')}")
        if result.content:
            print(f"   {result.content[:200]}...")
        print()


def cmd_stats(args):
    """Show statistics"""
    metadata = ProjectMetadata()
    project_info = metadata.find_project_by_path(args.project_path)

    if project_info:
        # Use the registered project's path for correct project_id
        query_interface = QueryInterface(project_info['path'])
    else:
        query_interface = QueryInterface(args.project_path)

    stats = query_interface.get_stats()

    if getattr(args, 'json', False):
        print(json.dumps(stats, indent=2))
        return

    print(f"\nVector Database Statistics")
    print(f"Project ID: {stats['project_id']}\n")
    print("Collections:")

    total_points = 0
    for collection, info in stats["collections"].items():
        count = info["points_count"]
        total_points += count
        print(f"  {collection:20s}: {count:6d} points")

    print(f"\nTotal: {total_points} points")


def cmd_install_hook(args):
    """Install git post-commit hook"""
    import shutil

    git_dir = Path(args.project_path) / ".git"
    if not git_dir.exists():
        print("Error: Not a git repository")
        return

    hook_source = Path.home() / ".local/share/code-vector-db/post-commit-hook"
    hook_dest = git_dir / "hooks" / "post-commit"

    # Backup existing hook if present
    if hook_dest.exists():
        backup = hook_dest.with_suffix(".backup")
        shutil.copy(hook_dest, backup)
        print(f"[OK] Backed up existing hook to {backup}")

    # Install hook
    shutil.copy(hook_source, hook_dest)
    hook_dest.chmod(0o755)

    print(f"[OK] Installed post-commit hook at {hook_dest}")
    print("  Vector database will auto-update on commits")


def cmd_migrate_conversations(args):
    """Migrate conversation transcripts to vector database"""
    from code_vector_db.embeddings import get_text_embedder
    from code_vector_db.vector_store import VectorStore
    import json

    transcript_dir = Path(args.project_path) / ".claude-transcripts"
    if not transcript_dir.exists():
        print(f"No transcript directory found at {transcript_dir}")
        return

    # Find all transcript files
    transcript_files = list(transcript_dir.glob("*.jsonl"))
    if not transcript_files:
        print("No transcript files found")
        return

    print(f"Found {len(transcript_files)} transcript files")

    vector_store = VectorStore(args.project_path)
    text_embedder = get_text_embedder()

    total_messages = 0
    points = []

    for transcript_file in transcript_files:
        try:
            with open(transcript_file) as f:
                for line in f:
                    if not line.strip():
                        continue

                    message = json.loads(line)
                    role = message.get("role", "")
                    content = message.get("content", "")

                    if not content or role not in ["user", "assistant"]:
                        continue

                    # Extract text content
                    text_content = ""
                    if isinstance(content, str):
                        text_content = content
                    elif isinstance(content, list):
                        text_content = " ".join(
                            block.get("text", "") for block in content
                            if isinstance(block, dict) and "text" in block
                        )

                    if not text_content or len(text_content) < 10:
                        continue

                    # Generate embedding
                    embedding = text_embedder.embed(text_content)[0]

                    points.append({
                        "vector": embedding,
                        "metadata": {
                            "file_path": str(transcript_file.name),
                            "type": "conversation",
                            "role": role,
                            "content": text_content[:500],  # Store snippet
                            "session_id": transcript_file.stem,
                            "timestamp": message.get("timestamp", ""),
                            "model": message.get("model", ""),
                            "start_line": 0,
                            "end_line": 0,
                            "name": "",
                            "parent": "",
                            "language": "",
                            "content_hash": ""
                        }
                    })

                    total_messages += 1

                    # Batch insert every 100 messages
                    if len(points) >= 100:
                        vector_store.upsert_points(VectorStore.CONVERSATIONS, points)
                        points = []
                        print(f"  Migrated {total_messages} messages...")

        except Exception as e:
            print(f"  Error processing {transcript_file}: {e}")

    # Insert remaining points
    if points:
        vector_store.upsert_points(VectorStore.CONVERSATIONS, points)

    print(f"\n[OK] Migrated {total_messages} conversation messages")
    print(f"  Use search-conversations to search conversations")


def cmd_list_projects(args):
    """List all indexed projects"""
    from qdrant_client import QdrantClient
    from collections import defaultdict

    metadata = ProjectMetadata()

    # Get collections from Qdrant
    try:
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", 6333))
        client = QdrantClient(host=host, port=port)
        collections_response = client.get_collections()
        collections = [c.name for c in collections_response.collections]
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running")
        return

    # Group collections by project ID
    project_collections = defaultdict(list)
    for collection_name in collections:
        if "_" in collection_name:
            project_id = collection_name.rsplit("_", 2)[0]
            project_collections[project_id].append(collection_name)

    # Get metadata for all projects
    registered_projects = {p["project_id"]: p for p in metadata.list_all_projects()}

    if not project_collections:
        print("\nNo indexed projects found.")
        print("\nRun: code-vector-cli index /path/to/project")
        return

    print(f"\n{'='*70}")
    print("INDEXED PROJECTS")
    print(f"{'='*70}\n")

    for project_id in sorted(project_collections.keys()):
        collections = project_collections[project_id]
        metadata_info = registered_projects.get(project_id)

        print(f"Project ID: {project_id}")

        if metadata_info:
            print(f"  Path: {metadata_info['path']}")
            path_exists = Path(metadata_info['path']).exists()
            if not path_exists:
                print(f"  Status: [WARN]  Path no longer exists")
            else:
                print(f"  Status: [OK] Active")

            print(f"  Indexed: {metadata_info.get('indexed_at', 'unknown')}")
            print(f"  Updated: {metadata_info.get('last_updated', 'unknown')}")
            print(f"  Files: {metadata_info.get('file_count', 'unknown')}")

            if metadata_info.get('collection_stats'):
                total_points = sum(metadata_info['collection_stats'].values())
                print(f"  Vectors: {total_points:,}")
        else:
            print(f"  Path: [WARN]  Unknown (not in metadata registry)")
            print(f"  Status: Orphaned - no metadata")

        print(f"  Collections: {len(collections)}")

        # Get point counts from Qdrant
        total_vectors = 0
        for collection_name in collections:
            try:
                info = client.get_collection(collection_name)
                count = info.points_count
                total_vectors += count
                if args.verbose:
                    coll_type = collection_name.split("_", 1)[1] if "_" in collection_name else collection_name
                    print(f"    - {coll_type}: {count:,} points")
            except:
                pass

        if not args.verbose and total_vectors > 0:
            print(f"  Total vectors: {total_vectors:,}")

        print()

    print(f"{'='*70}\n")
    print(f"Total projects: {len(project_collections)}")

    # Check for orphaned metadata
    orphaned_meta = []
    for project_id, info in registered_projects.items():
        if project_id not in project_collections:
            orphaned_meta.append((project_id, info))

    if orphaned_meta:
        print(f"\n[WARN]  Found {len(orphaned_meta)} projects in metadata but not in Qdrant:")
        for project_id, info in orphaned_meta:
            print(f"  - {project_id}: {info['path']}")
        print("\nRun: code-vector-cli cleanup-metadata")


def cmd_cleanup_metadata(args):
    """Clean up metadata for projects that no longer exist"""
    metadata = ProjectMetadata()
    removed = metadata.cleanup_missing_projects()

    if removed:
        print(f"\n[OK] Removed {len(removed)} missing projects:")
        for item in removed:
            print(f"  - {item}")
    else:
        print("\n[OK] No missing projects found. All metadata is valid.")


def cmd_delete(args):
    """Delete all collections for the project"""
    from code_vector_db.vector_store import VectorStore
    from qdrant_client import QdrantClient
    import os

    project_id = getattr(args, 'project_id', None)

    if project_id:
        # Delete by project ID directly (for orphaned projects)
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", 6333))
        client = QdrantClient(host=host, port=port, timeout=300)

        if not args.force:
            print(f"\n[WARN]  WARNING: This will delete all indexed data for:")
            print(f"   Project ID: {project_id}")
            print(f"\nCollections to be deleted:")
            for collection in VectorStore.ALL_COLLECTIONS:
                print(f"   - {project_id}_{collection}")

            response = input("\nAre you sure you want to continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("\n[FAIL] Deletion cancelled")
                return

        print(f"\nDeleting collections for project ID: {project_id}")
        import requests
        for collection in VectorStore.ALL_COLLECTIONS:
            collection_name = f"{project_id}_{collection}"
            try:
                url = f"http://{host}:{port}/collections/{collection_name}"
                resp = requests.delete(url, timeout=30)
                if resp.status_code == 200:
                    print(f"[OK] Deleted collection: {collection_name}")
                elif resp.status_code == 404:
                    print(f"  Collection {collection_name} does not exist")
                else:
                    print(f"[WARN] Status {resp.status_code} for {collection_name}")
            except Exception as e:
                print(f"[WARN] Error deleting {collection_name}: {e}")

        print(f"\n[OK] Successfully deleted all data for project {project_id}")
    else:
        # Delete by path (original behavior)
        vector_store = VectorStore(args.project_path)

        if not args.force:
            print(f"\n[WARN]  WARNING: This will delete all indexed data for:")
            print(f"   Path: {args.project_path}")
            print(f"   Project ID: {vector_store.project_id}")
            print(f"\nCollections to be deleted:")
            for collection in VectorStore.ALL_COLLECTIONS:
                print(f"   - {collection}")

            response = input("\nAre you sure you want to continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("\n[FAIL] Deletion cancelled")
                return

        print(f"\nDeleting collections for: {args.project_path}")
        vector_store.delete_collections()

        # Clean up metadata
        metadata = ProjectMetadata()
        metadata.unregister_project(args.project_path)

        print(f"\n[OK] Successfully deleted all data for project")


def cmd_index_git(args):
    """Index git commit history"""
    indexer = CodebaseIndexer(args.project_path)
    max_commits = getattr(args, 'max_commits', 500)
    indexer.index_git_history(max_commits=max_commits)


def cmd_search_git(args):
    """Search git commit history"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.search_git_history(
        args.query,
        limit=args.limit,
        threshold=args.threshold,
        repo=getattr(args, 'repo', None)
    )

    if not results:
        print(f"\nNo git history found for: '{args.query}'")
        return

    if getattr(args, 'json', False):
        output = []
        for r in results:
            output.append({
                "score": round(r.score, 3),
                "commit": r.metadata.get("commit_hash", "")[:8],
                "author": r.metadata.get("author", ""),
                "date": r.metadata.get("date", ""),
                "message": r.metadata.get("message", ""),
                "repo": r.metadata.get("repo", ""),
                "files_changed": r.metadata.get("files_changed", [])
            })
        print(json.dumps(output, indent=2))
        return

    print(f"\nFound {len(results)} commits matching: '{args.query}'\n")
    for i, result in enumerate(results, 1):
        commit_hash = result.metadata.get("commit_hash", "")[:8]
        author = result.metadata.get("author", "")
        date = result.metadata.get("date", "")[:10]  # Just the date part
        message = result.metadata.get("message", "")
        repo = result.metadata.get("repo", "")
        files = result.metadata.get("files_changed", [])

        repo_prefix = f"[{repo}] " if repo else ""
        print(f"{i}. [{result.score:.3f}] {repo_prefix}{commit_hash} {date} ({author})")
        print(f"   {message}")
        if files:
            shown_files = files[:5]
            print(f"   Files: {', '.join(shown_files)}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        print()


def cmd_mcp(args):
    """Handle MCP commands"""
    mcp_command = getattr(args, 'mcp_command', None)
    if mcp_command == "serve":
        try:
            from code_vector_db.mcp_server import set_project_path, run_server
        except ImportError:
            print("Error: MCP package not installed. Install with: pip install mcp")
            sys.exit(1)
        path = os.path.abspath(getattr(args, 'path', '.'))
        set_project_path(path)
        run_server()
    else:
        print("Usage: code-vector-cli mcp serve [--path PATH]")
        print("\nStart an MCP server for Claude Code integration")


def main():
    parser = argparse.ArgumentParser(
        description="Code Vector Database CLI - Semantic code search powered by vector embeddings"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize vector database and index")
    _add_path_arg(init_parser)

    # index command
    index_parser = subparsers.add_parser("index", help="Index codebase")
    _add_path_arg(index_parser)
    index_parser.add_argument("--incremental", action="store_true", help="Incremental indexing")
    index_parser.add_argument("--repo", type=str, help="Index only specific repo in multi-repo workspace")

    # reindex-file command
    reindex_parser = subparsers.add_parser("reindex-file", help="Reindex single file")
    _add_path_arg(reindex_parser)
    reindex_parser.add_argument("file", help="File to reindex")

    # search command
    search_parser = subparsers.add_parser("search", help="Search code semantically")
    _add_path_arg(search_parser)
    search_parser.add_argument("query", help="Search query (natural language)")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    search_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold (0.0-1.0)")
    search_parser.add_argument("--show-parent", action="store_true", help="Show parent class/module")
    search_parser.add_argument("--show-content", action="store_true", help="Show code snippets")
    search_parser.add_argument("-C", "--context-lines", type=int, default=3, help="Context lines (default: 3)")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")
    search_parser.add_argument("--repo", type=str, help="Filter by repository name (e.g., base, frontend, mobile_app)")

    # search-hybrid command
    hybrid_parser = subparsers.add_parser("search-hybrid", help="Hybrid search (semantic + keyword BM25)")
    _add_path_arg(hybrid_parser)
    hybrid_parser.add_argument("query", help="Search query")
    hybrid_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    hybrid_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold")
    hybrid_parser.add_argument("--show-parent", action="store_true", help="Show parent class/module")
    hybrid_parser.add_argument("--show-content", action="store_true", help="Show code snippets")
    hybrid_parser.add_argument("-C", "--context-lines", type=int, default=3, help="Context lines around code snippets (default: 3)")
    hybrid_parser.add_argument("--bm25-weight", type=float, default=0.3, help="BM25 keyword weight (default: 0.3)")
    hybrid_parser.add_argument("--semantic-weight", type=float, default=0.7, help="Semantic similarity weight (default: 0.7)")
    hybrid_parser.add_argument("--json", action="store_true", help="Output as JSON")
    hybrid_parser.add_argument("--repo", type=str, help="Filter by repository name (e.g., base, frontend, mobile_app)")

    # similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar code")
    _add_path_arg(similar_parser)
    similar_parser.add_argument("query", help="File path OR semantic query")
    similar_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    similar_parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Score threshold (0.0-1.0)")
    similar_parser.add_argument("--show-content", action="store_true", help="Show code snippets")
    similar_parser.add_argument("-C", "--context-lines", type=int, default=3, help="Context lines (default: 3)")
    similar_parser.add_argument("--json", action="store_true", help="Output as JSON")
    similar_parser.add_argument("--repo", type=str, help="Filter by repository name (e.g., base, frontend, mobile_app)")

    # context command
    context_parser = subparsers.add_parser("context", help="Get context for task")
    _add_path_arg(context_parser)
    context_parser.add_argument("task", help="Task description")
    context_parser.add_argument("-n", "--limit", type=int, default=10, help="Max files")
    context_parser.add_argument("-t", "--threshold", type=float, default=0.4, help="Score threshold (0.0-1.0)")
    context_parser.add_argument("--json", action="store_true", help="Output as JSON")
    context_parser.add_argument("--repo", type=str, help="Filter by repository name (e.g., base, frontend, mobile_app)")

    # impact command
    impact_parser = subparsers.add_parser("impact", help="Analyze change impact")
    _add_path_arg(impact_parser)
    impact_parser.add_argument("query", help="File path OR semantic query")
    impact_parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Score threshold (0.0-1.0)")
    impact_parser.add_argument("--json", action="store_true", help="Output as JSON")
    impact_parser.add_argument("--repo", type=str, help="Filter by repository name (e.g., base, frontend, mobile_app)")

    # search-docs command
    docs_parser = subparsers.add_parser("search-docs", help="Search documentation")
    _add_path_arg(docs_parser)
    docs_parser.add_argument("query", help="Search query")
    docs_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    docs_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold (0.0-1.0)")
    docs_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # search-conversations command
    conv_parser = subparsers.add_parser("search-conversations", help="Search conversation history")
    _add_path_arg(conv_parser)
    conv_parser.add_argument("query", help="Search query")
    conv_parser.add_argument("-n", "--limit", type=int, default=5, help="Number of results")
    conv_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold (0.0-1.0)")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    _add_path_arg(stats_parser)
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # install-hook command
    install_hook_parser = subparsers.add_parser("install-hook", help="Install git post-commit hook")
    _add_path_arg(install_hook_parser)

    # migrate-conversations command
    migrate_parser = subparsers.add_parser("migrate-conversations", help="Migrate conversation transcripts")
    _add_path_arg(migrate_parser)

    # list-projects command
    list_parser = subparsers.add_parser("list-projects", help="List all indexed projects")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed stats")

    # cleanup-metadata command
    subparsers.add_parser("cleanup-metadata", help="Clean up metadata for missing projects")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete indexed data")
    _add_path_arg(delete_parser)
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    delete_parser.add_argument("--project-id", type=str, help="Delete by project ID (for orphaned projects)")

    # index-git command
    index_git_parser = subparsers.add_parser("index-git", help="Index git commit history")
    _add_path_arg(index_git_parser)
    index_git_parser.add_argument("--max-commits", type=int, default=500, help="Max commits per repo (default: 500)")

    # search-git command
    search_git_parser = subparsers.add_parser("search-git", help="Search git commit history")
    _add_path_arg(search_git_parser)
    search_git_parser.add_argument("query", help="Search query")
    search_git_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    search_git_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold")
    search_git_parser.add_argument("--repo", type=str, help="Filter by repository name")
    search_git_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # mcp command with serve subcommand
    mcp_parser = subparsers.add_parser("mcp", help="MCP server commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP subcommand")
    mcp_serve_parser = mcp_subparsers.add_parser("serve", help="Start MCP server (stdio transport)")
    mcp_serve_parser.add_argument(
        "--path",
        default=".",
        help="Project directory to serve"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Resolve path (use getattr since not all subparsers have --path)
    args.path = os.path.abspath(getattr(args, 'path', '.'))

    # Set legacy aliases for backward compatibility
    args.project_path = args.path
    args.workspace_path = args.path

    # Dispatch to command handler
    command_map = {
        "init": cmd_init,
        "index": cmd_index,
        "reindex-file": cmd_reindex_file,
        "search": cmd_search,
        "search-hybrid": cmd_search_hybrid,
        "similar": cmd_similar,
        "context": cmd_context,
        "impact": cmd_impact,
        "search-docs": cmd_search_docs,
        "search-conversations": cmd_search_conversations,
        "stats": cmd_stats,
        "install-hook": cmd_install_hook,
        "migrate-conversations": cmd_migrate_conversations,
        "list-projects": cmd_list_projects,
        "cleanup-metadata": cmd_cleanup_metadata,
        "delete": cmd_delete,
        "index-git": cmd_index_git,
        "search-git": cmd_search_git,
        "mcp": cmd_mcp,
    }

    handler = command_map.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\n\nInterrupted")
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
