#!/usr/bin/env python3
"""CLI for code vector database operations"""

import sys
import os
import argparse
import json
from pathlib import Path

from code_vector_db.indexer import CodebaseIndexer
from code_vector_db.query import QueryInterface
from code_vector_db.metadata import ProjectMetadata


def cmd_init(args):
    """Initialize vector database for project and index codebase"""
    indexer = CodebaseIndexer(args.project_path)
    indexer.initialize()
    print(f"\n✓ Initialized vector database for: {args.project_path}")
    print("\nIndexing codebase...")
    indexer.index_codebase(incremental=False)


def cmd_index(args):
    """Index the codebase"""
    indexer = CodebaseIndexer(args.project_path)
    repo_filter = getattr(args, 'repo', None)
    indexer.index_codebase(incremental=args.incremental, repo_filter=repo_filter)


def cmd_reindex_file(args):
    """Reindex a single file"""
    indexer = CodebaseIndexer(args.project_path)
    indexer.reindex_file(args.file)
    print(f"✓ Reindexed: {args.file}")


def _read_code_snippet(project_path, file_path, start_line, end_line, context_lines=3, max_lines=50):
    """Read code snippet from file with optional context lines"""
    try:
        # In workspace mode, file_path already includes the repo subdirectory
        # e.g., "builder/processors/build.js" or "cms/include/sendProgress.js"
        full_path = Path(project_path) / file_path

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


def cmd_search(args):
    """Search code"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.search_code(
        args.query,
        limit=args.limit,
        threshold=args.threshold
    )

    if not results:
        print(f"\nNo results found for: '{args.query}'")
        print(f"Threshold: {args.threshold}")
        print("Try lowering threshold with -t 0.1 or -t 0.0 for more results")
        return

    print(f"\nFound {len(results)} results (threshold: {args.threshold}):\n")
    for i, result in enumerate(results, 1):
        # Check if this is a workspace (has repo metadata)
        repo_prefix = f"[{result.metadata.get('repo', '')}] " if result.metadata.get('repo') else ""

        # Show complete line range
        line_range = f"{result.start_line}-{result.end_line}" if result.end_line > result.start_line else str(result.start_line)

        # Show language if available
        lang_suffix = f" [{result.language}]" if result.language else ""

        print(f"{i}. [{result.score:.3f}] {repo_prefix}{result.file_path}:{line_range}{lang_suffix}")

        # Show type and name
        if result.name:
            print(f"   {result.type}: {result.name}")
        elif result.type:
            print(f"   type: {result.type}")

        # Show parent if requested
        if args.show_parent and result.parent:
            print(f"   parent: {result.parent}")

        # Show code snippet if requested
        if args.show_content:
            # Use workspace_path if available, otherwise project_path
            base_path = getattr(args, 'workspace_path', args.project_path)
            snippet = _read_code_snippet(
                base_path,
                result.file_path,
                result.start_line,
                result.end_line,
                context_lines=args.context_lines
            )
            if snippet:
                print(f"\n{snippet}")

        print()


def cmd_similar(args):
    """Find similar code - accepts file path or semantic query"""
    query_interface = QueryInterface(args.project_path)

    results = query_interface.find_similar(
        args.query,
        limit=args.limit,
        threshold=args.threshold
    )

    # Detect if it was a file path
    from pathlib import Path
    is_file = Path(args.query).exists() and Path(args.query).is_file()
    query_type = "file" if is_file else "query"

    if not results:
        print(f"\nNo similar code found for {query_type}: '{args.query}'")
        print(f"Threshold: {args.threshold}")
        print("Try lowering threshold with -t 0.3 or -t 0.0 for more results")
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

        print(f"{i}. [{result.score:.3f}] {repo_prefix}{result.file_path}:{line_range}{lang_suffix}")

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
        threshold=args.threshold
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
        threshold=args.threshold
    )

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
    query_interface = QueryInterface(args.project_path)
    stats = query_interface.get_stats()

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
        print(f"✓ Backed up existing hook to {backup}")

    # Install hook
    shutil.copy(hook_source, hook_dest)
    hook_dest.chmod(0o755)

    print(f"✓ Installed post-commit hook at {hook_dest}")
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

    print(f"\n✓ Migrated {total_messages} conversation messages")
    print(f"  Use search-conversations to search conversations")


def cmd_list_projects(args):
    """List all indexed projects"""
    from qdrant_client import QdrantClient
    from collections import defaultdict

    metadata = ProjectMetadata()

    # Get collections from Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
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
                print(f"  Status: ⚠️  Path no longer exists")
            else:
                print(f"  Status: ✓ Active")

            print(f"  Indexed: {metadata_info.get('indexed_at', 'unknown')}")
            print(f"  Updated: {metadata_info.get('last_updated', 'unknown')}")
            print(f"  Files: {metadata_info.get('file_count', 'unknown')}")

            if metadata_info.get('collection_stats'):
                total_points = sum(metadata_info['collection_stats'].values())
                print(f"  Vectors: {total_points:,}")
        else:
            print(f"  Path: ⚠️  Unknown (not in metadata registry)")
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
        print(f"\n⚠️  Found {len(orphaned_meta)} projects in metadata but not in Qdrant:")
        for project_id, info in orphaned_meta:
            print(f"  - {project_id}: {info['path']}")
        print("\nRun: code-vector-cli cleanup-metadata")


def cmd_cleanup_metadata(args):
    """Clean up metadata for projects that no longer exist"""
    metadata = ProjectMetadata()
    removed = metadata.cleanup_missing_projects()

    if removed:
        print(f"\n✓ Removed {len(removed)} missing projects:")
        for item in removed:
            print(f"  - {item}")
    else:
        print("\n✓ No missing projects found. All metadata is valid.")


def cmd_delete(args):
    """Delete all collections for the project"""
    from code_vector_db.vector_store import VectorStore

    vector_store = VectorStore(args.project_path)

    # Confirm deletion unless --force is used
    if not args.force:
        project_id = vector_store.project_id
        print(f"\n⚠️  WARNING: This will delete all indexed data for:")
        print(f"   Path: {args.project_path}")
        print(f"   Project ID: {project_id}")
        print(f"\nCollections to be deleted:")
        for collection in VectorStore.ALL_COLLECTIONS:
            print(f"   - {collection}")

        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("\n✗ Deletion cancelled")
            return

    # Delete collections
    print(f"\nDeleting collections for: {args.project_path}")
    vector_store.delete_collections()

    # Clean up metadata
    metadata = ProjectMetadata()
    metadata.unregister_project(args.project_path)

    print(f"\n✓ Successfully deleted all data for project")


def main():
    parser = argparse.ArgumentParser(
        description="Code Vector Database CLI - Semantic code search powered by vector embeddings"
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Directory to index/search (auto-detects single-repo or multi-repo structure)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init command
    subparsers.add_parser("init", help="Initialize vector database and index")

    # index command
    index_parser = subparsers.add_parser("index", help="Index codebase")
    index_parser.add_argument("--incremental", action="store_true", help="Incremental indexing (coming soon)")
    index_parser.add_argument("--repo", type=str, help="Index only specific repo in multi-repo workspace")

    # reindex-file command
    reindex_parser = subparsers.add_parser("reindex-file", help="Reindex single file")
    reindex_parser.add_argument("file", help="File to reindex")

    # search command
    search_parser = subparsers.add_parser("search", help="Search code semantically")
    search_parser.add_argument("query", help="Search query (natural language)")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    search_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold (0.0-1.0)")
    search_parser.add_argument("--show-parent", action="store_true", help="Show parent class/module")
    search_parser.add_argument("--show-content", action="store_true", help="Show code snippets")
    search_parser.add_argument("-C", "--context-lines", type=int, default=3, help="Context lines (default: 3)")

    # similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar code")
    similar_parser.add_argument("query", help="File path OR semantic query")
    similar_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    similar_parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Score threshold (0.0-1.0)")
    similar_parser.add_argument("--show-content", action="store_true", help="Show code snippets")
    similar_parser.add_argument("-C", "--context-lines", type=int, default=3, help="Context lines (default: 3)")

    # context command
    context_parser = subparsers.add_parser("context", help="Get context for task")
    context_parser.add_argument("task", help="Task description")
    context_parser.add_argument("-n", "--limit", type=int, default=10, help="Max files")
    context_parser.add_argument("-t", "--threshold", type=float, default=0.4, help="Score threshold (0.0-1.0)")
    context_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # impact command
    impact_parser = subparsers.add_parser("impact", help="Analyze change impact")
    impact_parser.add_argument("query", help="File path OR semantic query")
    impact_parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Score threshold (0.0-1.0)")

    # search-docs command
    docs_parser = subparsers.add_parser("search-docs", help="Search documentation")
    docs_parser.add_argument("query", help="Search query")
    docs_parser.add_argument("-n", "--limit", type=int, default=10, help="Number of results")
    docs_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold (0.0-1.0)")

    # search-conversations command
    conv_parser = subparsers.add_parser("search-conversations", help="Search conversation history")
    conv_parser.add_argument("query", help="Search query")
    conv_parser.add_argument("-n", "--limit", type=int, default=5, help="Number of results")
    conv_parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Score threshold (0.0-1.0)")

    # stats command
    subparsers.add_parser("stats", help="Show index statistics")

    # install-hook command
    subparsers.add_parser("install-hook", help="Install git post-commit hook")

    # migrate-conversations command
    subparsers.add_parser("migrate-conversations", help="Migrate conversation transcripts")

    # list-projects command
    list_parser = subparsers.add_parser("list-projects", help="List all indexed projects")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed stats")

    # cleanup-metadata command
    subparsers.add_parser("cleanup-metadata", help="Clean up metadata for missing projects")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete indexed data")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Resolve path
    args.path = os.path.abspath(args.path)

    # Set legacy aliases for backward compatibility
    args.project_path = args.path
    args.workspace_path = args.path

    # Dispatch to command handler
    command_map = {
        "init": cmd_init,
        "index": cmd_index,
        "reindex-file": cmd_reindex_file,
        "search": cmd_search,
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
