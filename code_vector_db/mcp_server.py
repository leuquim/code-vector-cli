"""MCP server for code vector database - exposes search tools via Model Context Protocol"""

import os
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP(
    "code-vector-db",
    instructions=(
        "Semantic code search over a local vector index (zembed-1 embeddings + "
        "zerank-1 reranking). Use semantic_search for conceptual queries and "
        "hybrid_search when you know specific identifiers/filenames. get_context "
        "gathers relevant files for a task; analyze_impact finds code affected by "
        "a change before refactoring. Results include file_path and line ranges "
        "you can open directly."
    )
)

# Lazy singleton for QueryInterface
_query_interface = None
_project_path = None


def _get_query_interface():
    """Get or create the QueryInterface singleton"""
    global _query_interface, _project_path
    if _query_interface is None:
        if _project_path is None:
            raise RuntimeError("Project path not set. Start server with --path argument.")
        from code_vector_db.query import QueryInterface
        _query_interface = QueryInterface(_project_path)
    return _query_interface


def set_project_path(path: str):
    """Set the project path for the server"""
    global _project_path, _query_interface
    _project_path = os.path.abspath(path)
    _query_interface = None  # Reset to force re-init


@mcp.tool()
def semantic_search(query: str, limit: int = 10, threshold: float = 0.3, repo: Optional[str] = None) -> str:
    """Search code semantically using natural language.

    Find functions, classes, and files by describing what they do. Results are
    retrieved by vector similarity then reordered by a cross-encoder reranker
    for precision; each result includes file_path, line range, score, and
    rerank_score (when reranking is active).
    Examples: "user authentication logic", "database connection pooling", "error handling middleware"

    Args:
        query: Natural language search query
        limit: Maximum number of results (default: 10)
        threshold: Minimum similarity score 0.0-1.0 (default: 0.3)
        repo: Filter by repository name (e.g., "base", "frontend", "mobile_app")
    """
    qi = _get_query_interface()
    results = qi.search_code(query, limit=limit, threshold=threshold, repo=repo)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def hybrid_search(query: str, limit: int = 10, threshold: float = 0.3, repo: Optional[str] = None) -> str:
    """Hybrid search: vector similarity fused with BM25 keyword matching, then reranked.

    Better than semantic_search when you know specific identifiers or filenames.
    Dense retrieval and keyword scores select candidates; a cross-encoder makes
    the final ordering. Examples: "UpdatePausedTimeResource", "handleCheckout function", "toast error notification"

    Args:
        query: Search query (can include code identifiers)
        limit: Maximum number of results (default: 10)
        threshold: Minimum score 0.0-1.0 (default: 0.3)
        repo: Filter by repository name
    """
    qi = _get_query_interface()
    results = qi.search_hybrid(query, limit=limit, threshold=threshold, repo=repo)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def find_similar(query: str, limit: int = 10, threshold: float = 0.7) -> str:
    """Find code similar to a given file path or description.

    Use to find duplicate code, related implementations, or similar patterns.

    Args:
        query: File path or semantic description of code to find similar matches for
        limit: Maximum number of results (default: 10)
        threshold: Minimum similarity score 0.0-1.0 (default: 0.7)
    """
    qi = _get_query_interface()
    results = qi.find_similar(query, limit=limit, threshold=threshold)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def get_context(task: str, limit: int = 10, threshold: float = 0.4, repo: Optional[str] = None) -> str:
    """Get relevant files and context for a task description.

    Use when starting work on a feature or bug fix to find all related code.
    Examples: "implement pause/resume for appointments", "fix checkout timer bug"

    Args:
        task: Description of the task or feature
        limit: Maximum number of files to return (default: 10)
        threshold: Minimum relevance score 0.0-1.0 (default: 0.4)
        repo: Filter by repository name
    """
    qi = _get_query_interface()
    context_files = qi.get_context_for_task(task, max_files=limit, threshold=threshold, repo=repo)
    return json.dumps(context_files, indent=2)


@mcp.tool()
def analyze_impact(query: str, threshold: float = 0.6) -> str:
    """Analyze the impact of changing a file or code area.

    Shows directly and indirectly affected code. Use before refactoring.

    Args:
        query: File path or description of code to analyze
        threshold: Minimum similarity for impact detection 0.0-1.0 (default: 0.6)
    """
    qi = _get_query_interface()
    results = qi.analyze_impact(query, depth=2, threshold=threshold)
    output = {
        "query": query,
        "query_type": results.get("query_type", "unknown"),
        "direct": [r.to_dict() for r in results["direct"]],
        "indirect": [r.to_dict() for r in results["indirect"]],
    }
    return json.dumps(output, indent=2)


@mcp.tool()
def search_docs(query: str, limit: int = 10, threshold: float = 0.3) -> str:
    """Search project documentation (markdown, text, config files).

    Args:
        query: Search query for documentation
        limit: Maximum number of results (default: 10)
        threshold: Minimum relevance score 0.0-1.0 (default: 0.3)
    """
    qi = _get_query_interface()
    results = qi.search_documentation(query, limit=limit, threshold=threshold)
    return json.dumps([r.to_dict() for r in results], indent=2)


@mcp.tool()
def get_stats() -> str:
    """Get vector database index statistics.

    Shows collection sizes and total indexed vectors.
    """
    qi = _get_query_interface()
    stats = qi.get_stats()
    return json.dumps(stats, indent=2)


def run_server():
    """Run the MCP server (stdio transport)"""
    mcp.run(transport="stdio")
