"""Query interface for semantic search"""

from typing import List, Dict, Optional, Any
from pathlib import Path

from .vector_store import VectorStore


class SearchResult:
    """Represents a search result"""

    def __init__(self, score: float, metadata: Dict[str, Any]):
        self.score = score
        self.rerank_score = None  # set when a reranker reorders this result
        self.file_path = metadata.get("file_path", "")
        self.name = metadata.get("name", "")
        self.type = metadata.get("type", "")
        self.start_line = metadata.get("start_line", 0)
        self.end_line = metadata.get("end_line", 0)
        self.language = metadata.get("language", "")
        self.parent = metadata.get("parent", "")
        self.content = metadata.get("content", "")
        self.metadata = metadata

    # Bounded snippet size: enough to triage a result in one call without
    # flooding the agent's context on a multi-result search.
    SNIPPET_MAX_LINES = 20
    SNIPPET_MAX_CHARS = 1500

    def _signature(self) -> str:
        """Best-effort signature: the declaration line(s) up to the body start."""
        if not self.content:
            return ""
        sig_lines = []
        for line in self.content.splitlines():
            sig_lines.append(line)
            # Stop at the first body opener so we keep just the declaration.
            if "{" in line or line.rstrip().endswith(":") or line.rstrip().endswith("=>"):
                break
            if len(sig_lines) >= 4:  # guard against runaway multi-line headers
                break
        sig = "\n".join(sig_lines).strip()
        return sig[:300]

    def _snippet(self) -> str:
        """A bounded code snippet for one-call triage (full body via `lines`)."""
        if not self.content:
            return ""
        lines = self.content.splitlines()
        snippet = "\n".join(lines[:self.SNIPPET_MAX_LINES])
        if len(snippet) > self.SNIPPET_MAX_CHARS:
            snippet = snippet[:self.SNIPPET_MAX_CHARS]
        truncated = len(lines) > self.SNIPPET_MAX_LINES or len(snippet) >= self.SNIPPET_MAX_CHARS
        return snippet + ("\n  ... (truncated, read full range for more)" if truncated else "")

    def _kind(self) -> str:
        """Classify a result as source vs test vs docs from its path.

        Lets an agent prioritize implementation over tests without reading files.
        """
        p = self.file_path.lower().replace("\\", "/")
        if self.type in ("documentation", "configuration"):
            return self.type
        if "/test" in p or p.startswith("test") or ".test." in p or ".spec." in p or "tests/" in p:
            return "test"
        return "source"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a result shape optimized for AI agent consumption.

        - Single `relevance` score (rerank when present, else vector similarity)
          so the agent never has to guess which score ordered the list.
        - `signature` + bounded `code` snippet so an agent can triage in one call
          instead of issuing a follow-up read per result.
        - `kind` (source/test/...) to prioritize implementation over tests.
        - Empty fields are omitted to keep responses lean.
        """
        relevance = self.rerank_score if self.rerank_score is not None else self.score

        d = {
            "relevance": round(relevance, 3),
            "file_path": self.file_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "type": self.type,
        }
        if self.name:
            d["name"] = self.name
        if self.parent:
            d["parent"] = self.parent
        if self.language:
            d["language"] = self.language

        kind = self._kind()
        if kind != "source":
            d["kind"] = kind

        sig = self._signature()
        if sig:
            d["signature"] = sig

        snippet = self._snippet()
        if snippet:
            d["code"] = snippet

        repo = self.metadata.get("repo")
        if repo:
            d["repo"] = repo
        return d

    def __repr__(self):
        location = f"{self.file_path}:{self.start_line}"
        return f"<SearchResult score={self.score:.3f} {location} {self.name}>"


class QueryInterface:
    """Interface for querying the vector database"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.vector_store = VectorStore(str(self.project_path))
        self._code_embedder = None
        self._text_embedder = None

    @property
    def code_embedder(self):
        if self._code_embedder is None:
            from .embeddings import get_code_embedder
            self._code_embedder = get_code_embedder()
        return self._code_embedder

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            from .embeddings import get_text_embedder
            self._text_embedder = get_text_embedder()
        return self._text_embedder

    def _lines_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
        """Check if two line ranges overlap"""
        return start1 <= end2 and start2 <= end1

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results from overlapping line ranges in the same file.

        Keeps the higher-scored, more specific result. Specificity order:
        function > class > file
        """
        if not results:
            return results

        # Specificity ranking (lower = more specific = preferred)
        specificity = {"function": 0, "method": 0, "class": 1, "file": 2}

        deduplicated = []
        for result in results:
            is_duplicate = False
            for i, existing in enumerate(deduplicated):
                if existing.file_path != result.file_path:
                    continue
                if not self._lines_overlap(
                    existing.start_line, existing.end_line,
                    result.start_line, result.end_line
                ):
                    continue

                # Same file, overlapping lines — keep the better one
                existing_spec = specificity.get(existing.type, 1)
                result_spec = specificity.get(result.type, 1)

                if result.score > existing.score or (
                    result.score == existing.score and result_spec < existing_spec
                ):
                    # New result is better — replace existing
                    deduplicated[i] = result

                is_duplicate = True
                break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated

    def _result_text_for_rerank(self, result: "SearchResult") -> str:
        """Build the text a cross-encoder should score for a result.

        Prefer the real code/doc content (stored in the payload); fall back to a
        synthetic context line built from identifiers when content is absent.
        """
        content = result.content or ""
        header = " ".join(p for p in [result.file_path, result.parent, result.name] if p)
        if content:
            return f"{header}\n{content}".strip()
        return f"{header} ({result.type})".strip()

    def _maybe_rerank(
        self,
        query_text: Optional[str],
        results: List["SearchResult"],
        limit: int
    ) -> List["SearchResult"]:
        """Rerank with zerank-1 when enabled; otherwise just truncate."""
        from .reranker import rerank_enabled, get_reranker

        if not query_text or not results or not rerank_enabled():
            return results[:limit]

        def _set_score(r, s):
            r.rerank_score = s

        reranked = get_reranker().rerank(
            query=query_text,
            candidates=results,
            get_text=self._result_text_for_rerank,
            top_n=limit,
            set_score=_set_score,
        )
        return reranked[:limit]

    def _vector_pool(
        self,
        query_vector: List[float],
        limit: int = 50,
        filters: Optional[Dict] = None,
        threshold: float = 0.3,
    ) -> List[SearchResult]:
        """Return a deduplicated candidate pool by pure vector score (no rerank).

        Used by hybrid search, which fuses keyword scores then reranks once at
        the end. Keeping retrieval and reranking separate avoids double-ordering.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        collections = [
            VectorStore.CODE_FUNCTIONS,
            VectorStore.CODE_CLASSES,
            VectorStore.CODE_FILES,
        ]
        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.vector_store.search,
                    collection=c, query_vector=query_vector,
                    limit=limit, filters=filters, score_threshold=threshold,
                ): c for c in collections
            }
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    print(f"Error searching collection {futures[future]}: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        search_results = [SearchResult(r["score"], r["metadata"]) for r in results]
        return self._deduplicate_results(search_results)

    def _search_parallel(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
        threshold: float = 0.3,
        query_text: Optional[str] = None,
    ) -> List[SearchResult]:
        """Run searches across code collections in parallel, then rerank.

        When reranking is enabled we gather a WIDER candidate pool per collection
        so the cross-encoder has real choices, then reorder to `limit`.
        query_text is the raw search string the reranker scores against.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .reranker import rerank_enabled

        collections = [
            VectorStore.CODE_FUNCTIONS,
            VectorStore.CODE_CLASSES,
            VectorStore.CODE_FILES
        ]

        # Widen the pool when we'll rerank (more candidates -> better top-N).
        pool_limit = max(limit, 30) if (query_text and rerank_enabled()) else limit

        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_collection = {
                executor.submit(
                    self.vector_store.search,
                    collection=collection,
                    query_vector=query_vector,
                    limit=pool_limit,
                    filters=filters,
                    score_threshold=threshold
                ): collection for collection in collections
            }

            for future in as_completed(future_to_collection):
                try:
                    collection_results = future.result()
                    results.extend(collection_results)
                except Exception as e:
                    print(f"Error searching collection {future_to_collection[future]}: {e}")

        # Sort by vector score, build results, dedup overlapping ranges
        results.sort(key=lambda x: x["score"], reverse=True)
        search_results = [SearchResult(r["score"], r["metadata"]) for r in results]
        search_results = self._deduplicate_results(search_results)

        # Final stage: cross-encoder rerank to the requested limit
        return self._maybe_rerank(query_text, search_results, limit)

    def search_code(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        threshold: float = 0.3,
        repo: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for code using semantic similarity across multiple collections in parallel"""
        query_vector = self.code_embedder.embed(query, input_type="query")[0]

        # Build filters
        search_filters = dict(filters) if filters else {}
        if repo:
            search_filters["repo"] = repo

        return self._search_parallel(query_vector, limit, search_filters or None, threshold, query_text=query)

    def search_documentation(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """Search documentation"""
        query_vector = self.text_embedder.embed(query, input_type="query")[0]

        results = self.vector_store.search(
            collection=VectorStore.DOCUMENTATION,
            query_vector=query_vector,
            limit=limit,
            score_threshold=threshold
        )

        return [SearchResult(r["score"], r["metadata"]) for r in results]

    def search_conversations(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """Search conversation history"""
        query_vector = self.text_embedder.embed(query, input_type="query")[0]

        results = self.vector_store.search(
            collection=VectorStore.CONVERSATIONS,
            query_vector=query_vector,
            limit=limit,
            score_threshold=threshold
        )

        return [SearchResult(r["score"], r["metadata"]) for r in results]

    def find_similar_to_file(
        self,
        file_path: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Find code similar to a specific file"""
        path = Path(file_path)
        if not path.exists():
            return []

        # Read file content
        try:
            content = path.read_text(encoding='utf-8', errors='replace')
        except (IOError, OSError) as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            return []

        # Generate embedding (searching WITH this file's content -> query side)
        query_vector = self.code_embedder.embed(content, input_type="query")[0]

        # Use parallel search (rerank against the file's own content)
        results = self._search_parallel(query_vector, limit, threshold=threshold, query_text=content[:4000])

        # Filter out the file itself
        results = [r for r in results if r.file_path != str(path)]
        return results

    def find_similar(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        repo: Optional[str] = None
    ) -> List[SearchResult]:
        """Find similar code - accepts file path OR semantic query"""
        # Check if it's a file path
        path = Path(query)
        if path.exists() and path.is_file():
            return self.find_similar_to_file(str(path), limit, threshold)

        # Otherwise treat as semantic query
        query_vector = self.code_embedder.embed(query, input_type="query")[0]
        filters = {"repo": repo} if repo else None
        return self._search_parallel(query_vector, limit, filters, threshold=threshold, query_text=query)

    def get_context_for_task(
        self,
        task_description: str,
        max_files: int = 10,
        threshold: float = 0.4,
        repo: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context files for a task"""
        # Search code
        code_results = self.search_code(task_description, limit=max_files, threshold=threshold, repo=repo)

        # Search documentation
        doc_results = self.search_documentation(task_description, limit=max_files // 2, threshold=threshold)

        # Combine and deduplicate by file
        file_scores = {}
        for result in code_results + doc_results:
            if result.file_path not in file_scores:
                file_scores[result.file_path] = {
                    "file_path": result.file_path,
                    "score": result.score,
                    "reason": f"{result.type}: {result.name}" if result.name else result.type,
                    "lines": f"{result.start_line}-{result.end_line}" if result.start_line else ""
                }
            else:
                # Update score if higher
                if result.score > file_scores[result.file_path]["score"]:
                    file_scores[result.file_path]["score"] = result.score

        # Sort by score
        context_files = sorted(file_scores.values(), key=lambda x: x["score"], reverse=True)
        return context_files[:max_files]

    def analyze_impact(
        self,
        query: str,
        depth: int = 2,
        threshold: float = 0.6,
        repo: Optional[str] = None
    ) -> Dict[str, List[SearchResult]]:
        """Analyze impact - accepts file path OR semantic query"""
        # Check if it's a file path
        path = Path(query)
        is_file = path.exists() and path.is_file()

        # Find directly similar code
        direct_results = self.find_similar(query, limit=20, threshold=threshold, repo=repo)

        if depth <= 1:
            return {"direct": direct_results, "indirect": [], "query_type": "file" if is_file else "semantic"}

        # Find indirectly similar code (2nd hop)
        indirect_results = []
        seen_paths = {r.file_path for r in direct_results}
        if is_file:
            seen_paths.add(str(path))

        for result in direct_results[:10]:  # Limit to top 10 for performance
            if Path(result.file_path).exists():
                try:
                    similar = self.find_similar_to_file(
                        result.file_path,
                        limit=10,
                        threshold=threshold
                    )
                    for sim in similar:
                        if sim.file_path not in seen_paths:
                            indirect_results.append(sim)
                            seen_paths.add(sim.file_path)
                except Exception:
                    # Skip files that can't be analyzed
                    continue

        # Sort indirect results by score
        indirect_results.sort(key=lambda x: x.score, reverse=True)

        return {
            "direct": direct_results,
            "indirect": indirect_results[:20],
            "query_type": "file" if is_file else "semantic"
        }

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        repo: Optional[str] = None
    ) -> List[SearchResult]:
        """Hybrid search combining BM25 keyword matching with semantic search

        Args:
            query: Search query
            limit: Number of results to return
            threshold: Minimum semantic similarity threshold
            bm25_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
            repo: Filter by repository name

        Pipeline: dense retrieval builds a wide pool -> bm25s keyword scores are
        fused in (weighted) to SELECT the candidates -> zerank-1 (if enabled)
        does the FINAL ordering. When reranking is off, the fused score orders
        results directly.
        """
        import bm25s

        # Dense candidate pool WITHOUT internal reranking (we rerank once at the
        # end). Pull the raw vector results via a dedicated pool helper.
        query_vector = self.code_embedder.embed(query, input_type="query")[0]
        filters = {"repo": repo} if repo else None
        pool = self._vector_pool(query_vector, limit=limit * 5,
                                 filters=filters, threshold=max(0.1, threshold - 0.2))

        if not pool:
            return []

        # bm25s keyword scoring over real content (path + parent + name + code).
        corpus = [self._result_text_for_rerank(r).lower() for r in pool]
        corpus_tokens = bm25s.tokenize(corpus, stopwords=None, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)

        query_tokens = bm25s.tokenize(query.lower(), stopwords=None, show_progress=False)
        # Score every doc in the pool (k = pool size) and remap to pool order.
        idxs, scores = retriever.retrieve(query_tokens, k=len(pool), show_progress=False)
        bm25_by_idx = {int(i): float(s) for i, s in zip(idxs[0], scores[0])}
        max_bm25 = max(bm25_by_idx.values()) if bm25_by_idx and max(bm25_by_idx.values()) > 0 else 1.0

        # Fuse dense + keyword to select/score candidates.
        for i, result in enumerate(pool):
            bm25_norm = bm25_by_idx.get(i, 0.0) / max_bm25
            result.score = semantic_weight * result.score + bm25_weight * bm25_norm

        pool.sort(key=lambda r: r.score, reverse=True)

        # Final stage: zerank-1 reranks the fused top candidates (graceful
        # fallback to fused order when disabled/unavailable).
        return self._maybe_rerank(query, pool, limit)

    def search_git_history(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3,
        repo: Optional[str] = None
    ) -> List[SearchResult]:
        """Search git commit history"""
        query_vector = self.text_embedder.embed(query, input_type="query")[0]

        filters = {"repo": repo} if repo else None

        results = self.vector_store.search(
            collection=VectorStore.GIT_HISTORY,
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            score_threshold=threshold
        )

        return [SearchResult(r["score"], r["metadata"]) for r in results]

    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        return self.vector_store.get_stats()
