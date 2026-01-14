"""Query interface for semantic search"""

from typing import List, Dict, Optional, Any
from pathlib import Path

from .embeddings import get_code_embedder, get_text_embedder
from .vector_store import VectorStore


class SearchResult:
    """Represents a search result"""

    def __init__(self, score: float, metadata: Dict[str, Any]):
        self.score = score
        self.file_path = metadata.get("file_path", "")
        self.name = metadata.get("name", "")
        self.type = metadata.get("type", "")
        self.start_line = metadata.get("start_line", 0)
        self.end_line = metadata.get("end_line", 0)
        self.language = metadata.get("language", "")
        self.parent = metadata.get("parent", "")
        self.content = metadata.get("content", "")
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "score": round(self.score, 3),
            "file_path": self.file_path,
            "name": self.name,
            "type": self.type,
            "lines": f"{self.start_line}-{self.end_line}",
            "language": self.language,
            "parent": self.parent,
        }

    def __repr__(self):
        location = f"{self.file_path}:{self.start_line}"
        return f"<SearchResult score={self.score:.3f} {location} {self.name}>"


class QueryInterface:
    """Interface for querying the vector database"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.vector_store = VectorStore(str(self.project_path))
        self.code_embedder = get_code_embedder()
        self.text_embedder = get_text_embedder()

    def _search_parallel(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """Helper to run searches across code collections in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        collections = [
            VectorStore.CODE_FUNCTIONS,
            VectorStore.CODE_CLASSES,
            VectorStore.CODE_FILES
        ]

        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_collection = {
                executor.submit(
                    self.vector_store.search,
                    collection=collection,
                    query_vector=query_vector,
                    limit=limit,
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

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        return [SearchResult(r["score"], r["metadata"]) for r in results]

    def search_code(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """Search for code using semantic similarity across multiple collections in parallel"""
        # Generate query embedding
        query_vector = self.code_embedder.embed(query)[0]
        return self._search_parallel(query_vector, limit, filters, threshold)

    def search_documentation(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """Search documentation"""
        query_vector = self.text_embedder.embed(query)[0]

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
        query_vector = self.text_embedder.embed(query)[0]

        results = self.vector_store.search(
            collection=VectorStore.CONVERSATIONS,
            query_vector=query_vector,
            limit=limit,
            score_threshold=threshold
        )

        return [SearchResult(r["score"], r["metadata"]) for r in results]

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.3
    ) -> Dict[str, List[SearchResult]]:
        """Search across code, docs, and conversations"""
        return {
            "code": self.search_code(query, limit=limit // 3, threshold=threshold),
            "documentation": self.search_documentation(query, limit=limit // 3, threshold=threshold),
            "conversations": self.search_conversations(query, limit=limit // 3, threshold=threshold)
        }

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
            content = path.read_text(errors='ignore')
        except (IOError, OSError) as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            return []

        # Generate embedding
        query_vector = self.code_embedder.embed(content)[0]

        # Use parallel search
        results = self._search_parallel(query_vector, limit, threshold=threshold)

        # Filter out the file itself
        results = [r for r in results if r.file_path != str(path)]
        return results

    def find_similar(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Find similar code - accepts file path OR semantic query"""
        # Check if it's a file path
        path = Path(query)
        if path.exists() and path.is_file():
            return self.find_similar_to_file(str(path), limit, threshold)

        # Otherwise treat as semantic query
        query_vector = self.code_embedder.embed(query)[0]
        
        # Use parallel search
        return self._search_parallel(query_vector, limit, threshold=threshold)

    def get_context_for_task(
        self,
        task_description: str,
        max_files: int = 10,
        threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Get relevant context files for a task"""
        # Search code
        code_results = self.search_code(task_description, limit=max_files, threshold=threshold)

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
        threshold: float = 0.6
    ) -> Dict[str, List[SearchResult]]:
        """Analyze impact - accepts file path OR semantic query"""
        # Check if it's a file path
        path = Path(query)
        is_file = path.exists() and path.is_file()

        # Find directly similar code
        direct_results = self.find_similar(query, limit=20, threshold=threshold)

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
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """Hybrid search combining BM25 keyword matching with semantic search

        Args:
            query: Search query
            limit: Number of results to return
            threshold: Minimum semantic similarity threshold
            bm25_weight: Weight for BM25 scores (0-1)
            semantic_weight: Weight for semantic scores (0-1)
        """
        from rank_bm25 import BM25Okapi
        import numpy as np

        # Get semantic results (broader set for re-ranking)
        semantic_results = self.search_code(query, limit=limit * 3, threshold=max(0.1, threshold - 0.2))

        if not semantic_results:
            return []

        # Build BM25 index from semantic results
        corpus = []
        for result in semantic_results:
            # Combine name, type, and file path for keyword matching
            doc_text = f"{result.name} {result.type} {result.file_path}"
            corpus.append(doc_text.lower().split())

        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Normalize scores to 0-1 range
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_normalized = bm25_scores / max_bm25

        # Combine scores
        combined_results = []
        for i, result in enumerate(semantic_results):
            combined_score = (
                semantic_weight * result.score +
                bm25_weight * bm25_normalized[i]
            )

            # Create new result with combined score
            combined_result = SearchResult(combined_score, result.metadata)
            combined_results.append(combined_result)

        # Sort by combined score and apply threshold
        combined_results.sort(key=lambda x: x.score, reverse=True)
        filtered_results = [r for r in combined_results if r.score >= threshold]

        return filtered_results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        return self.vector_store.get_stats()
