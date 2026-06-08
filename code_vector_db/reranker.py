"""Cross-encoder reranking via ZeroEntropy zerank-1.

Two-stage retrieval: a wide candidate pool from dense (+BM25) search is
re-scored by a cross-encoder that sees the query and each candidate jointly.
This is the single highest-leverage retrieval-quality upgrade (Anthropic's
contextual-retrieval study measured reranking as the largest single gain).

Enabled by default when EMBEDDING_PROVIDER=zeroentropy. Set RERANK_ENABLED=false
to disable. Degrades gracefully: any API error falls back to the original
vector-score order, so search never hard-fails on the reranker.
"""

import os
from typing import List, Callable, Any, Optional

ZERANK_MODEL = "zerank-1"


def rerank_enabled() -> bool:
    """Whether reranking should run for the current configuration."""
    explicit = os.environ.get("RERANK_ENABLED", "").strip().lower()
    if explicit in ("true", "1", "yes"):
        return True
    if explicit in ("false", "0", "no"):
        return False
    # Default: on when the ZeroEntropy provider is active (key is present).
    provider = os.environ.get("EMBEDDING_PROVIDER", "").strip().lower()
    if not provider and os.environ.get("USE_OPENAI_EMBEDDINGS", "").lower() == "true":
        provider = "openai"
    if not provider:
        provider = "zeroentropy"
    return provider == "zeroentropy" and bool(os.environ.get("ZEROENTROPY_API_KEY"))


class Reranker:
    """Reranks candidates with zerank-1, reusing the ZeroEntropy client."""

    def __init__(self, model: str = ZERANK_MODEL):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from zeroentropy import ZeroEntropy
            api_key = os.environ.get("ZEROENTROPY_API_KEY")
            if not api_key:
                raise ValueError("ZEROENTROPY_API_KEY not set")
            self._client = ZeroEntropy(api_key=api_key)
        return self._client

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        get_text: Callable[[Any], str],
        top_n: Optional[int] = None,
        set_score: Optional[Callable[[Any, float], None]] = None,
    ) -> List[Any]:
        """Reorder `candidates` by cross-encoder relevance to `query`.

        Args:
            query: the search string.
            candidates: arbitrary items to reorder.
            get_text: maps a candidate to the text the reranker should score.
            top_n: keep only the top N after reranking (default: all).
            set_score: optional hook to record the rerank score on each item.

        Returns the candidates reordered best-first. On any failure, returns the
        original list unchanged (truncated to top_n if given).
        """
        if not candidates:
            return candidates

        n = top_n or len(candidates)

        try:
            documents = [get_text(c) for c in candidates]
            resp = self.client.models.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=n,
            )
            reordered = []
            for r in resp.results:
                item = candidates[r.index]
                if set_score is not None:
                    set_score(item, r.relevance_score)
                reordered.append(item)
            return reordered
        except Exception as e:
            # Never let the reranker break search — fall back to vector order.
            print(f"  [rerank] falling back to vector order ({type(e).__name__}: {str(e)[:100]})")
            return candidates[:n]


_reranker = None


def get_reranker() -> Reranker:
    """Get the shared Reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
