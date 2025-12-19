"""Qdrant vector store management"""

import os
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path
from qdrant_client import QdrantClient
from code_vector_db import normalize_path_for_id
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType
)


class VectorStore:
    """Manages Qdrant collections and operations"""

    # Collection names
    CODE_FUNCTIONS = "code_functions"
    CODE_CLASSES = "code_classes"
    CODE_FILES = "code_files"
    DOCUMENTATION = "documentation"
    GIT_HISTORY = "git_history"
    CONVERSATIONS = "conversations"

    ALL_COLLECTIONS = [
        CODE_FUNCTIONS, CODE_CLASSES, CODE_FILES,
        DOCUMENTATION, GIT_HISTORY, CONVERSATIONS
    ]

    def __init__(self, project_path: str, host: str = "localhost", port: int = 6333):
        # Check for local mode (no server required - faster on Windows)
        use_local = os.environ.get("QDRANT_LOCAL", "").lower() == "true"
        local_path = os.environ.get("QDRANT_LOCAL_PATH",
                                     os.path.expanduser("~/.local/share/code-vector-db/qdrant-local"))

        if use_local:
            # Local embedded mode - no server required, native performance
            os.makedirs(local_path, exist_ok=True)
            self.client = QdrantClient(path=local_path)
            self._mode = "local"
        else:
            # Remote server mode - check env vars for host/port
            host = os.environ.get("QDRANT_HOST", host)
            port = int(os.environ.get("QDRANT_PORT", port))
            self.client = QdrantClient(
                host=host,
                port=port,
                timeout=300  # 5 minute timeout for operations
            )
            self._mode = "remote"

        self.project_path = project_path
        self.project_id = self._get_project_id(project_path)

    def _get_project_id(self, project_path: str) -> str:
        """Generate unique project ID from normalized path (cross-platform compatible)"""
        normalized = normalize_path_for_id(project_path)
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _collection_name(self, base_name: str) -> str:
        """Get project-specific collection name"""
        return f"{self.project_id}_{base_name}"

    def initialize_collections(self):
        """Create all collections for the project"""
        # Detect which embedder is being used via environment variable
        use_openai = os.environ.get("USE_OPENAI_EMBEDDINGS", "").lower() == "true"

        if use_openai:
            # OpenAI embeddings: text-embedding-3-small = 1536 dimensions
            code_vector_size = 1536
            text_vector_size = 1536
            print(f"  Using OpenAI embeddings (1536 dimensions)")
        else:
            # Local embeddings: CodeT5+ = 256, mpnet = 768
            code_vector_size = 256
            text_vector_size = 768
            print(f"  Using local embeddings (CodeT5+: 256, mpnet: 768 dimensions)")

        # Code collections
        for collection in [self.CODE_FUNCTIONS, self.CODE_CLASSES, self.CODE_FILES]:
            self._create_collection(
                self._collection_name(collection),
                vector_size=code_vector_size,
                distance=Distance.COSINE
            )

        # Text collections
        for collection in [self.DOCUMENTATION, self.GIT_HISTORY, self.CONVERSATIONS]:
            self._create_collection(
                self._collection_name(collection),
                vector_size=text_vector_size,
                distance=Distance.COSINE
            )

    def _create_collection(self, name: str, vector_size: int, distance: Distance):
        """Create a collection with optimal configuration"""
        try:
            self.client.get_collection(name)
            print(f"Collection {name} already exists")
        except Exception:
            from qdrant_client.models import OptimizersConfigDiff

            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99
                    )
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=50000,  # Higher threshold reduces segments
                    max_segment_size=200000,   # Larger segments
                    memmap_threshold=50000     # Use memory mapping for large segments
                )
            )
            print(f"[OK] Created collection: {name}")

    def delete_collections(self):
        """Delete all project collections"""
        for collection in self.ALL_COLLECTIONS:
            try:
                self.client.delete_collection(self._collection_name(collection))
                print(f"[OK] Deleted collection: {collection}")
            except Exception as e:
                print(f"  Collection {collection} does not exist or already deleted")

    def upsert_points(
        self,
        collection: str,
        points: List[Dict[str, Any]]
    ):
        """Insert or update points in a collection"""
        collection_name = self._collection_name(collection)

        # Convert all points to Qdrant format first
        qdrant_points = []
        for i, point in enumerate(points):
            point_id = point.get("id", hashlib.md5(
                f"{point['metadata']['file_path']}:{point['metadata']['start_line']}".encode()
            ).hexdigest())

            qdrant_points.append(PointStruct(
                id=point_id,
                vector=point["vector"],
                payload=point["metadata"]
            ))

        # Batch upsert to avoid timeouts on large datasets
        # Qdrant can handle large batches, but network/timeout issues may occur
        batch_size = 1000
        total_batches = (len(qdrant_points) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1

            if len(qdrant_points) > 1000:  # Show progress for large upserts
                print(f"    Storing batch {batch_num}/{total_batches} ({len(batch)} vectors)...")

            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True  # Wait for write to complete before continuing
            )

    def delete_by_file(self, collection: str, file_path: str):
        """Delete all points for a specific file"""
        collection_name = self._collection_name(collection)

        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path)
                    )
                ]
            )
        )

    def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        collection_name = self._collection_name(collection)

        # Build filter
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range filter
                    if "$gte" in value or "$lte" in value or "$gt" in value or "$lt" in value:
                        range_filter = {}
                        if "$gte" in value:
                            range_filter["gte"] = value["$gte"]
                        if "$lte" in value:
                            range_filter["lte"] = value["$lte"]
                        if "$gt" in value:
                            range_filter["gt"] = value["$gt"]
                        if "$lt" in value:
                            range_filter["lt"] = value["$lt"]
                        must_conditions.append(
                            FieldCondition(key=key, range=Range(**range_filter))
                        )
                else:
                    # Exact match
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "metadata": result.payload
            }
            for result in results
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {"project_id": self.project_id, "collections": {}}

        for collection in self.ALL_COLLECTIONS:
            collection_name = self._collection_name(collection)
            try:
                info = self.client.get_collection(collection_name)
                stats["collections"][collection] = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                }
            except Exception:
                # Collection doesn't exist yet
                stats["collections"][collection] = {
                    "points_count": 0,
                    "vectors_count": 0,
                }

        return stats
