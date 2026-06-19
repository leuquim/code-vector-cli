"""Qdrant vector store management"""

import os
import uuid
import hashlib
from typing import List, Dict, Optional, Any
from pathlib import Path

# Stable namespace so the same logical ID always maps to the same UUID.
_POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

# Load .env file if it exists (before importing qdrant_client)
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

            # Prefer gRPC: a single persistent HTTP/2 connection. The REST
            # transport opens a new TCP socket per request, which exhausts
            # Windows ephemeral ports during large batch upserts
            # (WinError 10048: "Only one usage of each socket address...").
            prefer_grpc = os.environ.get("QDRANT_PREFER_GRPC", "true").lower() == "true"
            grpc_port = int(os.environ.get("QDRANT_GRPC_PORT", 0)) or None

            # check_compatibility=False: the client/server minor-version check
            # emits a noisy UserWarning whenever the server drifts >1 minor ahead.
            # We track a compatible client in requirements; skip the runtime check.
            client_kwargs = {"host": host, "timeout": 300, "check_compatibility": False}
            if prefer_grpc:
                client_kwargs["prefer_grpc"] = True
                client_kwargs["grpc_port"] = grpc_port or (port + 1)
                client_kwargs["port"] = port
            else:
                client_kwargs["port"] = port

            self.client = QdrantClient(**client_kwargs)
            self._mode = "remote"
            self._verify_connection(host, port)

        self.project_path = project_path
        self.project_id = self._get_project_id(project_path)

    def _verify_connection(self, host: str, port: int):
        """Fail loudly if the Qdrant server is unreachable.

        Without this, every search silently degrades to "no results", which
        reads as an empty index rather than a down database.
        """
        try:
            self.client.get_collections()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Qdrant at {host}:{port}. Is the server running?\n"
                f"  Start it with: docker compose up -d\n"
                f"  Or use embedded mode: set QDRANT_LOCAL=true in ~/.code-vector-db.env\n"
                f"  (underlying error: {type(e).__name__}: {str(e)[:120]})"
            ) from e

    def _get_project_id(self, project_path: str) -> str:
        """Generate unique project ID from normalized path (cross-platform compatible)"""
        normalized = normalize_path_for_id(project_path)
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _collection_name(self, base_name: str) -> str:
        """Get project-specific collection name"""
        return f"{self.project_id}_{base_name}"

    @staticmethod
    def _to_point_id(raw_id: Any) -> Any:
        """Coerce an arbitrary id into a Qdrant-valid point id.

        Qdrant accepts only UUID strings or unsigned ints. Ints and
        already-valid UUIDs pass through; everything else (commit SHAs,
        path:line keys, md5 hexdigests) maps to a deterministic UUID5 so
        re-upserting the same logical point overwrites rather than duplicates.
        """
        if isinstance(raw_id, int):
            return raw_id
        s = str(raw_id)
        try:
            return str(uuid.UUID(s))  # already a valid UUID
        except (ValueError, AttributeError, TypeError):
            return str(uuid.uuid5(_POINT_ID_NAMESPACE, s))

    def initialize_collections(self):
        """Create all collections for the project.

        Vector sizes are derived from the active embedding provider (single
        source of truth in embeddings.py) rather than hardcoded. With the
        default zembed-1 provider, code and text share one dimension.
        """
        from code_vector_db.embeddings import (
            get_provider, get_code_dimension, get_text_dimension
        )

        provider = get_provider()
        code_vector_size = get_code_dimension()
        text_vector_size = get_text_dimension()
        print(f"  Using {provider} embeddings (code: {code_vector_size}, text: {text_vector_size} dimensions)")

        # Guard against silent corruption: if collections already exist at a
        # different dimension (e.g. provider was switched), fail loudly.
        self._check_dimension_compatibility(code_vector_size, text_vector_size, provider)

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

    def _check_dimension_compatibility(self, code_size: int, text_size: int, provider: str):
        """Fail loudly if existing collections were built at a different dimension.

        Switching embedding providers/dimensions without reindexing silently
        corrupts search (vectors of different sizes can't be compared). Detect
        the mismatch up front and tell the user exactly what to do.
        """
        checks = [
            (self.CODE_FUNCTIONS, code_size),
            (self.DOCUMENTATION, text_size),
        ]
        for base_name, expected in checks:
            name = self._collection_name(base_name)
            try:
                info = self.client.get_collection(name)
            except Exception:
                continue  # doesn't exist yet — nothing to conflict with

            try:
                existing = info.config.params.vectors.size
            except Exception:
                continue  # can't read size — skip the guard rather than crash

            if existing != expected:
                raise RuntimeError(
                    f"Embedding dimension mismatch for this project.\n"
                    f"  Existing collections were built at {existing} dimensions, "
                    f"but the current provider '{provider}' produces {expected}.\n"
                    f"  You changed embedding provider/model/dimension. Delete and reindex:\n"
                    f"    code-vector-cli delete --force\n"
                    f"    code-vector-cli index"
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
                    distance=distance,
                    on_disk=True  # f32 originals only rescore; int8 copies serve search from RAM
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                ),
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    # Unit is KB of vectors per segment, NOT a point count. Must stay
                    # below segment size or HNSW never builds and searches full-scan.
                    indexing_threshold=10000
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
            raw_id = point.get(
                "id",
                f"{point['metadata']['file_path']}:{point['metadata']['start_line']}"
            )
            # Qdrant only accepts UUID or unsigned-int point IDs. Commit SHAs and
            # path:line strings are neither, so map any string to a deterministic
            # UUID5 (stable across runs -> idempotent upserts).
            qdrant_points.append(PointStruct(
                id=self._to_point_id(raw_id),
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

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "metadata": point.payload
            }
            for point in results.points
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
                    "indexed_vectors_count": info.indexed_vectors_count,
                }
            except Exception:
                # Collection doesn't exist yet
                stats["collections"][collection] = {
                    "points_count": 0,
                    "indexed_vectors_count": 0,
                }

        return stats
