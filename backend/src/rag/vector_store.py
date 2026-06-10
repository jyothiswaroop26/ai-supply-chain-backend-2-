"""
FAISS-backed vector store utilities for RAG retrieval.

This module wraps FAISS indexing with a small, project-friendly API for:
- Building an in-memory vector index
- Running similarity search
- Persisting/loading index data and metadata
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Simple FAISS vector store with metadata tracking."""

    def __init__(self, embedding_dim: Optional[int] = None, metric: str = "cosine"):
        """
        Initialize a FAISSVectorStore.

        Args:
            embedding_dim: Vector dimension. If None, inferred from first added vectors.
            metric: Similarity metric. Supported: 'cosine', 'l2'.
        """
        self.embedding_dim = embedding_dim
        self.metric = metric.lower().strip()
        self._faiss = self._import_faiss()

        if self.metric not in {"cosine", "l2"}:
            raise ValueError("metric must be one of: 'cosine', 'l2'")

        self.index = self._build_index(embedding_dim) if embedding_dim else None
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    @staticmethod
    def _import_faiss() -> Any:
        """Import faiss lazily with a clear installation error."""
        try:
            import faiss  # type: ignore

            return faiss
        except ImportError as exc:
            raise ImportError(
                "faiss is not installed. Install dependency: faiss-cpu"
            ) from exc

    def _build_index(self, dim: int) -> Any:
        """Create a FAISS index for the selected metric."""
        if self.metric == "cosine":
            # Cosine similarity in FAISS is done using inner product on normalized vectors.
            return self._faiss.IndexFlatIP(dim)
        return self._faiss.IndexFlatL2(dim)

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Validate vectors and convert them to float32 with expected dimensionality."""
        arr = np.asarray(vectors, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("vectors must be 2D or 1D array-like")

        if self.embedding_dim is None:
            self.embedding_dim = int(arr.shape[1])
            self.index = self._build_index(self.embedding_dim)
        elif int(arr.shape[1]) != self.embedding_dim:
            raise ValueError(
                f"vector dimension mismatch: expected {self.embedding_dim}, got {arr.shape[1]}"
            )

        if self.metric == "cosine":
            self._faiss.normalize_L2(arr)

        return arr

    def add(
        self,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.

        Args:
            vectors: Embeddings array of shape (n, d) or (d,).
            ids: Optional list of IDs, length n. Auto-generated if omitted.
            metadata: Optional list of metadata dictionaries, length n.

        Returns:
            List of IDs assigned to newly added vectors.
        """
        prepared = self._prepare_vectors(vectors)
        n_vectors = int(prepared.shape[0])

        if ids is None:
            start = len(self.ids)
            ids = [f"vec_{start + i}" for i in range(n_vectors)]
        if len(ids) != n_vectors:
            raise ValueError("ids length must match number of vectors")

        if metadata is None:
            metadata = [{} for _ in range(n_vectors)]
        if len(metadata) != n_vectors:
            raise ValueError("metadata length must match number of vectors")

        self.index.add(prepared)
        self.ids.extend(ids)
        self.metadata.extend(metadata)

        logger.info("Added %s vectors to FAISS store (size=%s)", n_vectors, len(self.ids))
        return ids

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search nearest vectors for a query vector.

        Args:
            query_vector: Query embedding of shape (d,) or (1, d).
            k: Number of results to return.

        Returns:
            List of result dictionaries with id, score, and metadata.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query = self._prepare_vectors(query_vector)
        top_k = min(max(int(k), 1), len(self.ids))
        distances, indices = self.index.search(query, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "id": self.ids[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )

        return results

    def save(self, index_path: str, meta_path: Optional[str] = None) -> None:
        """Persist FAISS index and companion metadata to disk."""
        if self.index is None:
            raise ValueError("Cannot save empty vector store")

        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(index_file))

        if meta_path is None:
            meta_file = index_file.with_suffix(index_file.suffix + ".meta.json")
        else:
            meta_file = Path(meta_path)

        meta_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "ids": self.ids,
            "metadata": self.metadata,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info("Saved FAISS index to %s and metadata to %s", index_file, meta_file)

    @classmethod
    def load(cls, index_path: str, meta_path: Optional[str] = None) -> "FAISSVectorStore":
        """Load FAISS index and companion metadata from disk."""
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")

        if meta_path is None:
            meta_file = index_file.with_suffix(index_file.suffix + ".meta.json")
        else:
            meta_file = Path(meta_path)

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(
            embedding_dim=payload.get("embedding_dim"),
            metric=payload.get("metric", "cosine"),
        )
        store.index = store._faiss.read_index(str(index_file))
        store.ids = payload.get("ids", [])
        store.metadata = payload.get("metadata", [])

        if len(store.ids) != store.index.ntotal:
            raise ValueError(
                "Loaded metadata does not match index size "
                f"(ids={len(store.ids)}, index={store.index.ntotal})"
            )

        return store

    def clear(self) -> None:
        """Reset index and remove all vectors and metadata."""
        self.index = self._build_index(self.embedding_dim) if self.embedding_dim else None
        self.ids.clear()
        self.metadata.clear()

    def __len__(self) -> int:
        return len(self.ids)
