"""
Semantic retriever utilities for RAG pipelines.

This module combines embedding generation and FAISS indexing to provide
top-k semantic document retrieval.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .embeddings import EmbeddingGenerator
from .loader import Document
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

_QUERY_CACHE_MAX = 128  # max number of cached query embeddings


class SemanticRetriever:
    """Build and query a semantic document index."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
        metric: str = "cosine",
    ):
        """
        Initialize semantic retriever.

        Args:
            embedding_model: Embedding model name or 'tfidf'.
            use_gpu: Whether to use GPU for supported embedding models.
            metric: Similarity metric for FAISS vector store.
        """
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            use_gpu=use_gpu,
        )
        self.vector_store = FAISSVectorStore(metric=metric)
        self.documents_by_id: Dict[str, Document] = {}
        self.is_index_built = False
        # LRU cache for query embeddings: avoids re-embedding repeated queries
        self._query_cache: OrderedDict[str, Any] = OrderedDict()

    def build_index(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Build FAISS index for a collection of documents.

        Args:
            documents: Documents to index.
            ids: Optional IDs for each document.

        Returns:
            IDs used for indexing.
        """
        if not documents:
            raise ValueError("Cannot build index with empty document list")

        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids length must match documents length")

        embeddings = self.embedding_generator.embed_documents(documents)
        resolved_ids = ids or [f"doc_{idx}" for idx in range(len(documents))]

        metadata: List[Dict[str, Any]] = []
        for doc_id, doc in zip(resolved_ids, documents):
            self.documents_by_id[doc_id] = doc
            metadata.append(
                {
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
            )

        self.vector_store.clear()
        self.vector_store.add(embeddings, ids=resolved_ids, metadata=metadata)
        self.is_index_built = True

        # Invalidate query cache whenever the index is rebuilt
        self._query_cache.clear()

        logger.info("Built semantic index for %s documents", len(documents))
        return resolved_ids

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Run semantic search against the indexed documents.

        Args:
            query: Natural-language query.
            k: Number of top results.

        Returns:
            Ranked result dictionaries containing document data and score.
        """
        if not self.is_index_built:
            raise ValueError("Index is not built. Call build_index(documents) first.")

        # Serve from cache when the same query has been seen before
        query_embedding = self._query_cache.get(query)
        if query_embedding is None:
            query_embedding = self.embedding_generator.embed_query(query)
            # Evict oldest entry if cache is full (simple LRU eviction)
            if len(self._query_cache) >= _QUERY_CACHE_MAX:
                self._query_cache.popitem(last=False)
            self._query_cache[query] = query_embedding
        else:
            # Move to end to mark as recently used
            self._query_cache.move_to_end(query)

        matches = self.vector_store.search(query_embedding, k=k)

        results: List[Dict[str, Any]] = []
        for rank, match in enumerate(matches, start=1):
            doc_id = match["id"]
            doc = self.documents_by_id.get(doc_id)

            if doc is None:
                continue

            results.append(
                {
                    "rank": rank,
                    "id": doc_id,
                    "score": match["score"],
                    "content": doc.content,
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
            )

        return results

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Run semantic search for multiple queries in a single vectorised pass.

        Embeddings for all queries are generated in one call and the FAISS index
        is searched once per query vector, but the embedding model only runs once
        for the whole batch — much faster than calling search() in a loop.

        Args:
            queries: List of natural-language queries.
            k: Number of top results per query.

        Returns:
            List of result lists, one per query (same order as input).
        """
        if not self.is_index_built:
            raise ValueError("Index is not built. Call build_index(documents) first.")

        if not queries:
            return []

        # Separate queries that are already cached from those that need embedding
        cached_embeddings: Dict[str, Any] = {}
        uncached_queries: List[str] = []
        for q in queries:
            if q in self._query_cache:
                self._query_cache.move_to_end(q)
                cached_embeddings[q] = self._query_cache[q]
            else:
                uncached_queries.append(q)

        # Embed uncached queries in a single batch call
        if uncached_queries:
            from .loader import Document as _Doc  # avoid circular at module level
            batch_embeddings = self.embedding_generator.embed_documents(
                [type('_Q', (), {'content': q, 'source': '', 'metadata': {}})() for q in uncached_queries]
            )
            for q, emb in zip(uncached_queries, batch_embeddings):
                if len(self._query_cache) >= _QUERY_CACHE_MAX:
                    self._query_cache.popitem(last=False)
                self._query_cache[q] = emb
                cached_embeddings[q] = emb

        # Search for every query and build results
        all_results: List[List[Dict[str, Any]]] = []
        for query in queries:
            matches = self.vector_store.search(cached_embeddings[query], k=k)
            results: List[Dict[str, Any]] = []
            for rank, match in enumerate(matches, start=1):
                doc = self.documents_by_id.get(match["id"])
                if doc is None:
                    continue
                results.append({
                    "rank": rank,
                    "id": match["id"],
                    "score": match["score"],
                    "content": doc.content,
                    "source": doc.source,
                    "metadata": doc.metadata,
                })
            all_results.append(results)

        logger.info("Batch search: %s queries, k=%s", len(queries), k)
        return all_results

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """
        Incrementally add documents to an existing index.

        Args:
            documents: New documents to add.
            ids: Optional document IDs.

        Returns:
            IDs assigned to added documents.
        """
        if not documents:
            return []

        if ids is not None and len(ids) != len(documents):
            raise ValueError("ids length must match documents length")

        start_idx = len(self.documents_by_id)
        resolved_ids = ids or [f"doc_{start_idx + idx}" for idx in range(len(documents))]

        embeddings = self.embedding_generator.embed_documents(documents)
        metadata: List[Dict[str, Any]] = []
        for doc_id, doc in zip(resolved_ids, documents):
            self.documents_by_id[doc_id] = doc
            metadata.append(
                {
                    "source": doc.source,
                    "metadata": doc.metadata,
                }
            )

        self.vector_store.add(embeddings, ids=resolved_ids, metadata=metadata)
        self.is_index_built = True

        logger.info("Added %s documents to semantic index", len(documents))
        return resolved_ids


def semantic_search(query: str, documents: List[Document], k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience helper for one-shot semantic search.

    Args:
        query: Search query.
        documents: Documents to search.
        k: Number of results.

    Returns:
        Semantic search results.
    """
    retriever = SemanticRetriever()
    retriever.build_index(documents)
    return retriever.search(query, k=k)
