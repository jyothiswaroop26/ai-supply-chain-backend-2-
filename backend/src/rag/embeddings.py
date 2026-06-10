"""
Embeddings Generation for RAG (Retrieval Augmented Generation).

This module provides utilities for generating embeddings from documents
for semantic search and retrieval in RAG pipelines.

Supports multiple embedding methods:
- Sentence Transformers (neural embeddings)
- TF-IDF (statistical embeddings as fallback)
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for documents and queries using various methods.
    
    Supports:
    - Sentence Transformers for semantic embeddings
    - TF-IDF for statistical embeddings (fallback)
    - Caching of embeddings to disk
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            model_name: Name of the sentence transformer model or 'tfidf' for TF-IDF.
                       Default: 'all-MiniLM-L6-v2' (384-dim, fast, lightweight)
            use_gpu: Whether to use GPU if available (only for sentence transformers)
            
        Raises:
            ValueError: If model initialization fails
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.embedding_dim = None
        self.embedding_method = None
        self.vectorizer = None  # For TF-IDF
        self.vocabulary = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the embedding model."""
        try:
            if self.model_name.lower() == "tfidf":
                self._initialize_tfidf()
            else:
                self._initialize_sentence_transformer()
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            logger.info("Falling back to TF-IDF embeddings")
            self._initialize_tfidf()
    
    def _initialize_sentence_transformer(self) -> None:
        """Initialize sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            
            device = "cuda" if self.use_gpu else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.embedding_method = "sentence_transformer"
            
            logger.info(
                f"Initialized SentenceTransformer model: {self.model_name} "
                f"(dimension: {self.embedding_dim}, device: {device})"
            )
        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to TF-IDF.")
            self._initialize_tfidf()
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {str(e)}")
            self._initialize_tfidf()
    
    def _initialize_tfidf(self) -> None:
        """Initialize TF-IDF vectorizer as fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\w{1,}',
                ngram_range=(1, 2),
                lowercase=True
            )
            self.embedding_method = "tfidf"
            
            logger.info("Initialized TF-IDF vectorizer as embedding method")
        except ImportError:
            logger.error("scikit-learn not installed. Cannot initialize TF-IDF.")
            raise
    
    def embed_documents(self, documents: List[Any]) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of Document objects or strings
            
        Returns:
            NumPy array of shape (n_documents, embedding_dim)
        """
        if not documents:
            logger.warning("Empty document list provided")
            return np.array([])
        
        # Extract text content
        texts = []
        for doc in documents:
            if hasattr(doc, 'content'):
                texts.append(doc.content)
            elif isinstance(doc, str):
                texts.append(doc)
            else:
                logger.warning(f"Unexpected document type: {type(doc)}")
                texts.append(str(doc))
        
        if self.embedding_method == "sentence_transformer":
            return self._embed_with_sentence_transformer(texts)
        else:
            return self._embed_with_tfidf(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            NumPy array of shape (embedding_dim,) or (1, embedding_dim)
        """
        if self.embedding_method == "sentence_transformer":
            return self._embed_query_sentence_transformer(query)
        else:
            return self._embed_query_tfidf(query)
    
    def _embed_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Embed texts using sentence transformers."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            if len(texts) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            logger.info(f"Generated embeddings for {len(texts)} documents using SentenceTransformer")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding with SentenceTransformer: {str(e)}")
            raise
    
    def _embed_query_sentence_transformer(self, query: str) -> np.ndarray:
        """Embed single query using sentence transformers."""
        try:
            embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Error embedding query with SentenceTransformer: {str(e)}")
            raise
    
    def _embed_with_tfidf(self, texts: List[str]) -> np.ndarray:
        """Embed texts using TF-IDF vectorizer."""
        try:
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            self.embedding_dim = embeddings.shape[1]
            
            logger.info(f"Generated embeddings for {len(texts)} documents using TF-IDF")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding with TF-IDF: {str(e)}")
            raise
    
    def _embed_query_tfidf(self, query: str) -> np.ndarray:
        """Embed single query using TF-IDF vectorizer."""
        try:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not initialized")
            
            embedding = self.vectorizer.transform([query]).toarray().flatten()
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query with TF-IDF: {str(e)}")
            raise
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity_score = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(np.clip(similarity_score, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def batch_similarity(self, query_embedding: np.ndarray, 
                        document_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query and multiple documents.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            document_embeddings: Document embeddings of shape (n_docs, embedding_dim)
            
        Returns:
            Array of similarity scores of shape (n_docs,)
        """
        try:
            # Normalize
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            
            if query_norm == 0:
                return np.zeros(len(document_embeddings))
            
            # Compute cosine similarity
            similarities = np.dot(document_embeddings, query_embedding) / (doc_norms.flatten() * query_norm)
            
            return np.clip(similarities, -1.0, 1.0)
        except Exception as e:
            logger.error(f"Error calculating batch similarity: {str(e)}")
            return np.zeros(len(document_embeddings))
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings array to save
            file_path: Path to save the embeddings
        """
        try:
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.endswith('.npy'):
                np.save(file_path, embeddings)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(embeddings, f)
            
            logger.info(f"Saved embeddings to {file_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            file_path: Path to load embeddings from
            
        Returns:
            Loaded embeddings array
        """
        try:
            if file_path.endswith('.npy'):
                embeddings = np.load(file_path)
            else:
                with open(file_path, 'rb') as f:
                    embeddings = pickle.load(f)
            
            logger.info(f"Loaded embeddings from {file_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise


class EmbeddingStore:
    """
    Store and manage embeddings with metadata for documents.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize EmbeddingStore.
        
        Args:
            cache_dir: Directory to cache embeddings. Defaults to ./cache/embeddings/
        """
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent.parent / "cache" / "embeddings")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"EmbeddingStore initialized with cache directory: {self.cache_dir}")
    
    def store(self, doc_id: str, embedding: np.ndarray, 
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store an embedding with metadata.
        
        Args:
            doc_id: Unique identifier for the document
            embedding: Embedding vector
            metadata: Optional metadata dictionary
        """
        self.embeddings[doc_id] = embedding
        self.metadata[doc_id] = metadata or {}
        logger.debug(f"Stored embedding for document: {doc_id}")
    
    def retrieve(self, doc_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve an embedding and its metadata.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        if doc_id in self.embeddings:
            return self.embeddings[doc_id], self.metadata.get(doc_id, {})
        return None
    
    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all embeddings as a matrix.
        
        Returns:
            Array of shape (n_documents, embedding_dim)
        """
        if not self.embeddings:
            return np.array([])
        
        return np.array(list(self.embeddings.values()))
    
    def get_document_ids(self) -> List[str]:
        """Get list of all stored document IDs."""
        return list(self.embeddings.keys())
    
    def save_store(self, file_path: str) -> None:
        """
        Save the entire embedding store to disk.
        
        Args:
            file_path: Path to save the store
        """
        try:
            store_data = {
                "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
                "metadata": self.metadata
            }
            
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(store_data, f, indent=2)
            
            logger.info(f"Saved embedding store to {file_path}")
        except Exception as e:
            logger.error(f"Error saving embedding store: {str(e)}")
            raise
    
    def load_store(self, file_path: str) -> None:
        """
        Load an embedding store from disk.
        
        Args:
            file_path: Path to load the store from
        """
        try:
            with open(file_path, 'r') as f:
                store_data = json.load(f)
            
            self.embeddings = {
                k: np.array(v) for k, v in store_data.get("embeddings", {}).items()
            }
            self.metadata = store_data.get("metadata", {})
            
            logger.info(f"Loaded embedding store from {file_path} ({len(self.embeddings)} embeddings)")
        except Exception as e:
            logger.error(f"Error loading embedding store: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear all stored embeddings and metadata."""
        self.embeddings.clear()
        self.metadata.clear()
        logger.info("Cleared all embeddings from store")


# Convenience functions for quick usage

def generate_embeddings(documents: List[Any], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Quick function to generate embeddings for documents.
    
    Args:
        documents: List of Document objects or strings
        model_name: Embedding model to use
        
    Returns:
        Array of embeddings
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_documents(documents)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Quick function to generate embedding for a query.
    
    Args:
        query: Query text
        model_name: Embedding model to use
        
    Returns:
        Query embedding vector
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_query(query)
