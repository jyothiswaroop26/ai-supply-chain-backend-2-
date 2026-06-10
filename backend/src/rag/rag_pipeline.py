"""
RAG (Retrieval-Augmented Generation) Pipeline.

This module combines semantic retrieval with language model generation
to create a complete RAG pipeline for question-answering over documents.

Components:
- Retriever: Semantic search over indexed documents
- LLM: Language model for generating responses based on retrieved context
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .loader import Document, DocumentLoader
from .retriever import SemanticRetriever

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum number of tokens in response
            temperature: Controls randomness (0-1, higher = more random)

        Returns:
            Generated response text
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT model provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env variable
            model: Model name (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai library required. Install: pip install openai")

        self.model = model
        if api_key:
            openai.api_key = api_key
        logger.info(f"Initialized OpenAI provider with model: {model}")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai library required. Install: pip install openai")

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class HuggingFaceProvider(LLMProvider):
    """Hugging Face transformers-based LLM provider."""

    def __init__(self, model_name: str = "gpt2", use_gpu: bool = False):
        """
        Initialize Hugging Face provider.

        Args:
            model_name: Hugging Face model name
            use_gpu: Whether to use GPU acceleration
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers library required. Install: pip install transformers")

        device = 0 if use_gpu else -1
        self.model_name = model_name
        self.pipeline = pipeline("text-generation", model=model_name, device=device)
        logger.info(f"Initialized Hugging Face provider with model: {model_name}")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using Hugging Face model."""
        response = self.pipeline(prompt, max_length=max_tokens, temperature=temperature)
        return response[0]["generated_text"].strip()


class LocalLLMProvider(LLMProvider):
    """Simple local LLM provider (echo-based for testing)."""

    def __init__(self):
        """Initialize local LLM provider."""
        logger.info("Initialized LocalLLMProvider (echo-based for testing)")

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Simple echo-based response for testing.
        
        In production, replace with actual local model integration.
        """
        return f"[LocalLLM Response]\n{prompt[:max_tokens]}"


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.

    Workflow:
    1. Load and index documents using semantic retriever
    2. On query: retrieve relevant documents
    3. Build prompt with retrieved context
    4. Generate response using LLM
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        retriever: Optional[SemanticRetriever] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
    ):
        """
        Initialize RAG pipeline.

        Args:
            llm_provider: LLM provider instance
            retriever: Optional pre-built retriever. If None, creates new one.
            embedding_model: Embedding model name for semantic retrieval
            use_gpu: Whether to use GPU for embeddings
        """
        self.llm_provider = llm_provider
        self.retriever = retriever or SemanticRetriever(
            embedding_model=embedding_model,
            use_gpu=use_gpu,
        )
        self.is_ready = False
        logger.info("Initialized RAG pipeline")

    def load_documents(self, documents_dir: Optional[str] = None) -> None:
        """
        Load documents from a directory and build retriever index.

        Args:
            documents_dir: Path to directory with documents. 
                          Defaults to ./data/documents
        """
        loader = DocumentLoader(documents_dir)
        documents = loader.load_documents()

        if not documents:
            raise ValueError(f"No documents found in {loader.documents_dir}")

        self.retriever.build_index(documents)
        self.is_ready = True
        logger.info(f"RAG pipeline ready. Indexed {len(documents)} documents")

    def load_documents_from_list(self, documents: List[Document]) -> None:
        """
        Load documents from a list and build retriever index.

        Args:
            documents: List of Document objects
        """
        if not documents:
            raise ValueError("Cannot load empty document list")

        self.retriever.build_index(documents)
        self.is_ready = True
        logger.info(f"RAG pipeline ready. Indexed {len(documents)} documents")

    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Build a prompt combining query and retrieved context.

        Args:
            query: User query
            context_docs: Retrieved documents with metadata

        Returns:
            Combined prompt string
        """
        context_text = "\n---\n".join(
            [f"Source: {doc.get('source', 'Unknown')}\n{doc.get('content', '')}" 
             for doc in context_docs]
        )

        prompt = f"""You are a helpful supply chain assistant. Use the following context to answer the question.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
        return prompt

    def query(
        self,
        query: str,
        num_retrieval: int = 5,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Execute RAG pipeline: retrieve documents and generate response.

        Args:
            query: User query/question
            num_retrieval: Number of documents to retrieve
            max_tokens: Maximum tokens in LLM response
            temperature: LLM temperature parameter

        Returns:
            Dictionary with query, retrieved docs, and generated answer
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Call load_documents() first.")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.search(query, k=num_retrieval)
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query")

        # Step 2: Build prompt with context
        prompt = self._build_prompt(query, retrieved_docs)

        # Step 3: Generate response
        response = self.llm_provider.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "answer": response,
            "num_retrieved": len(retrieved_docs),
        }

    def batch_query(
        self,
        queries: List[str],
        num_retrieval: int = 5,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Execute RAG pipeline for multiple queries.

        Args:
            queries: List of queries
            num_retrieval: Number of documents to retrieve per query
            max_tokens: Maximum tokens in LLM response
            temperature: LLM temperature parameter

        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.query(
                query,
                num_retrieval=num_retrieval,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)
        return results


def create_rag_pipeline(
    llm_provider_type: str = "openai",
    llm_config: Optional[Dict[str, Any]] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    use_gpu: bool = False,
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.

    Args:
        llm_provider_type: Type of LLM provider ('openai', 'huggingface', 'local')
        llm_config: Configuration dict for LLM provider
        embedding_model: Embedding model name
        use_gpu: Whether to use GPU

    Returns:
        Initialized RAGPipeline instance

    Example:
        >>> pipeline = create_rag_pipeline(
        ...     llm_provider_type="openai",
        ...     llm_config={"model": "gpt-3.5-turbo"}
        ... )
        >>> pipeline.load_documents("./data/documents")
        >>> result = pipeline.query("What is inventory status?")
    """
    llm_config = llm_config or {}

    if llm_provider_type.lower() == "openai":
        provider = OpenAIProvider(**llm_config)
    elif llm_provider_type.lower() == "huggingface":
        provider = HuggingFaceProvider(**llm_config)
    elif llm_provider_type.lower() == "local":
        provider = LocalLLMProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider_type}")

    retriever = SemanticRetriever(
        embedding_model=embedding_model,
        use_gpu=use_gpu,
    )

    pipeline = RAGPipeline(
        llm_provider=provider,
        retriever=retriever,
    )

    logger.info(f"Created RAG pipeline with {llm_provider_type} provider")
    return pipeline
