"""
RAG Pipeline Usage Examples

This module demonstrates how to use the RAG pipeline with different LLM providers.
"""

from backend.src.rag.rag_pipeline import (
    create_rag_pipeline,
    RAGPipeline,
    LocalLLMProvider,
)
from backend.src.rag.loader import Document


def example_basic_usage():
    """Basic usage with local LLM provider (for testing)."""
    # Create pipeline with local LLM
    pipeline = create_rag_pipeline(llm_provider_type="local")
    
    # Load documents from directory
    pipeline.load_documents("./data/documents")
    
    # Query
    result = pipeline.query("What is the current inventory status?")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Documents retrieved: {result['num_retrieved']}")


def example_with_custom_documents():
    """Usage with custom documents."""
    # Create documents
    docs = [
        Document(
            content="Inventory: 500 units in stock. Reorder point: 100 units.",
            source="inventory_report.txt",
            metadata={"date": "2024-06-04"}
        ),
        Document(
            content="Shipment delayed due to weather. ETA: 2 days.",
            source="shipment_log.txt",
            metadata={"type": "shipment"}
        ),
    ]
    
    # Create pipeline
    pipeline = create_rag_pipeline(llm_provider_type="local")
    
    # Load custom documents
    pipeline.load_documents_from_list(docs)
    
    # Query
    result = pipeline.query("When will shipment arrive?")
    print(f"Answer: {result['answer']}")


def example_with_openai():
    """Usage with OpenAI provider (requires API key)."""
    # Create pipeline with OpenAI
    pipeline = create_rag_pipeline(
        llm_provider_type="openai",
        llm_config={"model": "gpt-3.5-turbo"}
    )
    
    # Load documents
    pipeline.load_documents("./data/documents")
    
    # Query
    result = pipeline.query(
        "What are the key supply chain risks?",
        num_retrieval=5,
        max_tokens=500,
        temperature=0.7
    )
    print(f"Answer: {result['answer']}")


def example_batch_queries():
    """Batch query example."""
    pipeline = create_rag_pipeline(llm_provider_type="local")
    pipeline.load_documents("./data/documents")
    
    queries = [
        "What is inventory status?",
        "When is next shipment?",
        "Any supply chain issues?",
    ]
    
    results = pipeline.batch_query(queries)
    for result in results:
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}\n")


if __name__ == "__main__":
    print("RAG Pipeline Examples\n")
    
    # Try basic example
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Example 1 error: {e}\n")
    
    # Try custom documents example
    try:
        example_with_custom_documents()
    except Exception as e:
        print(f"Example 2 error: {e}\n")
