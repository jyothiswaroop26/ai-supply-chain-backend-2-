"""
RAG (Retrieval Augmented Generation) module for supply chain system.

This module provides utilities for loading documents and building
RAG pipelines for knowledge-enhanced decision making.
"""

from .loader import (
    Document,
    DocumentLoader,
    load_all_documents,
    load_document
)

__all__ = [
    'Document',
    'DocumentLoader',
    'load_all_documents',
    'load_document'
]
