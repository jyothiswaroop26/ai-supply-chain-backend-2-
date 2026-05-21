"""
Document Loader for RAG (Retrieval Augmented Generation).

This module provides utilities for loading documents from various sources
and formats for use in RAG pipelines.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)


class Document:
    """
    Represents a loaded document with metadata.
    """
    
    def __init__(self, content: str, source: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Document.
        
        Args:
            content: The text content of the document
            source: The source file path or identifier
            metadata: Optional metadata dictionary (filename, type, etc.)
        """
        self.content = content
        self.source = source
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Document(source='{self.source}', content_length={len(self.content)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata
        }


class DocumentLoader:
    """
    Loader for documents from various file formats.
    
    Supports loading from:
    - Plain text files (.txt)
    - PDF files (.pdf)
    - JSON files (.json)
    - Directories containing multiple documents
    """
    
    SUPPORTED_FORMATS = {'.txt', '.pdf', '.json'}
    
    def __init__(self, documents_dir: Optional[str] = None):
        """
        Initialize DocumentLoader.
        
        Args:
            documents_dir: Path to directory containing documents.
                         Defaults to ./data/documents/
        """
        if documents_dir is None:
            documents_dir = str(Path(__file__).parent.parent.parent / "data" / "documents")
        
        self.documents_dir = Path(documents_dir)
        
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory does not exist: {self.documents_dir}")
        else:
            logger.info(f"DocumentLoader initialized. Documents directory: {self.documents_dir}")
    
    def load_text(self, file_path: str) -> Document:
        """
        Load a plain text document.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Document object with loaded content
        """
        file_path_obj = Path(file_path)
        
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "filename": file_path_obj.name,
                "type": "text",
                "path": str(file_path_obj)
            }
            
            logger.info(f"Loaded text document from {file_path_obj}")
            return Document(content=content, source=str(file_path_obj), metadata=metadata)
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path_obj}")
            raise
        except Exception as e:
            logger.error(f"Error loading text file {file_path_obj}: {str(e)}")
            raise
    
    def load_pdf(self, file_path: str) -> Document:
        """
        Load a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document object with extracted text
            
        Note:
            Requires PyPDF2 or pdfplumber. Falls back to placeholder if unavailable.
        """
        file_path_obj = Path(file_path)
        
        try:
            # Try to import PDF libraries
            try:
                import pdfplumber  # type: ignore
                with pdfplumber.open(file_path_obj) as pdf:
                    content = ""
                    for page in pdf.pages:
                        content += page.extract_text() + "\n"
            except ImportError:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                    with open(file_path_obj, 'rb') as f:
                        pdf_reader = PdfReader(f)
                        content = ""
                        for page in pdf_reader.pages:
                            content += page.extract_text() + "\n"
                except ImportError:
                    logger.warning(f"PDF libraries not available. Using placeholder for {file_path_obj.name}")
                    content = f"[PDF Content Placeholder] - {file_path_obj.name}"
            
            metadata = {
                "filename": file_path_obj.name,
                "type": "pdf",
                "path": str(file_path_obj)
            }
            
            logger.info(f"Loaded PDF document from {file_path_obj}")
            return Document(content=content.strip(), source=str(file_path_obj), metadata=metadata)
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path_obj}")
            raise
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path_obj}: {str(e)}")
            raise
    
    def load_json(self, file_path: str) -> Document:
        """
        Load a JSON document.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Document object with JSON content as formatted string
        """
        file_path_obj = Path(file_path)
        
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            content = json.dumps(data, indent=2)
            
            metadata = {
                "filename": file_path_obj.name,
                "type": "json",
                "path": str(file_path_obj)
            }
            
            logger.info(f"Loaded JSON document from {file_path_obj}")
            return Document(content=content, source=str(file_path_obj), metadata=metadata)
        
        except FileNotFoundError:
            logger.error(f"File not found: {file_path_obj}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path_obj}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path_obj}: {str(e)}")
            raise
    
    def load_file(self, file_path: str) -> Document:
        """
        Load a document from any supported file format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path_obj = Path(file_path)
        suffix = file_path_obj.suffix.lower()
        
        if suffix == '.txt':
            return self.load_text(str(file_path))
        elif suffix == '.pdf':
            return self.load_pdf(str(file_path))
        elif suffix == '.json':
            return self.load_json(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: {self.SUPPORTED_FORMATS}")
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load multiple documents.
        
        Args:
            file_paths: List of file paths to load. If None, loads all documents
                       from the documents directory.
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if file_paths is None:
            # Load all documents from directory
            if not self.documents_dir.exists():
                logger.warning(f"Documents directory does not exist: {self.documents_dir}")
                return documents
            
            file_paths = [
                str(f) for f in self.documents_dir.glob('*')
                if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
            ]
            
            logger.info(f"Found {len(file_paths)} documents to load from {self.documents_dir}")
        
        for file_path in file_paths:
            try:
                doc = self.load_file(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load document {file_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def load_from_directory(self, directory: Optional[str] = None, 
                           recursive: bool = False) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory: Path to directory. If None, uses documents_dir.
            recursive: Whether to recursively load from subdirectories.
            
        Returns:
            List of Document objects
        """
        if directory is None:
            dir_path = self.documents_dir
        else:
            dir_path = Path(directory)
        
        documents = []
        
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return documents
        
        # Use glob pattern for recursive search if needed
        pattern = '**/*' if recursive else '*'
        
        file_paths = [
            f for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        logger.info(f"Found {len(file_paths)} documents in {dir_path} (recursive={recursive})")
        
        for file_path in file_paths:
            try:
                doc = self.load_file(str(file_path))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load document {file_path}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(documents)} documents from directory")
        return documents


def load_all_documents(documents_dir: Optional[str] = None) -> List[Document]:
    """
    Convenience function to load all documents from a directory.
    
    Args:
        documents_dir: Path to documents directory. Defaults to ./data/documents/
        
    Returns:
        List of Document objects
    """
    loader = DocumentLoader(documents_dir)
    return loader.load_documents()


def load_document(file_path: str) -> Document:
    """
    Convenience function to load a single document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Document object
    """
    loader = DocumentLoader()
    return loader.load_file(file_path)
