"""Abstract interface for Document Intelligence services

This interface defines the contract that all DI service implementations must follow.
Different providers (Google File Search, OpenAI, Milvus, etc.) implement this interface.

The key design principle:
- Documents app only knows about document IDs
- DI service queries the database to get all needed information
- DI service manages its own metadata and storage
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


class DocumentIntelligenceService(ABC):
    """Abstract base class for Document Intelligence services

    Implementations handle their own database access and metadata management.
    The documents app only needs to pass document/session IDs.
    """

    @abstractmethod
    def create_store_for_session(self, session_id: str) -> None:
        """Create a document intelligence store for a session

        The DI service:
        1. Queries DB to get session details (PO number, vendor ID, etc.)
        2. Creates a store/collection in the DI system
        3. Saves the store identifier back to session metadata

        Args:
            session_id: UUID of the session (as string)

        Raises:
            ValueError: If session not found
            Exception: If store creation fails

        Example:
            di_service.create_store_for_session("550e8400-e29b-41d4-a716-446655440000")
        """
        pass

    @abstractmethod
    def index_document(self, document_id: str) -> None:
        """Index a document in the DI system

        The DI service:
        1. Queries DB to get document details (file_path, session_id, filename, etc.)
        2. Gets session to find the DI store identifier
        3. Retrieves the document from S3 (or other storage)
        4. Processes and indexes the document
        5. Saves DI metadata back to the document record

        Args:
            document_id: UUID of the document (as string)

        Raises:
            ValueError: If document or session not found
            Exception: If indexing fails

        Example:
            di_service.index_document("650e8400-e29b-41d4-a716-446655440001")
        """
        pass

    @abstractmethod
    def query(self, session_id: str, question: str) -> tuple[str, dict]:
        """Query documents in a session

        The DI service:
        1. Queries DB to get session
        2. Gets DI store identifier from session metadata
        3. Queries the DI system with the question
        4. Returns answer and citations

        Args:
            session_id: UUID of the session (as string)
            question: User's question to answer from documents

        Returns:
            Tuple of (answer, citations_metadata)
            - answer: String response to the question
            - citations_metadata: Dict with citation/grounding information

        Raises:
            ValueError: If session not found or no DI store exists
            Exception: If query fails

        Example:
            answer, citations = di_service.query(
                session_id="550e8400-e29b-41d4-a716-446655440000",
                question="What is the delivery date?"
            )
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove a document from the DI system

        The DI service:
        1. Queries DB to get document metadata
        2. Gets the DI document identifier
        3. Deletes from the DI system

        Args:
            document_id: UUID of the document (as string)

        Note:
            This doesn't delete from S3 or database, only from DI system
        """
        pass

    @abstractmethod
    def delete_store(self, session_id: str) -> None:
        """Delete a DI store and all its documents

        The DI service:
        1. Queries DB to get session metadata
        2. Gets the DI store identifier
        3. Deletes the entire store from DI system

        Args:
            session_id: UUID of the session (as string)

        Note:
            This doesn't delete documents from S3 or database
        """
        pass


class DocumentIntelligenceServiceError(Exception):
    """Base exception for Document Intelligence service errors"""
    pass


class StoreNotFoundError(DocumentIntelligenceServiceError):
    """Raised when a DI store is not found for a session"""
    pass


class DocumentNotFoundError(DocumentIntelligenceServiceError):
    """Raised when a document is not found in the DI system"""
    pass


class IndexingError(DocumentIntelligenceServiceError):
    """Raised when document indexing fails"""
    pass


class QueryError(DocumentIntelligenceServiceError):
    """Raised when querying the DI system fails"""
    pass


# Indexing Workflow Interface

@dataclass
class TextBlock:
    """Represents an extracted text block with metadata"""
    text: str
    page_number: int
    filename: str
    block_type: str  # e.g., "paragraph", "table", "header"
    metadata: dict  # Additional metadata like position, font, etc.


@dataclass
class Chunk:
    """Represents a text chunk with enriched metadata"""
    text: str
    embedding: Optional[list[float]] = None
    metadata: Optional[dict] = None  # Includes page_number, filename, block_type, chunk_index, etc.

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentIndexingService(ABC):
    """Abstract interface for document indexing workflow

    This service handles the complete pipeline:
    1. Document download from URL/S3
    2. Text extraction with metadata
    3. Chunking with metadata preservation
    4. Embedding generation
    5. Vector database storage
    """

    @abstractmethod
    def download_document(self, url_or_path: str, destination: Optional[str] = None) -> str:
        """Download document from URL (S3) or copy local file to destination

        For S3 URLs, downloads to destination.
        For local paths, copies file to destination if provided.

        Args:
            url_or_path: S3 URL or local file path
            destination: Local destination path (required)

        Returns:
            file_path: Local file path to the downloaded/copied document

        Raises:
            DownloadError: If download fails or file doesn't exist
            ValueError: If destination is not provided

        Example:
            path = service.download_document("s3://bucket/doc.pdf", "/tmp/doc.pdf")
        """
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> list[TextBlock]:
        """Extract text from document with metadata

        Args:
            file_path: Path to the document file

        Returns:
            List of TextBlock objects with text and metadata

        Raises:
            Exception: If extraction fails

        Example:
            blocks = service.extract_text("/tmp/document.pdf")
        """
        pass

    @abstractmethod
    def chunk_text(self, text_blocks: list[TextBlock], strategy: str = "hybrid") -> list[Chunk]:
        """Chunk text blocks using specified strategy

        Args:
            text_blocks: List of TextBlock objects from extraction
            strategy: Chunking strategy ("hybrid", "fixed", "semantic")

        Returns:
            List of Chunk objects with metadata preserved and enriched

        Example:
            chunks = service.chunk_text(blocks, strategy="hybrid")
        """
        pass

    @abstractmethod
    def generate_embeddings(self, chunks: list[Chunk]) -> list[Chunk]:
        """Generate embeddings for chunks

        Args:
            chunks: List of Chunk objects

        Returns:
            List of Chunk objects with embeddings populated

        Example:
            chunks_with_embeddings = service.generate_embeddings(chunks)
        """
        pass

    @abstractmethod
    def store_in_vectordb(self, chunks: list[Chunk], collection_name: str) -> None:
        """Store chunks with embeddings in vector database

        Args:
            chunks: List of Chunk objects with embeddings
            collection_name: Name of the collection/index to store in

        Raises:
            Exception: If storage fails

        Example:
            service.store_in_vectordb(chunks, "session_123")
        """
        pass

    @abstractmethod
    def index_document_workflow(
        self,
        url_or_path: str,
        collection_name: Optional[str] = None,
        process_id: Optional[str] = None,
        chunking_strategy: str = "hybrid",
        cleanup_temp: bool = True
    ) -> str:
        """Complete workflow to index a document

        Executes the full pipeline:
        1. Download document (if URL)
        2. Extract text with metadata
        3. Chunk text
        4. Generate embeddings
        5. Store in vector database
        6. Cleanup temporary files (if downloaded)

        Args:
            url_or_path: S3 URL or local file path
            collection_name: Optional collection name (auto-generated if not provided)
            process_id: Optional UUID for tracking (auto-generated if not provided)
            chunking_strategy: Strategy for chunking ("hybrid", "fixed", "semantic")
            cleanup_temp: Whether to cleanup downloaded temp files (default: True)

        Returns:
            process_id (UUID) for tracking the indexing process

        Raises:
            Exception: If any step in the workflow fails

        Example:
            process_id = service.index_document_workflow(
                "s3://bucket/document.pdf",
                collection_name="po_123",
                process_id="550e8400-e29b-41d4-a716-446655440000",
                cleanup_temp=True
            )
        """
        pass


class IndexingServiceError(Exception):
    """Base exception for indexing service errors"""
    pass


class DownloadError(IndexingServiceError):
    """Raised when document download fails"""
    pass


class ExtractionError(IndexingServiceError):
    """Raised when text extraction fails"""
    pass


class ChunkingError(IndexingServiceError):
    """Raised when chunking fails"""
    pass


class EmbeddingError(IndexingServiceError):
    """Raised when embedding generation fails"""
    pass


class VectorDBError(IndexingServiceError):
    """Raised when vector database storage fails"""
    pass


# Query Workflow Interface

@dataclass
class Citation:
    """Represents a citation linking answer text to source chunks"""
    answer_text: str  # The part of the answer being cited
    chunk_text: str  # The source chunk text
    chunk_id: int  # Chunk identifier
    filename: str  # Source filename
    page_number: int  # Page number in source document
    confidence: float  # Confidence score (0-1)
    metadata: Optional[dict] = None  # Additional metadata


@dataclass
class QueryResponse:
    """Response structure for query operations"""
    query: str  # Original query
    answer: str  # Generated answer
    citations: list[Citation]  # List of citations
    is_good_answer: bool  # Quality check result
    confidence_score: float  # Overall confidence (0-1)
    retrieved_chunks: list[dict]  # Raw chunks retrieved from vector DB
    metadata: Optional[dict] = None  # Additional metadata (processing time, etc.)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentQueryService(ABC):
    """Abstract interface for document querying workflow

    This service handles the complete query pipeline:
    1. Query preprocessing and prompt generation
    2. Vector database search
    3. Answer generation from retrieved chunks
    4. Answer quality validation
    5. Citation extraction
    """

    @abstractmethod
    def generate_search_prompts(self, query: str) -> list[str]:
        """Generate relevant search prompts from user query

        Args:
            query: User's natural language query

        Returns:
            List of search prompts optimized for vector DB retrieval

        Example:
            prompts = service.generate_search_prompts("What is the delivery date?")
            # Returns: ["delivery date", "shipping schedule", "arrival timeline"]
        """
        pass

    @abstractmethod
    def search_vectordb(self, prompts: list[str], collection_name: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant chunks from vector database

        Args:
            prompts: List of search prompts
            collection_name: Name of the vector DB collection
            top_k: Number of top results to retrieve per prompt

        Returns:
            List of retrieved chunks with metadata

        Raises:
            VectorDBError: If search fails

        Example:
            chunks = service.search_vectordb(
                prompts=["delivery date"],
                collection_name="po_123",
                top_k=5
            )
        """
        pass

    @abstractmethod
    def generate_answer(self, query: str, chunks: list[dict]) -> str:
        """Generate answer using retrieved chunks and query

        Args:
            query: User's original query
            chunks: Retrieved chunks from vector database

        Returns:
            Generated answer text

        Raises:
            Exception: If answer generation fails

        Example:
            answer = service.generate_answer(
                query="What is the delivery date?",
                chunks=[{...}, {...}]
            )
        """
        pass

    @abstractmethod
    def validate_answer(self, query: str, answer: str, chunks: list[dict]) -> bool:
        """Check if the generated answer is good quality

        Args:
            query: Original query
            answer: Generated answer
            chunks: Source chunks used for generation

        Returns:
            True if answer meets quality criteria, False otherwise

        Example:
            is_good = service.validate_answer(query, answer, chunks)
        """
        pass

    @abstractmethod
    def extract_citations(self, answer: str, chunks: list[dict]) -> list[Citation]:
        """Extract citations mapping answer segments to source chunks

        Args:
            answer: Generated answer text
            chunks: Source chunks used for answer generation

        Returns:
            List of Citation objects linking answer to sources

        Example:
            citations = service.extract_citations(answer, chunks)
        """
        pass

    @abstractmethod
    def query_workflow(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5
    ) -> QueryResponse:
        """Complete workflow to query documents

        Executes the full pipeline:
        1. Generate search prompts from query
        2. Search vector database
        3. Generate answer from chunks
        4. Validate answer quality
        5. Extract citations

        Args:
            query: User's natural language query
            collection_name: Vector DB collection to search
            top_k: Number of top results to retrieve

        Returns:
            QueryResponse object with answer and citations

        Raises:
            VectorDBError: If vector DB operations fail
            Exception: If any step in the workflow fails

        Example:
            response = service.query_workflow(
                query="What is the delivery date?",
                collection_name="po_123",
                top_k=5
            )
            print(response.answer)
            for citation in response.citations:
                print(f"Source: {citation.filename}, Page: {citation.page_number}")
        """
        pass


class QueryServiceError(Exception):
    """Base exception for query service errors"""
    pass


class PromptGenerationError(QueryServiceError):
    """Raised when prompt generation fails"""
    pass


class AnswerGenerationError(QueryServiceError):
    """Raised when answer generation fails"""
    pass


class CitationExtractionError(QueryServiceError):
    """Raised when citation extraction fails"""
    pass


# Vector Database Interface

class VectorDBService(ABC):
    """Abstract interface for vector database operations

    This interface defines operations for vector storage and retrieval.
    Different implementations (Milvus, Pinecone, Weaviate, etc.) implement this interface.
    """

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """Create a new collection in the vector database

        Args:
            collection_name: Name of the collection to create

        Raises:
            VectorDBError: If collection creation fails

        Note:
            If collection already exists, it should be dropped and recreated
        """
        pass

    @abstractmethod
    def insert(self, collection_name: str, data: list[dict]) -> None:
        """Insert vectors and metadata into a collection

        Args:
            collection_name: Name of the collection
            data: List of records to insert, each containing:
                - id: Record identifier
                - vector: Embedding vector
                - Additional metadata fields (text, filename, etc.)

        Raises:
            VectorDBError: If insertion fails

        Example:
            data = [
                {
                    'id': 0,
                    'vector': [0.1, 0.2, ...],
                    'text': 'chunk text',
                    'filename': 'doc.pdf',
                    'page_number': 1
                }
            ]
            service.insert('collection_name', data)
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vectors: list[list[float]],
        limit: int = 5,
        output_fields: list[str] | None = None
    ) -> list[dict]:
        """Search for similar vectors in a collection

        Args:
            collection_name: Name of the collection to search
            query_vectors: List of query embedding vectors
            limit: Maximum number of results to return per query
            output_fields: List of fields to include in results

        Returns:
            List of search results with similarity scores and metadata

        Raises:
            VectorDBError: If search fails

        Example:
            results = service.search(
                collection_name='po_123',
                query_vectors=[[0.1, 0.2, ...]],
                limit=5,
                output_fields=['text', 'filename', 'page_number']
            )
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the vector database

        Args:
            collection_name: Name of the collection to delete

        Raises:
            VectorDBError: If deletion fails
        """
        pass

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        pass
