"""Tests for AI Engine functionality"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

from apps.ai_engine.service import MilvusDocumentIndexingService
from apps.ai_engine.interface import DownloadError


class TestDownloadDocument:
    """Test cases for MilvusDocumentIndexingService.download_document method"""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing"""
        from apps.ai_engine.service import MilvusVectorDBService, EmbeddingService

        # Use the correct path to milvus_storage in ai_engine directory
        db_path = str(project_root / "apps" / "ai_engine" / "milvus_storage" / "test.db")

        # Initialize with test database
        vectordb = MilvusVectorDBService(db_path=db_path)
        embedding = EmbeddingService()

        return MilvusDocumentIndexingService(
            embedding_service=embedding,
            vectordb_service=vectordb
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup after test
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create a sample test file"""
        file_path = Path(temp_dir) / "sample.txt"
        file_path.write_text("This is a test file content")
        return str(file_path)

    def test_download_local_file_with_destination(self, service, sample_file, temp_dir):
        """Test copying a local file to a specified destination"""
        destination = Path(temp_dir) / "destination" / "copied_file.txt"

        result = service.download_document(sample_file, str(destination))

        assert result == str(destination)
        assert destination.exists()
        assert destination.read_text() == "This is a test file content"

    def test_download_local_file_without_destination(self, service, sample_file):
        """Test accessing a local file without providing destination"""
        result = service.download_document(sample_file)

        assert result == str(Path(sample_file).absolute())
        assert Path(result).exists()

    def test_download_nonexistent_file(self, service, temp_dir):
        """Test that DownloadError is raised for non-existent files"""
        nonexistent_path = Path(temp_dir) / "nonexistent.txt"
        destination = Path(temp_dir) / "destination.txt"

        with pytest.raises(DownloadError, match="File not found"):
            service.download_document(str(nonexistent_path), str(destination))

    def test_download_url_without_destination(self, service):
        """Test that DownloadError is raised when downloading URL without destination"""
        test_url = "s3://bucket/file.pdf"

        with pytest.raises(DownloadError, match="destination parameter is required"):
            service.download_document(test_url)

    def test_download_http_url_without_destination(self, service):
        """Test that DownloadError is raised when downloading HTTP URL without destination"""
        test_url = "https://example.com/file.pdf"

        with pytest.raises(DownloadError, match="destination parameter is required"):
            service.download_document(test_url)

    def test_download_creates_destination_directory(self, service, sample_file, temp_dir):
        """Test that destination directory is created if it doesn't exist"""
        destination = Path(temp_dir) / "nested" / "dir" / "structure" / "file.txt"

        result = service.download_document(sample_file, str(destination))

        assert result == str(destination)
        assert destination.exists()
        assert destination.parent.exists()

    def test_download_preserves_file_content(self, service, sample_file, temp_dir):
        """Test that file content is preserved during copy"""
        original_content = Path(sample_file).read_text()
        destination = Path(temp_dir) / "copied.txt"

        service.download_document(sample_file, str(destination))

        assert destination.read_text() == original_content

    def test_download_with_special_characters_in_path(self, service, temp_dir):
        """Test handling files with special characters in the name"""
        # Create a file with spaces and special chars
        source_file = Path(temp_dir) / "test file with spaces.txt"
        source_file.write_text("Test content")

        destination = Path(temp_dir) / "dest" / "copied file.txt"

        result = service.download_document(str(source_file), str(destination))

        assert Path(result).exists()
        assert Path(result).read_text() == "Test content"

    def test_download_real_file_from_url(self, service, temp_dir):
        """Test downloading a real file from HTTPS URL"""
        test_url = "https://ibm-po-conditioning.s3.ap-south-1.amazonaws.com/contracts/b3199da1-afb0-4a09-a4c9-1024c8341514/runs/1/272752e5-50db-4c51-ab73-423fd7a8c25d_Sample-2.pdf"
        destination = Path(temp_dir) / "downloaded_sample.pdf"

        result = service.download_document(test_url, str(destination))

        assert result == str(destination)
        assert destination.exists()
        assert destination.stat().st_size > 0  # File should have content
        # Verify it's a PDF by checking magic bytes
        with open(destination, 'rb') as f:
            header = f.read(4)
            assert header == b'%PDF'  # PDF files start with %PDF


def test_basic_service_initialization():
    """Test that the service can be initialized successfully"""
    service = MilvusDocumentIndexingService()
    assert service is not None
    assert service.embedding_service is not None
    assert service.vectordb_service is not None
    assert service.document_converter is not None


class TestDocumentIndexingAndQuerying:
    """Test cases for document indexing and querying with confidence scores"""

    @pytest.fixture
    def service(self):
        """Create a service instance for testing"""
        from apps.ai_engine.service import LangchainDocumentService

        # Use LangchainDocumentService which has the correct methods
        return LangchainDocumentService()

    @pytest.fixture
    def sample_pdfs(self):
        """Get paths to sample PDF files (1.1 to 1.4)"""
        samples_dir = project_root / "samples"
        return [
            str(samples_dir / "Sample_PO-1.1 Approval_BBU_payment terms.pdf"),
            str(samples_dir / "Sample_PO-1.2 (T&C for_mobilization_advance).pdf"),
            str(samples_dir / "Sample_PO-1.3 (Payment terms_PI_removed).pdf"),
            str(samples_dir / "Sample_PO-1.4 (SAP PO text).pdf"),
        ]

    @pytest.fixture
    def collection_name(self):
        """Collection name for test"""
        return "test_extraction_collection"

    def test_index_and_query_with_confidence(self, service, sample_pdfs, collection_name):
        """Test full workflow: index documents and query with confidence scoring"""
        import uuid

        print(f"\n=== Testing Document Indexing and Query with Confidence ===")

        # Step 1: Index all sample documents
        print(f"\n1. Indexing {len(sample_pdfs)} sample documents...")
        indexed_docs = []

        for i, pdf_path in enumerate(sample_pdfs):
            if not Path(pdf_path).exists():
                pytest.skip(f"Sample file not found: {pdf_path}")

            print(f"   - Indexing: {Path(pdf_path).name}")
            try:
                process_id = str(uuid.uuid4())
                result = service.index_document_workflow(
                    url_or_path=pdf_path,
                    collection_name=collection_name,
                    process_id=process_id,
                    cleanup_temp=True
                )
                indexed_docs.append({"process_id": result, "filename": Path(pdf_path).name})
                print(f"     ✓ Indexed successfully (process_id: {result})")
            except Exception as e:
                pytest.fail(f"Failed to index {pdf_path}: {str(e)}")

        assert len(indexed_docs) == len(sample_pdfs), "Not all documents were indexed"
        print(f"\n   Total documents indexed: {len(indexed_docs)}")

        # Step 2: Query documents with test questions
        print(f"\n2. Querying documents with test questions...")
        test_questions = [
            {
                "field_id": "test-field-1",
                "question": "What are the payment terms mentioned in these documents?",
                "extraction_instructions": "Extract specific payment terms, percentages, and conditions"
            },
            {
                "field_id": "test-field-2",
                "question": "What is the bank guarantee (BG) amount and validity period?",
                "extraction_instructions": "Look for BG amount, percentage, and validity dates"
            },
            {
                "field_id": "test-field-3",
                "question": "What is the mobilization advance percentage?",
                "extraction_instructions": "Find the percentage of mobilization advance mentioned"
            },
            {
                "field_id": "test-field-4",
                "question": "What is the total contract value?",
                "extraction_instructions": "Extract the total contract value or purchase order amount"
            }
        ]

        results = service.answer_field_questions(
            collection_name=collection_name,
            questions=test_questions,
            llm_model="gpt-4o",
            top_k=5
        )

        # Step 3: Verify results and confidence scores
        print(f"\n3. Verifying results and confidence scores...")
        assert len(results) == len(test_questions), "Should return result for each question"

        for i, result in enumerate(results):
            print(f"\n   Question {i+1}: {test_questions[i]['question']}")
            print(f"   Answer: {result['short_answer']}")
            print(f"   Confidence Score: {result['confidence_score']:.3f}")
            print(f"   Citations: {len(result['citations'])} document chunks")

            # Verify required fields
            assert "field_id" in result
            assert "answer" in result
            assert "confidence_score" in result
            assert "citations" in result

            # Verify confidence score is valid
            assert 0.0 <= result["confidence_score"] <= 1.0, \
                f"Confidence score must be between 0 and 1, got {result['confidence_score']}"

            # Verify citations have required fields
            for citation in result["citations"]:
                assert "doc_hash" in citation
                assert "filename" in citation
                assert "page_no" in citation
                assert "bbox" in citation
                assert "text_snippet" in citation

                # Verify bbox structure
                bbox = citation["bbox"]
                assert "l" in bbox and "t" in bbox and "r" in bbox and "b" in bbox
                print(f"      - {citation['filename']}, page {citation['page_no']}")

        print(f"\n✓ All tests passed!")
        print(f"✓ Confidence scores are properly calculated")
        print(f"✓ Citations include visual grounding metadata")


class TestProcessExtractionRun:
    """Test the complete extraction run workflow with database integration"""

    @pytest.mark.asyncio
    async def test_process_extraction_run_with_confidence(self):
        """Test process_extraction_run method that inserts into field_responses table"""
        from apps.ai_engine.service import LangchainDocumentService
        from models.base import SessionLocal
        from models.user import User  # Import User for foreign key relationship
        from models.category import Category  # Import Category for foreign key relationship
        from models.field_response import FieldResponse
        from sqlalchemy import select

        extraction_run_id = "ab8b5d39-3d5c-4b4f-8647-51cf815debc8"

        print(f"\n=== Testing process_extraction_run with Database Integration ===")
        print(f"Extraction Run ID: {extraction_run_id}")

        # Initialize service
        service = LangchainDocumentService()

        # Process the extraction run
        print("\n1. Processing extraction run (indexing + querying + DB insertion)...")
        try:
            result = await service.process_extraction_run(
                extraction_run_id=extraction_run_id,
                llm_model="gpt-4o",
                top_k=5,
                cleanup_temp=True
            )

            print(f"\n2. Extraction run completed successfully!")
            print(f"   Collection: {result['collection_name']}")
            print(f"   Documents processed: {result['document_count']}")
            print(f"   Fields processed: {result['field_count']}")
            print(f"   Succeeded: {result['succeeded_count']}")
            print(f"   Failed: {result['failed_count']}")
            print(f"   Field Response IDs: {result['field_responses']}")

            # Verify data was inserted into field_responses table
            print(f"\n3. Verifying field_responses table...")
            async with SessionLocal() as db:
                for field_response_id in result['field_responses']:
                    query_result = await db.execute(
                        select(FieldResponse).where(FieldResponse.id == field_response_id)
                    )
                    field_response = query_result.scalar_one_or_none()

                    assert field_response is not None, f"FieldResponse {field_response_id} not found in database"
                    assert field_response.status == "Processed", f"Expected status 'Processed', got '{field_response.status}'"
                    assert field_response.answer is not None, "Answer should not be None"
                    assert field_response.confidence_score is not None, "Confidence score should not be None"
                    assert 0.0 <= field_response.confidence_score <= 1.0, f"Confidence score must be between 0 and 1, got {field_response.confidence_score}"

                    print(f"   ✓ Field Response {field_response_id}:")
                    print(f"     Status: {field_response.status}")
                    print(f"     Confidence Score: {field_response.confidence_score:.3f}")
                    print(f"     Answer (first 100 chars): {field_response.answer[:100]}...")

                    # Check visual grounding data
                    if field_response.visual_grounding:
                        print(f"     Visual Grounding Citations: {len(field_response.visual_grounding)}")
                        for i, citation in enumerate(field_response.visual_grounding[:2]):  # Show first 2
                            print(f"       - Citation {i+1}: Page {citation.get('page_no', 'N/A')}, {citation.get('filename', 'N/A')}")

            print(f"\n✓ All tests passed!")
            print(f"✓ Confidence scores are properly stored in database")
            print(f"✓ Visual grounding metadata is included")

        except Exception as e:
            pytest.fail(f"Failed to process extraction run: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
