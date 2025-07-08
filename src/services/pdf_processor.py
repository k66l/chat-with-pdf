"""PDF processing service using LlamaIndex."""

import os
from typing import List, Dict, Any
import structlog
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
import pypdf

from ..core.config import settings
from ..core.models import DocumentChunk
from .vector_store import vector_store_service

logger = structlog.get_logger(__name__)


class PDFProcessorService:
    """Service for processing PDF documents."""

    def __init__(self):
        """Initialize the PDF processor service."""
        self.pdf_storage_path = settings.pdf_storage_path
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # PDF reader will use pypdf instead of PyMuPDF

        logger.info(
            "PDF processor service initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    async def process_pdf_file(self, file_path: str) -> List[DocumentChunk]:
        """Process a single PDF file and return document chunks."""
        try:
            logger.info("Processing PDF file", file_path=file_path)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            # Read PDF using pypdf
            documents = []
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            text=text,
                            metadata={
                                'page': page_num + 1,
                                'source': file_path
                            }
                        )
                        documents.append(doc)

            if not documents:
                logger.warning("No content extracted from PDF",
                               file_path=file_path)
                return []

            # Process each document (page)
            chunks = []
            file_name = os.path.basename(file_path)

            for page_idx, doc in enumerate(documents):
                if not doc.text.strip():
                    continue

                # Split text into chunks using get_nodes_from_documents
                temp_doc = Document(text=doc.text)
                text_nodes = self.text_splitter.get_nodes_from_documents([
                                                                         temp_doc])
                text_chunks = [node.text for node in text_nodes]

                for chunk_idx, chunk_text in enumerate(text_chunks):
                    if not chunk_text.strip():
                        continue

                    # Create document chunk
                    chunk = DocumentChunk(
                        content=chunk_text.strip(),
                        metadata={
                            'source': file_name,
                            'file_path': file_path,
                            'page': page_idx + 1,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(text_chunks),
                            'chunk_size': len(chunk_text)
                        }
                    )
                    chunks.append(chunk)

            logger.info(
                "Successfully processed PDF",
                file_path=file_path,
                total_chunks=len(chunks),
                total_pages=len(documents)
            )

            return chunks

        except Exception as e:
            logger.error("Error processing PDF file",
                         file_path=file_path, error=str(e))
            raise

    async def process_pdf_directory(self, directory_path: str = None, force: bool = False) -> int:
        """Process all PDF files in a directory."""
        try:
            if directory_path is None:
                directory_path = self.pdf_storage_path

            logger.info("Processing PDF directory",
                        directory_path=directory_path)

            if not os.path.exists(directory_path):
                logger.warning("PDF directory does not exist",
                               directory_path=directory_path)
                return 0

            # Check for existing data unless force is True
            if not force and vector_store_service.has_existing_data():
                existing_files = self.list_processed_files()
                total_docs = len(vector_store_service.documents)

                logger.warning(
                    "Vector store already contains data",
                    existing_files_count=len(existing_files),
                    total_documents=total_docs
                )

                error_msg = (
                    f"Vector store already contains {total_docs} documents "
                    f"from {len(existing_files)} files. "
                    "Please clear the existing data first using the 'clear' command "
                    "or use --force to append to existing data."
                )

                raise ValueError(error_msg)

            # Find all PDF files
            pdf_files = []
            for file_path in Path(directory_path).glob("**/*.pdf"):
                pdf_files.append(str(file_path))

            if not pdf_files:
                logger.warning("No PDF files found in directory",
                               directory_path=directory_path)
                return 0

            total_chunks = 0
            processed_files = 0

            for pdf_file in pdf_files:
                try:
                    # Process individual PDF
                    chunks = await self.process_pdf_file(pdf_file)

                    if chunks:
                        # Add to vector store
                        success = await vector_store_service.add_documents(chunks)

                        if success:
                            total_chunks += len(chunks)
                            processed_files += 1
                            logger.info(
                                "Added PDF to vector store",
                                file=os.path.basename(pdf_file),
                                chunks=len(chunks)
                            )
                        else:
                            logger.error(
                                "Failed to add PDF to vector store", file=pdf_file)

                except Exception as e:
                    logger.error(
                        "Error processing individual PDF",
                        file=pdf_file,
                        error=str(e)
                    )
                    continue

            logger.info(
                "Completed PDF directory processing",
                processed_files=processed_files,
                total_files=len(pdf_files),
                total_chunks=total_chunks
            )

            return total_chunks

        except Exception as e:
            logger.error("Error processing PDF directory", error=str(e))
            raise

    async def ingest_single_pdf(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """Ingest a single PDF file into the vector store."""
        try:
            # Check for existing data unless force is True
            if not force and vector_store_service.has_existing_data():
                existing_files = self.list_processed_files()
                total_docs = len(vector_store_service.documents)

                return {
                    'success': False,
                    'message': (
                        f'Vector store already contains {total_docs} documents '
                        f'from {len(existing_files)} files. '
                        'Please clear the existing data first using the "clear" command '
                        'or use --force to append to existing data.'
                    ),
                    'chunks_processed': 0,
                    'existing_files': [f['file_name'] for f in existing_files],
                    'existing_documents': total_docs
                }

            # Process PDF
            chunks = await self.process_pdf_file(file_path)

            if not chunks:
                return {
                    'success': False,
                    'message': 'No content extracted from PDF',
                    'chunks_processed': 0
                }

            # Add to vector store
            success = await vector_store_service.add_documents(chunks)

            if success:
                return {
                    'success': True,
                    'message': f'Successfully ingested {os.path.basename(file_path)}',
                    'chunks_processed': len(chunks),
                    'file_name': os.path.basename(file_path)
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to add documents to vector store',
                    'chunks_processed': 0
                }

        except Exception as e:
            logger.error("Error ingesting PDF",
                         file_path=file_path, error=str(e))
            return {
                'success': False,
                'message': f'Error processing PDF: {str(e)}',
                'chunks_processed': 0
            }

    def list_processed_files(self) -> List[Dict[str, Any]]:
        """List all files that have been processed and are in the vector store."""
        try:
            # Get unique sources from vector store
            sources = set()
            for doc in vector_store_service.documents:
                if 'source' in doc['metadata']:
                    sources.add(doc['metadata']['source'])

            files_info = []
            for source in sources:
                # Count chunks for this source
                chunk_count = sum(
                    1 for doc in vector_store_service.documents
                    if doc['metadata'].get('source') == source
                )

                files_info.append({
                    'file_name': source,
                    'chunk_count': chunk_count
                })

            return sorted(files_info, key=lambda x: x['file_name'])

        except Exception as e:
            logger.error("Error listing processed files", error=str(e))
            return []

    async def clear_all_documents(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            success = await vector_store_service.delete_all()
            if success:
                logger.info("Cleared all documents from vector store")
            return success
        except Exception as e:
            logger.error("Error clearing documents", error=str(e))
            return False


# Global PDF processor service instance
pdf_processor_service = PDFProcessorService()
