"""Vector store service using FAISS with LlamaIndex."""

import os
import pickle
from typing import List, Dict, Any, Optional
import structlog
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from llama_index.core import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..core.config import settings
from ..core.models import DocumentChunk

logger = structlog.get_logger(__name__)


class VectorStoreService:
    """FAISS-based vector store for document embeddings."""

    def __init__(self):
        """Initialize the vector store service."""
        self.vector_store_path = settings.vector_store_path
        self.dimension = 768  # Google text-embedding-004 dimension

        # Initialize Google embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key
        )

        # Initialize FAISS index
        self.index = None
        self.documents = []  # Store document metadata
        self.doc_id_to_index = {}  # Map document IDs to index positions

        # Load existing index if available
        self._load_index()

        logger.info("Vector store service initialized")

    def _load_index(self):
        """Load existing FAISS index and documents."""
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        docs_path = os.path.join(self.vector_store_path, "documents.pkl")

        try:
            if os.path.exists(index_path) and os.path.exists(docs_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)

                # Load documents metadata
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)

                # Rebuild doc_id mapping
                self.doc_id_to_index = {
                    doc['id']: i for i, doc in enumerate(self.documents)
                }

                logger.info(
                    "Loaded existing vector store",
                    num_documents=len(self.documents)
                )
            else:
                # Create new index
                # Inner product for cosine similarity
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info("Created new vector store index")

        except Exception as e:
            logger.error("Error loading vector store", error=str(e))
            # Create new index on error
            self.index = faiss.IndexFlatIP(self.dimension)

    def _save_index(self):
        """Save FAISS index and documents to disk."""
        try:
            os.makedirs(self.vector_store_path, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(
                self.vector_store_path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)

            # Save documents metadata
            docs_path = os.path.join(self.vector_store_path, "documents.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)

            logger.info("Saved vector store to disk")

        except Exception as e:
            logger.error("Error saving vector store", error=str(e))
            raise

    async def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store."""
        try:
            if not chunks:
                logger.warning("No chunks provided to add")
                return False

            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = []

            for text in texts:
                # Use Google embedding via LangChain
                embedding = await self.embedding_model.aembed_query(text)
                embeddings.append(embedding)

            # Convert to numpy array and normalize for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)

            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings_array)

            # Store document metadata
            for i, chunk in enumerate(chunks):
                doc_data = {
                    'id': f"doc_{start_idx + i}",
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'embedding': embeddings[i]
                }
                self.documents.append(doc_data)
                self.doc_id_to_index[doc_data['id']] = len(self.documents) - 1

            # Save to disk
            self._save_index()

            logger.info(
                "Added documents to vector store",
                num_chunks=len(chunks),
                total_docs=len(self.documents)
            )
            return True

        except Exception as e:
            logger.error(
                "Error adding documents to vector store", error=str(e))
            raise

    async def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            if self.index.ntotal == 0:
                logger.warning("Vector store is empty")
                return []

            # Generate query embedding
            query_embedding = await self.embedding_model.aembed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Search in FAISS
            scores, indices = self.index.search(
                query_vector, min(top_k, self.index.ntotal))

            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= score_threshold and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': float(score),
                        'id': doc['id']
                    })

            logger.info(
                "Vector search completed",
                query_length=len(query),
                results_count=len(results)
            )

            return results

        except Exception as e:
            logger.error("Error searching vector store", error=str(e))
            raise

    async def delete_all(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            # Reset index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = []
            self.doc_id_to_index = {}

            # Save empty index
            self._save_index()

            logger.info("Cleared all documents from vector store")
            return True

        except Exception as e:
            logger.error("Error clearing vector store", error=str(e))
            raise

    def has_existing_data(self) -> bool:
        """Check if the vector store contains any existing data."""
        try:
            # Check if index has documents
            if self.index and self.index.ntotal > 0:
                return True

            # Check if documents list has data
            if len(self.documents) > 0:
                return True

            # Even if files exist on disk, if we have no documents loaded,
            # consider it empty (this handles the case where files exist
            # but contain empty data after a clear operation)
            return False

        except Exception as e:
            logger.error("Error checking for existing data", error=str(e))
            # If we can't check, assume there might be data to be safe
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'storage_path': self.vector_store_path,
            'has_existing_data': self.has_existing_data()
        }


# Global vector store service instance
vector_store_service = VectorStoreService()
