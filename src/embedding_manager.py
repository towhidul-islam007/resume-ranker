"""
Embedding management for candidate matching system.
"""

import hashlib
import time
from typing import List, Optional, Any, Dict
from .azure_client import AzureEmbeddingClient
from .embedding_storage import EmbeddingStorage


class EmbeddingManager:
    """Manages embedding generation and storage with caching."""

    def __init__(
        self,
        azure_client: AzureEmbeddingClient,
        storage: EmbeddingStorage,
    ):
        """Initialize with Azure client and storage."""
        self.azure_client = azure_client
        self.storage = storage

    def get_embeddings_with_storage(
        self,
        # Can be Skill, Experience, Education, Certification, or strings
        items: List[Any],
        category: str,
        candidate_name: Optional[str] = None,
    ) -> List[List[float]]:
        """Get embeddings with caching using model methods."""
        embeddings = []
        items_to_fetch = []
        fetch_indices = []

        # Extract texts and check cache
        for i, item in enumerate(items):
            # Get text representation
            if hasattr(item, "get_text"):
                text = item.get_text()
            else:
                text = str(item)  # Fallback for strings (job requirements)

            cached_embedding = self.get_embedding(
                text, category, candidate_name or "")
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                items_to_fetch.append(item)
                fetch_indices.append(i)

        # Fetch missing embeddings from API
        if items_to_fetch:
            # Extract texts for API call
            texts_to_fetch = []
            metadatas_to_store = []

            for item in items_to_fetch:
                if hasattr(item, "get_text"):
                    texts_to_fetch.append(item.get_text())
                    # Get metadata from model and add category
                    metadata = item.get_metadata()
                    metadata["category"] = category
                    metadatas_to_store.append(metadata)
                else:
                    texts_to_fetch.append(str(item))
                    metadatas_to_store.append({"category": category})

            fetched_embeddings = self.azure_client.generate_embeddings(
                texts_to_fetch)

            # Update results with fetched embeddings
            for i, embedding in enumerate(fetched_embeddings):
                original_index = fetch_indices[i]
                embeddings[original_index] = embedding

            # Store new embeddings with metadata from models
            self.store_embeddings(
                texts_to_fetch,
                fetched_embeddings,
                category,
                candidate_name,
                metadatas=metadatas_to_store,
            )

        return embeddings

    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding-related statistics."""
        storage_stats = self.storage.get_storage_statistics()
        api_calls = self.azure_client.get_api_call_count()

        total_requests = storage_stats["cache_hits"] + api_calls
        hit_rate = (storage_stats["cache_hits"] /
                    total_requests * 100) if total_requests > 0 else 0

        return {
            **storage_stats,
            "api_calls": api_calls,
            "hit_rate": hit_rate,
        }

    def clear_storage(self) -> None:
        """Clear all stored embeddings."""
        self.storage.clear_storage()

    def _generate_embedding_id(self, text: str, category: str = "general", candidate_name: str = "") -> str:
        """Generate a consistent ID for an embedding."""
        text_hash = hashlib.md5(
            f"{self.storage.embedding_model}:{text}".encode()).hexdigest()
        return f"{category}_{candidate_name}_{text_hash}"

    def get_embedding(self, text: str, category: str = "general", candidate_name: str = "") -> Optional[List[float]]:
        """Retrieve embedding from ChromaDB collection if it exists."""
        embedding_id = self._generate_embedding_id(
            text, category, candidate_name)

        try:
            results = self.storage.collection.get(
                ids=[embedding_id],
                include=["embeddings", "documents"],
            )

            if results["ids"] and len(results["ids"]) > 0:
                self.storage.cache_hits += 1
                return results["embeddings"][0]

        except Exception as e:
            print(f"  ⚠️ Error retrieving embedding: {e}")

        return None

    def store_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        category: str = "general",
        candidate_name: Optional[str] = None,
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Store embeddings in ChromaDB collection with optional custom metadata."""
        ids = []
        documents = []
        final_metadatas = []
        embeddings_to_store = []

        for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
            embedding_id = self._generate_embedding_id(
                text, category, candidate_name or "")

            # Check if already exists to avoid duplicates
            if self.get_embedding(text, category, candidate_name or "") is None:
                ids.append(embedding_id)
                documents.append(text)
                embeddings_to_store.append(embedding)

                # Base metadata
                metadata = {
                    "category": category,
                    "model": self.storage.embedding_model,
                    "text_length": len(text),
                    "cached_at": time.time(),
                }

                if candidate_name:
                    metadata["candidate_name"] = candidate_name

                # Add custom metadata if provided
                if metadatas and i < len(metadatas):
                    metadata.update(metadatas[i])

                final_metadatas.append(metadata)

        if ids:  # Only add if there are new embeddings
            try:
                self.storage.collection.add(
                    ids=ids,
                    embeddings=embeddings_to_store,
                    documents=documents,
                    metadatas=final_metadatas,
                )

            except Exception as e:
                print(f"  ⚠️ Error storing embeddings: {e}")

    def query_candidate_data(
        self,
        query_embedding: List[float],
        candidate_name: str,
        category: str,
        n_results: int = 3,
    ) -> Dict:
        """Query for candidate data using vector similarity."""
        return self.storage.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"category": category},
                    {"candidate_name": candidate_name},
                ],
            },
            include=["distances", "documents", "embeddings", "metadatas"],
        )
