"""
ChromaDB storage manager for embeddings with caching capabilities.
"""

import hashlib
import time
from typing import Dict, List, Optional
import chromadb


class EmbeddingStorage:
    """Manages embedding storage and retrieval using ChromaDB."""

    def __init__(self, collection_name: str = "skill_embeddings", embedding_model: str = "text-embedding-ada-002"):
        """Initialize ChromaDB storage."""
        self.embedding_model = embedding_model
        self.chroma_client = chromadb.PersistentClient(path="./chroma")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": f"All skill embeddings using {embedding_model}"},
        )

        self.cache_hits = 0

        print(f"ChromaDB collection initialized: {self.collection.name}")



    def get_storage_statistics(self) -> Dict[str, any]:
        """Get detailed storage statistics."""
        try:
            total_count = self.collection.count()

            # Get breakdown by category
            candidate_categories = [
                "candidate_skills",
                "candidate_experience",
                "candidate_education",
                "candidate_certifications",
            ]
            job_categories = ["job_skills", "job_experience",
                              "job_education", "job_certifications"]

            candidate_count = 0
            job_count = 0

            for category in candidate_categories:
                try:
                    results = self.collection.get(
                        where={"category": category}, include=["metadatas"])
                    candidate_count += len(results["ids"]
                                           ) if results["ids"] else 0
                except:
                    pass

            for category in job_categories:
                try:
                    results = self.collection.get(
                        where={"category": category}, include=["metadatas"])
                    job_count += len(results["ids"]) if results["ids"] else 0
                except:
                    pass

            return {
                "cache_hits": self.cache_hits,
                "total_embeddings": total_count,
                "candidate_embeddings": candidate_count,
                "job_embeddings": job_count,
            }

        except Exception as e:
            print(f"💾 Storage statistics unavailable: {e}")
            return {
                "cache_hits": self.cache_hits,
                "total_embeddings": 0,
                "candidate_embeddings": 0,
                "job_embeddings": 0,
            }

    def clear_storage(self) -> None:
        """Clear all stored embeddings."""
        try:
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = self.chroma_client.create_collection(
                name="skill_embeddings",
                metadata={
                    "description": f"All skill embeddings using {self.embedding_model}"},
            )
            print("🗑️ Embedding storage cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing storage: {e}")


if __name__ == "__main__":
    storage = EmbeddingStorage()
    storage.get_storage_statistics()
