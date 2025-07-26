"""
Azure OpenAI client wrapper for embedding generation.
"""

import numpy as np
import os
from typing import List, Optional
from openai import AzureOpenAI


class AzureEmbeddingClient:
    """Handles Azure OpenAI embedding generation."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """Initialize Azure OpenAI client."""
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version
        self.embedding_model = embedding_model

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure OpenAI endpoint and API key must be provided either as "
                "parameters or environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)",
            )

        print("Initializing Azure OpenAI client...")
        print(f"Endpoint: {self.endpoint}")
        print(f"Model: {self.embedding_model}")

        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        self.api_calls_made = 0

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Azure OpenAI."""
        try:
            print(f"🌐 Generating embeddings for {len(texts)} texts...")
            self.api_calls_made += 1

            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model,
            )

            embeddings = [item.embedding for item in response.data]
            print(f"✅ Successfully generated {len(embeddings)} embeddings (API call #{self.api_calls_made})")

            return embeddings

        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise

    def get_api_call_count(self) -> int:
        """Get the number of API calls made."""
        return self.api_calls_made


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# generate two embeddings and find similarity
if __name__ == "__main__":
    client = AzureEmbeddingClient(embedding_model="text-embedding-ada-002")
    embeddings = client.generate_embeddings(["Database", "PostgreSQL"])
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity: {similarity}")
