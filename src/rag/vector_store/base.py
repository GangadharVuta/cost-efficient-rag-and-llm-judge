from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def upsert(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Upserts document chunks and vector embeddings.
        Idempotent — existing chunk_ids should be updated or skipped without duplication.
        Returns count of inserted/updated vectors.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search with top_k and metadata filtering.
        Returns list of matching chunks sorted by similarity score with metadata.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns total vector count stored in the index."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Deletes all stored vectors and tables."""
        pass
