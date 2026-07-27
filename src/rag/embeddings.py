import math
import numpy as np
from typing import List

class EmbeddingGenerator:
    """
    Generates text embeddings using SentenceTransformers (all-MiniLM-L6-v2)
    with a deterministic fallback embedding model if sentence-transformers is unavailable.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dimensionality = 384
        self.st_model = None

        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name)
            if hasattr(self.st_model, "get_embedding_dimension"):
                self.dimensionality = self.st_model.get_embedding_dimension()
            else:
                self.dimensionality = self.st_model.get_sentence_embedding_dimension()
        except Exception as e:
            # Fallback to local deterministic feature encoder
            self.st_model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.st_model is not None:
            embeddings = self.st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.tolist()
        else:
            return [self._fallback_encode(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def _fallback_encode(self, text: str) -> List[float]:
        """Deterministic 384-dimensional normalized n-gram embedding fallback."""
        vec = np.zeros(384, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for word in words:
            # Hash characters to indices
            for i in range(len(word)):
                char_code = ord(word[i])
                idx = (char_code * 31 + i * 17) % 384
                vec[idx] += 1.0

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
