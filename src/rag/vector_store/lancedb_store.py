import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.rag.vector_store.base import VectorStore

class LanceDBVectorStore(VectorStore):
    """
    LanceDB-backed embedded vector store.
    Zero infrastructure cost, Parquet columnar disk layout, fast vector search and metadata filtering.
    """
    def __init__(self, db_path: str = "./data/vector_store/lancedb"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.table = None
        self.table_name = "document_chunks"

        try:
            import lancedb
            self.db = lancedb.connect(str(self.db_path))
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
        except Exception:
            self.db = None

        self.fallback_file = self.db_path / "fallback_store.json"
        self._memory_data: Dict[str, Dict[str, Any]] = {}
        self._load_fallback()

    def _load_fallback(self):
        if self.fallback_file.exists():
            try:
                with open(self.fallback_file, "r", encoding="utf-8") as f:
                    self._memory_data = json.load(f)
            except Exception:
                self._memory_data = {}

    def _save_fallback(self):
        with open(self.fallback_file, "w", encoding="utf-8") as f:
            json.dump(self._memory_data, f)

    def upsert(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or len(chunks) != len(embeddings):
            return 0

        inserted_count = 0

        if self.db is not None:
            try:
                records = []
                for chunk, emb in zip(chunks, embeddings):
                    rec = {
                        "chunk_id": chunk["chunk_id"],
                        "vector": emb,
                        "text": chunk["text"],
                        "doc_id": chunk["doc_id"],
                        "source_path": chunk["source_path"],
                        "file_type": chunk["file_type"],
                        "chunk_index": chunk["chunk_index"],
                        "category": chunk.get("category", "general"),
                        "char_count": chunk.get("char_count", len(chunk["text"])),
                        "content_hash": chunk.get("content_hash", "")
                    }
                    records.append(rec)

                if self.table is None or self.table_name not in self.db.table_names():
                    self.table = self.db.create_table(self.table_name, data=records, mode="overwrite")
                else:
                    existing_ids = set()
                    try:
                        existing_df = self.table.to_pandas()
                        if "chunk_id" in existing_df.columns:
                            existing_ids = set(existing_df["chunk_id"].values)
                    except Exception:
                        pass

                    new_records = [r for r in records if r["chunk_id"] not in existing_ids]
                    if new_records:
                        self.table.add(new_records)
                inserted_count = len(records)
            except Exception:
                pass

        for chunk, emb in zip(chunks, embeddings):
            cid = chunk["chunk_id"]
            self._memory_data[cid] = {
                "chunk": chunk,
                "vector": emb
            }
            inserted_count = max(inserted_count, len(chunks))

        self._save_fallback()
        return inserted_count

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        if self.table is not None:
            try:
                q = self.table.search(query_vector).limit(top_k * 4)

                if metadata_filter:
                    filter_exprs = []
                    for k, v in metadata_filter.items():
                        if isinstance(v, str):
                            filter_exprs.append(f"{k} = '{v}'")
                        else:
                            filter_exprs.append(f"{k} = {v}")
                    if filter_exprs:
                        where_clause = " AND ".join(filter_exprs)
                        q = q.where(where_clause)

                results = q.limit(top_k * 4).to_pandas()
                output = []
                for _, row in results.iterrows():
                    v_vec = np.array(row["vector"], dtype=np.float32)
                    v_norm = np.linalg.norm(v_vec)
                    if v_norm > 0:
                        v_vec = v_vec / v_norm

                    # Compute exact Cosine Similarity
                    sim = float(np.dot(q_vec, v_vec))
                    sim = max(0.0, min(1.0, sim))

                    output.append({
                        "chunk_id": row["chunk_id"],
                        "doc_id": row["doc_id"],
                        "source_path": row["source_path"],
                        "file_type": row["file_type"],
                        "chunk_index": int(row["chunk_index"]),
                        "text": row["text"],
                        "similarity_score": round(sim, 4),
                        "distance": round(1.0 - sim, 4),
                        "category": row.get("category", "general")
                    })

                output.sort(key=lambda x: x["similarity_score"], reverse=True)
                if output:
                    return output[:top_k]
            except Exception:
                pass

        # Fallback search
        if not self._memory_data:
            return []

        scored = []
        for cid, item in self._memory_data.items():
            chunk = item["chunk"]
            if metadata_filter:
                match = True
                for fk, fv in metadata_filter.items():
                    if chunk.get(fk) != fv and chunk.get("metadata", {}).get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            v_vec = np.array(item["vector"], dtype=np.float32)
            v_norm = np.linalg.norm(v_vec)
            if v_norm > 0:
                v_vec = v_vec / v_norm

            sim = float(np.dot(q_vec, v_vec))
            sim = max(0.0, min(1.0, sim))

            scored.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_path": chunk["source_path"],
                "file_type": chunk["file_type"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "similarity_score": round(sim, 4),
                "distance": round(1.0 - sim, 4),
                "category": chunk.get("category", "general")
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        if self.table is not None:
            try:
                return self.table.count_rows()
            except Exception:
                pass
        return len(self._memory_data)

    def clear(self) -> None:
        if self.db is not None:
            try:
                if self.table_name in self.db.table_names():
                    self.db.drop_table(self.table_name)
                self.table = None
            except Exception:
                pass
        self._memory_data = {}
        self._save_fallback()
