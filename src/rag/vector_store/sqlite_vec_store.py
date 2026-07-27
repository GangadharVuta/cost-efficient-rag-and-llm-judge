import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.rag.vector_store.base import VectorStore

class SQLiteVecStore(VectorStore):
    """
    SQLite-backed embedded vector store.
    Uses native SQLite tables with JSON blob storage for vectors and metadata,
    allowing 0-cost embedded vector search and relational SQL metadata filtering.
    """
    def __init__(self, db_path: str = "./data/vector_store/sqlite_vec.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    char_count INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON document_chunks(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON document_chunks(file_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON document_chunks(category)")
            conn.commit()

    def upsert(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or len(chunks) != len(embeddings):
            return 0

        inserted = 0
        with self._get_conn() as conn:
            for chunk, emb in zip(chunks, embeddings):
                conn.execute("""
                    INSERT INTO document_chunks (
                        chunk_id, doc_id, source_path, file_type, chunk_index,
                        text, category, char_count, content_hash, vector_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        text=excluded.text,
                        vector_json=excluded.vector_json,
                        char_count=excluded.char_count
                """, (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk["source_path"],
                    chunk["file_type"],
                    chunk["chunk_index"],
                    chunk["text"],
                    chunk.get("category", "general"),
                    chunk.get("char_count", len(chunk["text"])),
                    chunk.get("content_hash", ""),
                    json.dumps(emb)
                ))
                inserted += 1
            conn.commit()
        return inserted

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

        sql = "SELECT * FROM document_chunks"
        params = []
        where_clauses = []

        if metadata_filter:
            for k, v in metadata_filter.items():
                if k in ["file_type", "doc_id", "category"]:
                    where_clauses.append(f"{k} = ?")
                    params.append(v)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored = []
        for r in rows:
            v_vec = np.array(json.loads(r["vector_json"]), dtype=np.float32)
            v_norm = np.linalg.norm(v_vec)
            if v_norm > 0:
                v_vec = v_vec / v_norm

            sim = float(np.dot(q_vec, v_vec))
            sim = max(0.0, min(1.0, sim))

            scored.append({
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "source_path": r["source_path"],
                "file_type": r["file_type"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "similarity_score": round(sim, 4),
                "distance": round(1.0 - sim, 4),
                "category": r["category"]
            })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        with self._get_conn() as conn:
            res = conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()
            return res[0] if res else 0

    def clear(self) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM document_chunks")
            conn.commit()
