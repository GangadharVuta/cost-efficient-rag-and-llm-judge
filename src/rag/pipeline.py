import time
import os
import re
from typing import List, Dict, Any, Optional
from src.config import settings
from src.rag.ingest import DocumentIngestor, DocumentChunk
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store.base import VectorStore
from src.rag.vector_store.lancedb_store import LanceDBVectorStore
from src.rag.vector_store.sqlite_vec_store import SQLiteVecStore

class RAGPipeline:
    """
    RAG Pipeline supporting multi-store backends, embedding generation,
    grounded LLM answer generation with citations, and telemetry logging.
    """
    def __init__(
        self,
        vector_store_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        store_type = vector_store_type or settings.rag.vector_store_type
        emb_model = embedding_model or settings.rag.embedding_model

        c_size = chunk_size or settings.rag.chunk_size
        c_overlap = chunk_overlap or settings.rag.chunk_overlap
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.rag.similarity_threshold

        self.ingestor = DocumentIngestor(chunk_size=c_size, chunk_overlap=c_overlap)
        self.embedder = EmbeddingGenerator(model_name=emb_model)

        if store_type.lower() == "sqlite_vec":
            self.store: VectorStore = SQLiteVecStore()
        else:
            self.store: VectorStore = LanceDBVectorStore()

        self.store_type = store_type

    def ingest_file(self, file_path: str, category: str = "general") -> Dict[str, Any]:
        start_time = time.time()
        chunks: List[DocumentChunk] = self.ingestor.process_file(file_path, category=category)
        if not chunks:
            return {"status": "empty", "chunks_added": 0, "latency_ms": 0.0}

        chunk_dicts = [c.to_dict() for c in chunks]
        texts = [c.text for c in chunks]

        embeddings = self.embedder.embed_texts(texts)
        count = self.store.upsert(chunk_dicts, embeddings)

        latency = (time.time() - start_time) * 1000.0
        return {
            "status": "success",
            "file_path": str(file_path),
            "doc_id": chunks[0].doc_id,
            "chunks_added": count,
            "latency_ms": round(latency, 2)
        }

    def ingest_directory(self, dir_path: str, category: str = "general") -> Dict[str, Any]:
        start_time = time.time()
        p = os.walk(dir_path)
        total_chunks = 0
        files_processed = 0

        for root, _, files in p:
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".pdf", ".html", ".htm", ".md", ".txt"]:
                    fp = os.path.join(root, f)
                    res = self.ingest_file(fp, category=category)
                    total_chunks += res.get("chunks_added", 0)
                    files_processed += 1

        latency = (time.time() - start_time) * 1000.0
        return {
            "files_processed": files_processed,
            "total_chunks_added": total_chunks,
            "latency_ms": round(latency, 2)
        }

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Step 1: Query embedding
        q_emb_start = time.time()
        q_embedding = self.embedder.embed_query(query_text)
        retrieval_start = time.time()

        # Step 2: Vector Store Retrieval
        retrieved_chunks = self.store.search(
            query_vector=q_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000.0

        # Step 3: Filter chunks by similarity threshold
        relevant_chunks = [c for c in retrieved_chunks if c["similarity_score"] >= self.similarity_threshold]

        # Step 4: Grounded LLM Generation
        gen_start = time.time()
        if not relevant_chunks:
            answer = "I do not have sufficient relevant context in the provided documents to answer your question."
            citations = []
            prompt_tokens = len(query_text) // 4
            completion_tokens = len(answer) // 4
        else:
            answer, citations, prompt_tokens, completion_tokens = self._generate_grounded_answer(
                query_text, relevant_chunks
            )

        gen_latency = (time.time() - gen_start) * 1000.0
        total_latency = (time.time() - start_time) * 1000.0

        return {
            "query": query_text,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "used_chunks_count": len(relevant_chunks),
            "telemetry": {
                "total_latency_ms": round(total_latency, 2),
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "generation_latency_ms": round(gen_latency, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "vector_store": self.store_type
            }
        }

    def _generate_grounded_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> tuple[str, List[str], int, int]:
        """Generates answer strictly grounded in retrieved chunks with citations."""

        context_str = ""
        citations = []
        for i, c in enumerate(chunks, 1):
            cite_tag = f"[Doc: {c['doc_id']}, Chunk: {c['chunk_index']}]"
            citations.append(cite_tag)
            context_str += f"\n--- Source {i}: {cite_tag} ---\n{c['text']}\n"

        prompt = f"""You are a helpful AI assistant. Answer the user question strictly using the provided context chunks below.
If the answer cannot be found in the context, reply: 'I do not have sufficient relevant context in the provided documents to answer your question.'
Always cite your source chunks inline using tags like [Doc: <doc_id>, Chunk: <chunk_index>].

Context:
{context_str}

Question: {query}
Answer:"""

        # Check API key / LLM provider
        provider = settings.llm.provider.lower()
        if provider == "openai" and settings.llm.openai_api_key:
            try:
                import requests
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.llm.openai_api_key}"},
                    json={
                        "model": settings.llm.generator_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    },
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    ans = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    return ans, citations, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            except Exception:
                pass

        # Offline deterministic answer synthesizer (Extracts exact sentences matching query keywords and adds citations)
        query_words = set(re.findall(r'\w+', query.lower())) - {"what", "is", "the", "a", "an", "how", "why", "where", "in", "of", "to", "for"}
        best_sentences = []

        for c in chunks:
            cite_tag = f"[Doc: {c['doc_id']}, Chunk: {c['chunk_index']}]"
            sentences = re.split(r'(?<=[.!?])\s+', c['text'])
            for sent in sentences:
                sent_words = set(re.findall(r'\w+', sent.lower()))
                overlap = len(query_words.intersection(sent_words))
                if overlap > 0:
                    best_sentences.append((overlap, f"{sent.strip()} {cite_tag}"))

        if best_sentences:
            best_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sents = [s[1] for s in best_sentences[:3]]
            synthesized_answer = " ".join(top_sents)
        else:
            # Fallback to direct top chunk extract with citation
            synthesized_answer = f"{chunks[0]['text'][:300]}... {citations[0]}"

        p_toks = len(prompt) // 4
        c_toks = len(synthesized_answer) // 4
        return synthesized_answer, citations, p_toks, c_toks
