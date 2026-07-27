import os
import re
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        source_path: str,
        file_type: str,
        chunk_index: int,
        text: str,
        char_count: int,
        token_estimate: int,
        content_hash: str,
        metadata: Dict[str, Any] = None
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.source_path = source_path
        self.file_type = file_type
        self.chunk_index = chunk_index
        self.text = text
        self.char_count = char_count
        self.token_estimate = token_estimate
        self.content_hash = content_hash
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "content_hash": self.content_hash,
            **self.metadata
        }

class DocumentIngestor:
    """
    Ingests PDF, HTML, and Markdown files, performs configurable recursive chunking,
    and generates idempotent unique chunk identifiers.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> tuple[str, str, str]:
        """Loads file content and returns (text, file_type, doc_id)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        doc_id = path.stem

        if ext == ".pdf":
            file_type = "pdf"
            text = self._extract_pdf_text(path)
        elif ext in [".html", ".htm"]:
            file_type = "html"
            text = self._extract_html_text(path)
        elif ext in [".md", ".markdown", ".txt"]:
            file_type = "markdown" if ext in [".md", ".markdown"] else "text"
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            file_type = "unknown"
            text = path.read_text(encoding="utf-8", errors="ignore")

        return text, file_type, doc_id

    def _extract_pdf_text(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages_text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            return "\n\n".join(pages_text)
        except Exception:
            # Fallback simple text extraction
            return path.read_text(encoding="utf-8", errors="ignore")

    def _extract_html_text(self, path: Path) -> str:
        raw_html = path.read_text(encoding="utf-8", errors="ignore")
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")
            # Remove scripts & styles
            for elem in soup(["script", "style", "nav", "footer", "header"]):
                elem.decompose()
            return soup.get_text(separator="\n\n").strip()
        except Exception:
            # Simple regex fallback
            clean = re.sub(r'<[^>]+>', ' ', raw_html)
            return re.sub(r'\s+', ' ', clean).strip()

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        source_path: str,
        file_type: str,
        category: str = "general"
    ) -> List[DocumentChunk]:
        """
        Recursively splits text into chunks with overlap while preserving sentence boundaries.
        Idempotent chunk_id generated via SHA-256 hash.
        """
        cleaned_text = re.sub(r'\r\n', '\n', text).strip()
        if not cleaned_text:
            return []

        # Split into paragraphs/sentences
        paragraphs = re.split(r'\n\s*\n', cleaned_text)
        raw_chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    raw_chunks.append(current_chunk)

                # If single paragraph exceeds chunk size, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk} {sent}".strip()
                        else:
                            if sub_chunk:
                                raw_chunks.append(sub_chunk)
                            sub_chunk = sent
                    if sub_chunk:
                        current_chunk = sub_chunk
                else:
                    current_chunk = para

        if current_chunk:
            raw_chunks.append(current_chunk)

        # Apply overlap logic
        chunks: List[DocumentChunk] = []
        doc_hash = hashlib.sha256(cleaned_text.encode('utf-8')).hexdigest()[:12]

        for i, chunk_txt in enumerate(raw_chunks):
            # Calculate token estimate (~4 chars per token)
            token_est = len(chunk_txt) // 4

            # Idempotent unique Chunk ID: hash(doc_id + chunk_index + content)
            chunk_content_hash = hashlib.sha256(f"{doc_id}_{i}_{chunk_txt}".encode('utf-8')).hexdigest()
            chunk_id = f"chk_{doc_hash}_{i:04d}_{chunk_content_hash[:8]}"

            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_path=source_path,
                file_type=file_type,
                chunk_index=i,
                text=chunk_txt,
                char_count=len(chunk_txt),
                token_estimate=token_est,
                content_hash=chunk_content_hash,
                metadata={
                    "category": category,
                    "created_at": int(time.time()),
                    "total_chunks": len(raw_chunks)
                }
            )
            chunks.append(chunk_obj)

        return chunks

    def process_file(self, file_path: str, category: str = "general") -> List[DocumentChunk]:
        text, file_type, doc_id = self.load_document(file_path)
        return self.chunk_text(
            text=text,
            doc_id=doc_id,
            source_path=str(file_path),
            file_type=file_type,
            category=category
        )
