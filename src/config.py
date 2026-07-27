import os
import sys
from pathlib import Path

# Automatically include user site-packages if needed
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

base_dir = Path(__file__).resolve().parent.parent

# Load .env file manually if python-dotenv is not installed
env_file = base_dir / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

class RAGSettings:
    def __init__(self):
        self.vector_store_type = os.getenv("RAG_VECTOR_STORE_TYPE", "lancedb")
        self.embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
        self.top_k = int(os.getenv("RAG_TOP_K", "4"))
        self.similarity_threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.20"))
        self.data_dir = str(base_dir / "data" / "vector_store")

class LLMSettings:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "offline")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.generator_model = os.getenv("LLM_GENERATOR_MODEL", "gpt-4o-mini")
        self.judge_model = os.getenv("LLM_JUDGE_MODEL", "gpt-4o")

class AppSettings:
    def __init__(self):
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.base_dir = base_dir
        self.rag = RAGSettings()
        self.llm = LLMSettings()

settings = AppSettings()
