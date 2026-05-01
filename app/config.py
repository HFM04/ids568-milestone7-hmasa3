"""Configuration for the closed-domain RAG assistant.

The system is intentionally scoped to one source document. If the answer is not
supported by retrieved text from that document, the assistant should refuse with
REFUSAL_MESSAGE.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
LOG_DIR = PROJECT_ROOT / "logs"

SOURCE_DOCX = DATA_DIR / "final_paper.docx"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
AUDIT_LOG_PATH = LOG_DIR / "audit-trail.jsonl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.1"
OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.25
REFUSAL_MESSAGE = "Not found in source."
