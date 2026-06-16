import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "faiss_index"

# Create directories
for d in [DATA_DIR, PDF_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3