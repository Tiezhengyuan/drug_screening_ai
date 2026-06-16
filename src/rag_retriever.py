import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Tuple

class PRISMRAGRetriever:
    """Retrieve relevant context from indexed PRISM publications."""
    
    def __init__(self, index_dir: Path, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load index and chunks
        self.index = faiss.read_index(str(index_dir / "prism_index.faiss"))
        with open(index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        with open(index_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "relevance_score": float(1.0 / (1.0 + distances[0][i]))
                })
        return results