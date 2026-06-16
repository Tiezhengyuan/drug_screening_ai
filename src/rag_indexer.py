import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
import hashlib

class PRISMRAGIndexer:
    """Index PRISM PDF publications for local RAG retrieval."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = 500  # characters per chunk
        self.chunk_overlap = 50
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks."""
        return self.embedding_model.encode(chunks, show_progress_bar=True)
    
    def index_pdfs(self, pdf_dir: Path, index_dir: Path):
        """Index all PDFs in a directory."""
        all_chunks = []
        all_metadata = []
        
        for pdf_path in pdf_dir.glob("*.pdf"):
            print(f"Processing: {pdf_path.name}")
            text = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": pdf_path.name,
                    "chunk_id": i,
                    "hash": hashlib.md5(chunk.encode()).hexdigest()
                })
        
        if not all_chunks:
            print("No text chunks extracted. Check PDF files.")
            return
        
        print(f"Created {len(all_chunks)} chunks from {len(list(pdf_dir.glob('*.pdf')))} PDFs")
        
        # Create embeddings
        embeddings = self.create_embeddings(all_chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_dir / "prism_index.faiss"))
        
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        with open(index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(all_metadata, f)
        
        print(f"Index saved to {index_dir}")