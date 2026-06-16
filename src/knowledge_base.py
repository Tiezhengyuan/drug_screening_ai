  import chromadb
  from chromadb.utils import embedding_functions
  import pdfplumber
  import os

  def build_knowledge_db(pdf_dir: str, persist_dir: str = "chroma_prism"):
      client = chromadb.PersistentClient(path=persist_dir)
      ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
      collection = client.get_or_create_collection(name="prism_papers", embedding_function=ef)

      for fname in os.listdir(pdf_dir):
          if fname.endswith(".pdf"):
              with pdfplumber.open(os.path.join(pdf_dir, fname)) as pdf:
                  text = "\n".join([p.extract_text() or "" for p in pdf.pages])
                  if not text.strip():
                      continue
                  # Chunking
                  chunk_size = 500
                  chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                  collection.add(
                      documents=chunks,
                      ids=[f"{fname}_{i}" for i in range(len(chunks))],
                      metadatas=[{"source": fname, "section": "auto_chunk"}] * len(chunks)
                  )
      return client

