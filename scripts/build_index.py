import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOC_PATH = "rag/docs.json"
INDEX_PATH = "rag/index.faiss"

def load_docs():
    with open(DOC_PATH, "r") as f:
        return json.load(f)

def main():
    print("📚 Loading documentation...")
    documents = load_docs()
    texts = [doc["text"] for doc in documents]

    print("🔍 Embedding documents...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("💾 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Saved FAISS index to {INDEX_PATH}")

if __name__ == "__main__":
    main()
