from sentence_transformers import SentenceTransformer
import faiss
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(docs):
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def save_index(index, path):
    faiss.write_index(index, path)
