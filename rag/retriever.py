from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(\"rag/index.faiss\")
with open(\"rag/docs.json\") as f:
    docs = json.load(f)

def retrieve(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    return [docs[i] for i in I[0]]
