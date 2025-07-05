import torch
import json
import faiss
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Paths
DOC_PATH = "rag/docs.json"
INDEX_PATH = "rag/index.faiss"
ADAPTER_PATH = "checkpoints/qlora-qwen"
BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"

# Load Retrieval Stuff 
print("🔍 Loading FAISS index and docs...")
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(DOC_PATH, "r") as f:
    documents = json.load(f)

def retrieve_context(query_code, k=3):
    query_vec = retriever_model.encode([query_code])
    D, I = index.search(query_vec, k)
    return [documents[i]["text"] for i in I[0]]

# Load Fine-Tuned Code Reviewer Model
print("🧠 Loading fine-tuned model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Combine Prompt
def create_prompt(code: str, context_docs: list) -> str:
    context_block = "\n\n".join([f"Doc #{i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
    return f"""
### Context:
{context_block}

### Code:
{code}

### Task:
You are a strict senior Python reviewer. Find ALL issues in the code below — formatting, naming, logic, style, structure — and suggest concrete improvements with examples.

### Review:
"""

# Inference Function
def review_code_with_context(code: str):
    context_docs = retrieve_context(code)
    prompt = create_prompt(code, context_docs)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            # temperature=0.7,
            # top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("### Review:")[-1].strip()

# CLI Entry
def read_code_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/review_with_context.py <path_to_code_file>")
        sys.exit(1)

    code_path = sys.argv[1]
    code = read_code_from_file(code_path)
    print("\n📎 Retrieving context and generating review...\n")
    review = review_code_with_context(code)
    print("🧪 Review Result:\n")
    print(review)

#python scripts/review_with_context.py examples/bad_code.py
