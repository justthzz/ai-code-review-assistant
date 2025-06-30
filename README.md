# 🧠 AI-Powered Code Review Assistant

An intelligent, context-aware code review assistant that leverages a fine-tuned Qwen 2.5B LLM and Retrieval-Augmented Generation (RAG) to provide accurate review comments, improvement suggestions, and security/style warnings.

---

## 🚀 Project Overview

This project combines QLoRA fine-tuning and RAG techniques to create a smart code review pipeline. It uses:

- **Qwen 2.5B**: A powerful open-source Large Language Model for code understanding.
- **QLoRA**: Efficient low-rank fine-tuning for adapting the base model to code review tasks.
- **FAISS**: Vector search engine for retrieving relevant documentation and coding guidelines.
- **RAG Pipeline**: Combines retrieved context with code snippets for LLM-powered insights.

---

## ✨ Features

- 🔍 Context-aware code review with retrieval from language docs and style guides
- 🤖 Fine-tuned LLM (Qwen 2.5B) using QLoRA on datasets like CodeParrot and CodeSearchNet
- ⚡ Fast document retrieval with FAISS and sentence-transformers
- 💬 Generates suggestions, refactor tips, and security/style warnings
- 🔧 Modular pipeline: Train → Index → Inference

---

## 🛠️ Technologies

- Qwen 2.5B (LLM)
- QLoRA (via Hugging Face PEFT)
- Hugging Face Transformers & Datasets
- FAISS (vector similarity search)
- Sentence Transformers
- Python (backend)

---

## 🧑‍💻 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-code-review-assistant.git
cd ai-code-review-assistant
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Fine-Tune the Model (QLoRA)

```bash
python src/train.py
```

You can change datasets or base models in `train.py`.

### 4. Build the FAISS Knowledge Base

```bash
python src/build_faiss.py --docs_path ./docs/
```

### 5. Run Inference

```bash
python src/inference.py --code_file sample_code.py
```

---

## 📁 Project Structure

```
.
├── data/                # Optional: Sample code snippets or diffs
├── docs/                # Curated programming docs and style guides
├── models/              # Trained LLM checkpoints
├── src/                 
│   ├── train.py         # Fine-tuning with QLoRA
│   ├── build_faiss.py   # Create FAISS index from documentation
│   ├── inference.py     # Run the code review assistant
│   └── utils.py         # Preprocessing, embedding, etc.
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📌 Example Use Case

Paste a code snippet into the input, and the system will:

- Retrieve relevant documentation or guidelines
- Analyze code context
- Generate human-like review comments, suggestions, and warnings

---

## ✅ To-Do

- [ ] GitHub PR integration (via GitHub Actions)
- [ ] Web UI with FastAPI or Streamlit
- [ ] Multi-language support
- [ ] CLI Tool version

---

## 📜 License

MIT License. See `LICENSE` for more details.

---

## 🙌 Credits

Built using open-source technologies from Hugging Face, Alibaba (Qwen), Meta (QLoRA), and community datasets.
