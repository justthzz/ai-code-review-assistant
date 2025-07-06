# AI-Powered Code Review Assistant

An intelligent, context-aware code review assistant that leverages a fine-tuned Qwen 0.5B Small Language Model (SLM) combined with Retrieval-Augmented Generation (RAG) to provide accurate review comments, improvement suggestions, and security/style warnings.

---

## Project Overview

This project integrates QLoRA fine-tuning and RAG techniques to build a smart code review pipeline. It utilizes:

- **Qwen 0.5B**: An efficient Small Language Model optimized for code understanding tasks.
- **QLoRA**: Low-rank adaptation method enabling efficient fine-tuning of large models on limited hardware.
- **FAISS**: Vector similarity search engine for retrieving relevant documentation and coding guidelines.
- **RAG Pipeline**: Combines retrieved contextual documents with code snippets for enhanced LLM-powered insights.

---

## Features

- Context-aware code review using retrieval from curated programming documentation and style guides.
- Fine-tuned Qwen 0.5B model with QLoRA on code review datasets.
- Fast document retrieval with FAISS and sentence-transformers embeddings.
- Generates detailed suggestions, refactoring tips, and security/style warnings.
- Modular pipeline covering training, indexing, and inference stages.

---

## Technologies Used

- Qwen 0.5B Small Language Model
- QLoRA fine-tuning via Hugging Face PEFT
- Hugging Face Transformers and Datasets
- FAISS for vector similarity search
- Sentence Transformers for text embedding
- Python backend implementation

---

### Clone the Repository

```bash
git clone https://github.com/justthzz/ai-code-review-assistant.git
cd ai-code-review-assistant

---

## 📁 Project Structure

```
.
├── checkpoints/         # Fine-tuned QLoRA adapter checkpoints (e.g., qlora-qwen)
├── configs/             # Configuration files for training and fine-tuning
├── data/                # Raw and processed datasets, e.g., code snippets and reviews
├── docs/                # Curated programming documentation and style guides used for retrieval
├── examples/            # Example code files for testing and inference
├── inference/           # Inference scripts and utilities
├── rag/                 # Retrieval-Augmented Generation related files (FAISS index, docs JSON)
├── scripts/             # Utility scripts for running training, inference, or preprocessing
├── training/            # Training related scripts and utilities
├── src/                 # Core source code (training, building FAISS index, inference, utils)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules

```

---

## 📌 Example Use Case

Paste a code snippet into the input, and the system will:

- Retrieve relevant documentation or guidelines
- Analyze code context
- Generate human-like review comments, suggestions, and warnings

---

## To-Do

- [ ] GitHub Pull Request integration via GitHub Actions.
- [ ] Web user interface using FastAPI or Streamlit.
- [ ] Support for multiple programming languages.
- [ ] Command-line interface (CLI) tool version.

---

## Credits

Built using open-source technologies from Hugging Face, Alibaba (Qwen), Meta (QLoRA), and community datasets.
