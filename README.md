# AI-Powered Code Review Assistant 

An advanced AI-driven code review assistant powered by Qwen 0.5B, a state-of-the-art Small Language Model (SLM), fine-tuned with QLoRA on open-source code datasets. The system integrates Retrieval-Augmented Generation (RAG) using FAISS-based vector search over curated programming documentation, style guides, and security best practices to deliver context-aware code review comments, improvement suggestions, and security warnings.

## Project Overview

This project combines QLoRA fine-tuning and RAG techniques to create a smart code review pipeline. It uses:

- Qwen 0.5B: An efficient open-source SLM for code understanding.
- QLoRA: Low-rank fine-tuning method for adapting the base model.
- FAISS: Vector search engine for context retrieval from documentation.
- SentenceTransformers: To embed code and doc snippets for retrieval.
- RAG Pipeline: Blends retrieved knowledge with code snippets to power code review insights.

## Features

- Context-aware code review based on real documentation
- Code suggestions, refactor tips, and PEP8/security/style warnings
- Fine-tuned model using datasets like CodeParrot and CodeSearchNet
- Human + Automatic evaluation support (BLEU, ROUGE, BERTScore)
- Modular training, retrieval, and inference pipelines

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/justthzz/ai-code-review-assistant.git
cd ai-code-review-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Fine-Tune the Model Using QLoRA
```bash
python src/train.py
```
Modify dataset or model configurations in `src/train.py` as needed.

### 4. Build the FAISS Knowledge Base
```bash
python src/build_faiss.py --docs_path ./docs/
```

### 5. Run Inference
```bash
python src/inference.py --code_file sample_code.py
```

## Project Structure

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

## Evaluation

You can evaluate the model with automatic metrics and human feedback:

```bash
python scripts/evaluate_predictions.py
python scripts/human_eval_viewer.py
```

Supported Metrics:
- BLEU
- ROUGE
- BERTScore

## Example Usage

Input a code snippet to the system, which will:

- Retrieve relevant documentation and guidelines
- Analyze the code with contextual information
- Generate human-like review comments, suggestions, and warnings

## Future Work

- GitHub Pull Request integration via GitHub Actions
- Web UI using FastAPI or Streamlit
- Support for multiple programming languages
- Command-line interface (CLI) tool version

## Credits

Built using open-source technologies from Hugging Face, Alibaba (Qwen), Meta (QLoRA), and community datasets.
