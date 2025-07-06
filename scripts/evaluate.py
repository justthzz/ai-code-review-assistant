from datasets import load_metric
import json
from transformers import pipeline

# Load predictions and references
with open("data/eval_predictions.json") as f:
    eval_data = json.load(f)

# Separate predictions and references
preds = [item["prediction"] for item in eval_data]
refs = [item["reference"] for item in eval_data]

# Load BLEU and ROUGE
bleu = load_metric("bleu")
rouge = load_metric("rouge")

# Format for BLEU (expects tokens)
bleu_preds = [pred.split() for pred in preds]
bleu_refs = [[ref.split()] for ref in refs]

# BLEU score
bleu_score = bleu.compute(predictions=bleu_preds, references=bleu_refs)
print(f"BLEU: {bleu_score['bleu']:.4f}")

# ROUGE score
rouge_score = rouge.compute(predictions=preds, references=refs)
print("ROUGE:", rouge_score)

# Optionally use BERTScore
from bert_score import score
P, R, F1 = score(preds, refs, lang="en", verbose=True)
print(f"BERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
