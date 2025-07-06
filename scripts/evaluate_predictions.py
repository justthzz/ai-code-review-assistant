from evaluate import load as load_metric
import json
from bert_score import score

# Load predictions and references
with open("data/eval_predictions.json") as f:
    eval_data = json.load(f)

# Separate predictions and references (as raw strings)
preds = [item["prediction"] for item in eval_data]
refs = [item["reference"] for item in eval_data]

# BLEU (HuggingFace style: expects raw strings)
bleu = load_metric("bleu")
bleu_score = bleu.compute(predictions=preds, references=refs)
print(f"BLEU: {bleu_score['bleu']:.4f}")

# ROUGE
rouge = load_metric("rouge")
rouge_score = rouge.compute(predictions=preds, references=refs)
print("ROUGE:", {k: f"{v:.4f}" for k, v in rouge_score.items()})

# BERTScore
P, R, F1 = score(preds, refs, lang="en", verbose=False)
print(f"BERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
