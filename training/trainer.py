from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig
from huggingface_hub import constants
import torch
import json
from datasets import DownloadConfig

# Increase timeout and retry settings for Huggingface Hub
constants.HF_HUB_DOWNLOAD_TIMEOUT = 300  # 5 minutes
download_config = DownloadConfig(
    max_retries=5,
    num_proc=1  # Reduce parallel downloads
)

# Load and format dataset
def get_dataset():
    ds = load_dataset(
        "mbpp",
        split="train[:50]",  # Use only first 50 examples
        download_config=download_config
    )

    def format(example):
        return {
            "prompt": f"### Code:\n{example['code']}\n### Task:\nHere is the code. Now, can you review it for issues or improvements?",
            "response": "..."
        }

    return ds.map(format)

# Model and tokenizer IDs
model_id = "Qwen/Qwen1.5-0.5B"

# Load tokenizer with trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model = model.to(torch.device("cpu"))  # or "cuda" if GPU is available

# Load LoRA config from JSON file
with open("configs/qlora_config.json", "r") as f:
    config_dict = json.load(f)
peft_config = LoraConfig(**config_dict)

# Apply LoRA to the base model
model = get_peft_model(model, peft_config)

# Prepare dataset
dataset = get_dataset()

# Tokenize function
def tokenize(example):
    full_text = f"{example['prompt']} {example['response']}"
    tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # causal LM expects labels
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=False)

# Training arguments
args = TrainingArguments(
    output_dir="checkpoints/qlora-qwen",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    num_train_epochs=3,
    fp16=False,  # disable on Mac MPS or CPU
    bf16=False,
    save_strategy="no",  # We'll save manually
)

# Data collator for causal language modeling (no MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer instance
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save QLoRA adapter manually 
adapter_output_dir = "checkpoints/qlora-qwen"
model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
print(f"\n✅ Adapter and tokenizer saved to: {adapter_output_dir}")
