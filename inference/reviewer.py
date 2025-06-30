import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model_id = "Qwen/Qwen1.5-0.5B"
adapter_path = "checkpoints/qlora-qwen"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def review_code(code: str) -> str:
    prompt = f"""
### Code:
{code}

### Task:
You are a senior Python code reviewer. Analyze the code and suggest improvements, highlight any bugs, bad practices, or refactoring suggestions.

### Review:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("### Review:")[-1].strip()

#PYTHONPATH=. python scripts/run_review.py examples/bad_code.py
