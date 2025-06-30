import json
import os

RAW_DATA_PATH = "data/raw_code_examples.json"
OUTPUT_PATH = "data/processed_dataset.json"

def convert_to_instruction_format(example):
    return {
        "instruction": "Review the following code and provide suggestions.",
        "input": example["code"],
        "output": example["review"]
    }

def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

    with open(RAW_DATA_PATH, "r") as infile:
        raw_data = json.load(infile)

    processed = [convert_to_instruction_format(item) for item in raw_data]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as outfile:
        json.dump(processed, outfile, indent=2)

    print(f"✅ Processed {len(processed)} examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
