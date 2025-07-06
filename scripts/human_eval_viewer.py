import json

DATA_PATH = "data/eval_predictions.json"
OUTPUT_PATH = "data/human_ratings.json"

def main():
    with open(DATA_PATH) as f:
        eval_data = json.load(f)

    print("\n=== Human Evaluation Interactive Rating ===\n")
    ratings = []

    for idx, item in enumerate(eval_data):
        print(f"🔢 Example {idx + 1}")
        print(f"💻 Prediction:\n{item['prediction']}")
        print(f"🎯 Reference:\n{item['reference']}")
        score = input("⭐ Rate the prediction (1–5): ")
        feedback = input("📝 Any feedback? (optional): ")
        ratings.append({
            "id": idx + 1,
            "prediction": item["prediction"],
            "reference": item["reference"],
            "score": score,
            "feedback": feedback
        })
        print("-" * 60)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(ratings, f, indent=2)
        print(f"\nSaved ratings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
