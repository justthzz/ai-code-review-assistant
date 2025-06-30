import sys
from inference.reviewer import review_code

def read_code_from_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_review.py <path_to_code_file>")
        sys.exit(1)

    code_path = sys.argv[1]
    code = read_code_from_file(code_path)
    review = review_code(code)
    print("\n🔍 Code Review Suggestion:\n")
    print(review)
