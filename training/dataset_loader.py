# from datasets import load_dataset

# def get_dataset():
#     ds = load_dataset("codeparrot/github-code", split="train[:1%]")
#     def format(example):
#         return {
#             "prompt": f"### Code:\n{example['content']}\n### Task:\nGive a code review comment.",
#             "response": "..."
#         }
#     return ds.map(format)
