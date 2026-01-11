from datasets import load_dataset

dataset = load_dataset("mteb/GerDaLIRSmall", "corpus", split="corpus")
print(dataset)

dataset = load_dataset("mteb/GerDaLIRSmall", "queries", split="queries")
print(dataset)

dataset = load_dataset("mteb/GerDaLIRSmall")
print(dataset)
