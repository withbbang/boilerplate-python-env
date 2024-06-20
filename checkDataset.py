from datasets import load_dataset

# 데이터셋 확인
data_files = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}
dataset =  load_dataset("csv", data_files=data_files, delimiter="\t")

print(dataset)

print(dataset['train']['en'][:3], dataset['train']['ko'][:3])