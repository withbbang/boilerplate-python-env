import torch
import numpy as np
import multiprocessing
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, load_metric
from checkDataset import dataset

# 1. gpu가 있다면 device를 세팅
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 모델 및 토큰 설정
model_ckpt = "KETI-AIR/ke-t5-base"
max_token_length = 64

# 3. 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

NUM_CPU = multiprocessing.cpu_count()


def convert_examples_to_features(examples):
    model_inputs = tokenizer(
        examples["en"],
        text_target=examples["ko"],
        max_length=max_token_length,
        truncation=True,
    )

    return model_inputs


tokenized_datasets = dataset.map(
    convert_examples_to_features,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=NUM_CPU,
)
