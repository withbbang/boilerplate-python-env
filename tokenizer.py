import torch
import numpy as np
import multiprocessing
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
from checkDataset import dataset

# 1. gpu가 있다면 device를 세팅
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 모델 및 토큰 설정
model_ckpt = "KETI-AIR/ke-t5-base"
max_token_length = 64

# 3. 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def convert_examples_to_features(examples):
    ###########################################################################
    # with 쓰는 옛날 방식
    # input_encodings = tokenizer(examples['en'], 
    #                             max_length=max_token_length, truncation=True)
    
    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    # target_encodings = tokenizer(text_target=examples['ko'], 
    #                             max_length=max_token_length, truncation=True)
    #
    #
    # return {
    #     "input_ids": input_encodings["input_ids"],
    #     "attention_mask": input_encodings["attention_mask"],
    #     "labels": target_encodings["input_ids"]
    # }
    
    # 그런데 이렇게 하면 인풋하고 한번에 처리 가능함.
    model_inputs = tokenizer(examples['en'],
                             text_target=examples['ko'], 
                             max_length=max_token_length, truncation=True)
    
    return model_inputs

