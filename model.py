from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from tokenizer import model_ckpt, device, tokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

encoder_inputs = tokenizer(
    ["Studies have been shown that owning a dog is good for you"], return_tensors="pt"
)["input_ids"].to(device)

decoder_targets = tokenizer(
    ["개를 키우는 것이 건강에 좋다는 연구 결과가 있습니다."], return_tensors="pt"
)["input_ids"].to(device)

print(encoder_inputs)
print(decoder_targets)
