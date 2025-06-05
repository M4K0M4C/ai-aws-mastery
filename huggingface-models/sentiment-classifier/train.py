from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("csv", data_files={"train": "data/sample.csv"}, split="train")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(output_dir="./results", per_device_train_batch_size=8, num_train_epochs=1)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
