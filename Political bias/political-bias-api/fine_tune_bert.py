import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer, RobertaForSequenceClassification


from transformers import DataCollatorWithPadding
import torch

# === Paths ===
DATA_PATH = "data/final_bias_dataset.csv"
MODEL_SAVE_PATH = "models/political-bias-model"

# === Load & Prepare Dataset ===
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna()
df = df[["text", "label"]]

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# === Train/Test Split ===
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# === Load Tokenizer & Model ===
print("🧠 Loading RoBERTa...")
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

# === Tokenization Function ===
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# === Data Collator ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=1
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Start Training ===
print("🚀 Starting fine-tuning...")
trainer.train()

# === Save Model ===
print(f"💾 Saving model to {MODEL_SAVE_PATH}")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("✅ Training complete!")
