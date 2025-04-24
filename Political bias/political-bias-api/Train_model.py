import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

# === Paths & Config ===
DATA_PATH = os.getenv("DATA_PATH", "data/final_bias_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models/political-bias-model")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Training hyperparameters
NUM_EPOCHS = 20  # balance under/overfitting
BATCH_SIZE = 16  # adjust to fit your GPU memory
MAX_LENGTH = 256  # truncate/pad length
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Training on device: {device}")

# === Load & preprocess dataset ===
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH).dropna(subset=["text","label"])

# simple shuffle + split
dataset = Dataset.from_pandas(df)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split['train'], split['test']

# === Tokenizer & Model ===
print("🧠 Initializing RoBERTa...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=3
)
model.to(device)

# === Tokenization function ===
def tokenize_fn(batch):
    return tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
eval_ds  = eval_ds.map(tokenize_fn, batched=True)

# set format for PyTorch
columns = ['input_ids','attention_mask','label']
train_ds.set_format(type='torch', columns=columns)
eval_ds.set_format(type='torch', columns=columns)

data_collator = DataCollatorWithPadding(tokenizer)

# === Metrics ===

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

# === TrainingArguments ===
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    fp16=torch.cuda.is_available(),  # mixed precision if CUDA enabled
    logging_dir=LOG_DIR,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# === Train ===
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# === Save final model/tokenizer ===
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"💾 Model and tokenizer saved to {MODEL_DIR}")
