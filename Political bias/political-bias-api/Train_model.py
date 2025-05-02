import os
import glob
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

# ─────── CONFIG ──────────────────────────────────────────────────────────────
# at the top, replace all your Windows‐style paths with their WSL mounts:
DATA_DIR   = "/mnt/c/Users/peeka/Downloads/Politcial-Bias-Model/data"
JSON_DIR   = "/mnt/c/Users/peeka/Downloads/Politcial-Bias-Model/data/Article-Bias-Prediction-main/data/jsons"

# DATA_DIR = r"C:\Users\peeka\Downloads\Politcial-Bias-Model\data"
TXT_LABELS = {"Left Data": 0, "Center Data": 1, "Right Data": 2}
# JSON_DIR = r"C:\Users\peeka\Downloads\Politcial-Bias-Model\data\Article-Bias-Prediction-main\data\jsons"
MODEL_SAVE_PATH = "models/political-bias-model"
MODEL_NAME = "roberta-base"
# ──────────────────────────────────────────────────────────────────────────────

def load_txt_df():
    rows = []
    for folder_name, label in TXT_LABELS.items():
        # WSL path: .../data/Center Data/Center Data
        folder = os.path.join(DATA_DIR, folder_name, folder_name)
        if not os.path.isdir(folder):
            print(f"⚠️  Missing folder: {folder}")
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(".txt"):
                full = os.path.join(folder, fn)
                with open(full, encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    rows.append({"text": txt, "label": label})
    print(f"✅ Loaded {len(rows)} TXT samples")
    return pd.DataFrame(rows)


def load_json_df():
    rows = []
    bias_map = {"left":0, "center":1, "right":2}
    if not os.path.isdir(JSON_DIR):
        print(f"⚠️  Missing JSON folder: {JSON_DIR}")
        return pd.DataFrame(rows)
    for path in glob.glob(os.path.join(JSON_DIR, "*.json")):
        obj = json.load(open(path, encoding="utf-8"))
        content   = obj.get("content","").strip()
        bias_txt  = obj.get("bias_text","").lower()
        if content and bias_txt in bias_map:
            rows.append({"text": content, "label": bias_map[bias_txt]})
    print(f"✅ Loaded {len(rows)} JSON samples")
    return pd.DataFrame(rows)

def main():
    # 1) load dataframes
    df_txt  = load_txt_df()
    df_json = load_json_df()
    df = pd.concat([df_txt, df_json], ignore_index=True).sample(frac=1, random_state=42)
    
    # 2) to HuggingFace Dataset and train/test split
    ds = Dataset.from_pandas(df[["text","label"]])
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, eval_ds = ds["train"], ds["test"]
    
    # 3) tokenizer & model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model     = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # 4) tokenization
    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    train_ds = train_ds.map(tok_fn, batched=True)
    eval_ds  = eval_ds.map(tok_fn, batched=True)
    
    # 5) data collator, metrics
    data_collator = DataCollatorWithPadding(tokenizer)
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted"),
        }
    
    # 6) Trainer + training args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        push_to_hub=False,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 7) train!
    print("🚀 Starting training...")
    trainer.train()
    os.makedirs("logs", exist_ok=True)
    with open("logs/trainer_state.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print("💾 Saving model to", MODEL_SAVE_PATH)
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
