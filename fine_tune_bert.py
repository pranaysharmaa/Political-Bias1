from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk 
import torch

# Load preprocessed datasets
train_dataset = load_from_disk("data/train_dataset")
test_dataset = load_from_disk("data/test_dataset")

# Load pre-trained BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir="bert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=100
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("political-bias-model")
print("✅ Fine-tuned model saved!")
