# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from datasets import load_dataset

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
torch.cuda.empty_cache()

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# %%
from dotenv import load_dotenv
load_dotenv()

os.chdir(os.getenv("PROJECT_ROOT_DIR"))
print(os.getcwd())

# %%
def read_data(path):
	with open(path, "r") as f:
		data = f.readlines()
		vectors = [token for token in data]
		return vectors


# %%
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_TRAIN_EPOCHS = 10
    BATCH_SIZE = 32
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIMIT = 2
    MAX_STEPS = int(32000)
    OUTPUT_DIR = "./results"
    LOGGING_DIR = "./logs"
    MAX_LENGTH = 512
    TRAIN_DATA_PATH = "data/exp/train_set.parquet"
    TEST_DATA_PATH = "data/exp/test_set.parquet"


# %%
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# %%
print("Loading data")



# %%
def tok(example):
    encodings = tokenizer(example['text'], truncation=True, padding=True, max_length=Config.MAX_LENGTH)
    return encodings

def convert_labels_to_long(example):
    example['label'] = int(example['label'])  # Convert to integer if it's not
    return example

train_dataset = load_dataset("./", data_files={'train': Config.TRAIN_DATA_PATH},  split="train", streaming=True)
# train_dataset.add_column("label", train_labels)
val_dataset = load_dataset("./", data_files={'train': Config.TEST_DATA_PATH}, split="train", streaming=True)
# val_dataset.add_column("label", test_labels)

train_dataset = train_dataset.map(convert_labels_to_long)
val_dataset = val_dataset.map(convert_labels_to_long)

train_dataset = train_dataset.map(tok, batched=True, batch_size=32)
val_dataset = val_dataset.map(tok, batched=True, batch_size=32)

train_dataset = train_dataset.with_format("torch")
val_dataset = val_dataset.with_format("torch")

# %%
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2).to(Config.DEVICE)

# %%
# Load pre-trained CodeBERT model and add a classification layer
# 2 classes: benign, malicious

training_args = TrainingArguments(
    output_dir=Config.OUTPUT_DIR,
    num_train_epochs=Config.NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=Config.BATCH_SIZE,
    per_device_eval_batch_size=Config.BATCH_SIZE,
    warmup_steps=Config.WARMUP_STEPS,
    weight_decay=Config.WEIGHT_DECAY,
    logging_dir=Config.LOGGING_DIR,
    save_total_limit=Config.SAVE_TOTAL_LIMIT,
	max_steps=Config.MAX_STEPS,
)


# Define training arguments and set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# %%
# Train the model
trainer.train()

# Evaluation
predictions, labels, _ = trainer.predict(val_dataset)
predictions = np.argmax(predictions, axis=1)

# %%
# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained('./results/checkpoint-5000').to(Config.DEVICE)

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=5,  # you can update this
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     save_total_limit=2,  # only last 2 checkpoints are saved, older ones are deleted.
# )

# trainer = Trainer(
#     model=model,  # the model you loaded
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Continue training
# trainer.train()


# %%
print(classification_report(labels, predictions))

# %%
print("Accuracy:", accuracy_score(labels, predictions))
print("Precision:", precision_score(labels, predictions))
print("Recall:", recall_score(labels, predictions))
print("F1:", f1_score(labels, predictions))

# %%
confusion = confusion_matrix(labels, predictions)

plt.subplots(figsize=(6, 6))
sns.set(font_scale=1.4)  # for label size
sns.heatmap(confusion, annot=True, fmt=".0f", annot_kws={"size": 16}, cbar=False)  # font size
plt.xlabel("Target (true) Class")
plt.ylabel("Output (predicted) class")
plt.title("Confusion Matrix")
plt.show()

plt.savefig("confusion_matrix.png")


