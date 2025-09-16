# Copyright (C) 2025 Maira Papadopoulou
# SPDX-License-Identifier: Apache-2.0

# Import basic libraries
# Random seed
import os
import random
import re

import contractions
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

# Libraries for neural networks
import torch
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch import nn
from torch.nn.functional import softmax
from torch.optim import Adam, AdamW

# Libraries for DistilBERT
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Tokenizer optimized for tweets, which handles hashtags, mentions, and emoticons.
tweet_tokenizer = TweetTokenizer()


# Text Preprocessing
def preprocess_text(text):
    text = text.lower()                                                 # lowercasing

    text = " ".join(tweet_tokenizer.tokenize(text))                     # tokenize Twitter-specific text

    text = contractions.fix(text)                                       # fix contractions like isn't -> is not

    text = re.sub(r"@\w+", "", text)                                    # remove mentions (@username)

    text = re.sub(r"#\w+", "", text)                                     # remove hashtags

    text = re.sub(r"http\S+|www\S+", "", text)                          # remove URLs

    # text = re.sub(r"[^\w\s]", "", text)                                  # remove special characters(punctuation)
                                                                        # an experiment which decreased model's accuracy

    # text = re.sub(r"[^\x00-\x7F]+", "", text)                            # remove non-ASCII characters
                                                                        # an experiment which decreased model's accuracy

    text = re.sub(r"\s+", " ", text)                                     # remove multiple spaces

    text = re.sub(r"(.)\1{2,}", r"\1", text)                             # remove repeated character (plzzzz -> plz)

    # text = re.sub(r'\d+', '', text)                                      # remove numbers
                                                                        # an experiment which decreased model's accuracy

    # fix some common mistakes
    text = re.sub(r"\b(luv)\b", "love", text)
    text = re.sub(r"\b(amzing)\b", "amazing", text)
    text = re.sub(r"\b(terible)\b", "terrible", text)
    text = re.sub(r"\b(excelent)\b", "excellent", text)
    text = re.sub(r"\b(perfomance)\b", "perfomance", text)
    text = re.sub(r"\b(gub)\b", "good", text)
    text = re.sub(r"\b(vry)\b", "very", text)
    text = re.sub(r"\b(fantstic)\b", "fantastic", text)
    text = re.sub(r"\b(gr8)\b", "great", text)
    text = re.sub(r"\b(horble)\b", "horrible", text)

    text = re.sub(r"\b(cuz)\b", "because", text)
    text = re.sub(r"\b(dnt)\b", "don't", text)
    text = re.sub(r"\b(thnx)\b", "thanks", text)
    text = re.sub(r"\b(plz)\b", "please", text)
    text = re.sub(r"\b(u)\b", "you", text)
    text = re.sub(r"\b(ur)\b", "your", text)
    text = re.sub(r"\b(idk)\b", "i do not know", text)

    return text


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

train_path = os.path.join(DATASET_DIR, "train_dataset.csv")
val_path = os.path.join(DATASET_DIR, "val_dataset.csv")
test_path = os.path.join(DATASET_DIR, "test_dataset.csv")

# Load data sets for kaggle
train_ds = pd.read_csv(train_path)
val_ds = pd.read_csv(val_path)
test_ds = pd.read_csv(test_path)

# Apply the preprocessing function to all data sets
train_ds["Text"] = train_ds["Text"].apply(preprocess_text)
val_ds["Text"] = val_ds["Text"].apply(preprocess_text)
test_ds["Text"] = test_ds["Text"].apply(preprocess_text)

# Split the data into train set, validation set and test set
x_train, y_train = train_ds["Text"], train_ds["Label"]
x_val, y_val = val_ds["Text"], val_ds["Label"]
x_test = test_ds["Text"]

# Take the sentences of training and validation sets
train_sentences = x_train.tolist()
val_sentences = x_val.tolist()

# Tokenizer of DistilBERT
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Watch the average length of tweets in order to choose the max length
lengths = [len(distilbert_tokenizer.tokenize(sent)) for sent in train_sentences]
plt.hist(lengths, bins=50)
plt.title("Average tweet length")
plt.xlabel("Token count")
plt.ylabel("Number of tweets")
plt.grid(True)
plt.show()

# Based on the graph we choose this max_len
max_len = 60


# Function that tokenizes the text in order to be input in DistilBERTmodel
def tokenized_sentences(sentences, max_length, tokenizer):
    encoding = tokenizer(sentences, padding="max_length", truncation=True, max_length=max_length,
    return_tensors="pt",
    return_attention_mask=True,
    add_special_tokens=True)

    return encoding["input_ids"], encoding["attention_mask"]


# Tokenize the tweets of the training, validation and test dataset
train_input_ids, train_attention_masks = tokenized_sentences(x_train.tolist(), max_len, distilbert_tokenizer)
val_input_ids, val_attention_masks = tokenized_sentences(x_val.tolist(), max_len, distilbert_tokenizer)
test_input_ids, test_attention_masks = tokenized_sentences(x_test.tolist(), max_len, distilbert_tokenizer)

# Turn it into tensor
y_train_tensor = torch.tensor(y_train.values)
y_val_tensor = torch.tensor(y_val.values)

# Create the dataloader
bert_train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_masks, y_train_tensor)
bert_val_dataset = torch.utils.data.TensorDataset(val_input_ids, val_attention_masks, y_val_tensor)
bert_test_dataset = torch.utils.data.TensorDataset(test_input_ids, test_attention_masks)
# Experiments for batch size
batch_sizee = 16
# batch_sizee = 64
# batch_sizee = 32
train_dataloader = torch.utils.data.DataLoader(bert_train_dataset, batch_size=batch_sizee, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(bert_val_dataset, batch_size=batch_sizee, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(bert_test_dataset, batch_size=batch_sizee, shuffle=False)


# Function that trains the BERT model
def train_model(model, train_dataloader, val_dataloader, device, optimizer, scheduler, epochs=4, learning_rate=2e-5, clip_grad=True, early_stop_patience=2):
    # Move model to GPU or CPU
    model.to(device)

    # Lists to store loss and accuracy metrics per epoch for the plots
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Variables for early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        total_loss, total_preds, total_labels = 0, [], []

        for batch in train_dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            # Reset gradients before each batch
            optimizer.zero_grad()

            # Forward pass with labels to compute loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Backpropagation
            loss.backward()

            # Apply gradient clipping to prevent exploding gradients during backpropagation
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()
            # Update learning rate
            scheduler.step()

            total_loss += loss.item()
            total_preds += torch.argmax(logits, dim=1).cpu().tolist()
            total_labels += labels.cpu().tolist()

        # Compute average training loss and accuracy for the epoch for the plots
        train_loss = total_loss / len(train_dataloader)
        train_acc = accuracy_score(total_labels, total_preds)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        # Set model to evaluation mode
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_preds += torch.argmax(logits, dim=1).cpu().tolist()
                val_loss += outputs.loss.item()
                total_preds += torch.argmax(logits, dim=1).cpu().tolist()
                val_labels += labels.cpu().tolist()

        # Compute average validation loss and accuracy for the plots
        val_loss /= len(val_dataloader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)


# # Experiment, which change dropout to 0.2
# config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=2,
#     dropout=0.2,
#     attention_dropout=0.2
# )
# distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

# Default DistilBERT
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Hyper-parameter tuning for DistilBERT model and the experiments
learning_rate = 2e-5
# learning_rate = 1e-5
# learning_rate = 3e-5
num_epochs = 2
# num_epochs = 4
# num_epochs = 3
clip_g = True
# optimizer = Adam(distilbert_model.parameters(), lr=learning_rate, betas=(0.9, 0.999),eps=1e-8)
optimizer = AdamW(distilbert_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
# scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps, lr_end=0.0, power=1.0)

# Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the DistilBERT model
distilbert_model, (tr_losses, val_losses, tr_accs, val_accs) = train_model(distilbert_model, train_dataloader, val_dataloader, device, optimizer, scheduler, num_epochs, learning_rate, clip_g, early_stop_patience=4)

# Create submission file
distilbert_model.eval()
y_test_pred = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask = [x.to(device) for x in batch]
        outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_test_pred += preds.cpu().tolist()

submission_ds = pd.DataFrame({"ID": test_ds["ID"], "Label": y_test_pred})
submission_ds.to_csv("submission.csv", index=False)

# Run DistilBERT model for evaluation on validation dataset
distilbert_model.eval()
y_pred_val = []
val_probs = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_pred_val += preds.cpu().tolist()
        val_probs += probs[:, 1].cpu().tolist()


y_val_np = y_val_tensor.cpu().numpy()
# Evaluation metrics
accuracy = accuracy_score(y_val_np, y_pred_val)
precision = precision_score(y_val_np, y_pred_val)
recall = recall_score(y_val_np, y_pred_val)
f1 = f1_score(y_val_np, y_pred_val)
print("\nEvaluation Metrics on Validation Dataset:")
print(f"Accuracy:  {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall:    {recall:.5f}")
print(f"F1-Score:  {f1:.5f}")
print("\nClassification Report:\n", classification_report(y_val_np, y_pred_val))

epochs = list(range(1, len(tr_accs) + 1))

# Plots for the learning curves
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
plt.plot(epochs, tr_accs, label="Train Accuracy", color="orange")
plt.plot(epochs, val_accs, label="Validation Accuracy", color="red")
plt.title("Learning Curve for DistilBERT Model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs, tr_losses, label="Train Loss", color="blue")
plt.plot(epochs, val_losses, label="Validation Loss", color="green")
plt.title("Loss Curve for DistilBERT Model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val_np, val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for DistilBERT model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val_np, y_pred_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix on Validation Set")
plt.grid(False)
plt.show()


# Optuna framework
def objective(trial):
    # Trials for learning rate, dropout, optimizer
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    config = DistilBertConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        dropout=dropout,
        attention_dropout=dropout
    )
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

    if optimizer_type == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = AdamW(distilbert_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    total_steps = len(train_dataloader) * 4
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (_, val_losses, _, val_accs) = train_model(model, train_dataloader, val_dataloader, device, optimizer, scheduler, epochs=4, learning_rate=learning_rate, clip_grad=True, early_stop_patience=2)

    return max(val_accs)


# Start an optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Print best hyperparameters based on optuna
print("Best Hyperparameters:")
print(study.best_params)

# Run DistilBERT with optuna
best = study.best_params
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    dropout=best["dropout"],
    attention_dropout=best["dropout"]
)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

if best["optimizer"] == "Adam":
    optimizer = Adam(model.parameters(), lr=best["learning_rate"], betas=(0.9, 0.999), eps=1e-8)
else:
    optimizer = AdamW(model.parameters(), lr=best["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

total_steps = len(train_dataloader) * 4
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train DistilBERT model
model, (tr_losses, val_losses, tr_accs, val_accs) = train_model(
    model, train_dataloader, val_dataloader, device, optimizer, scheduler,
    epochs=4, learning_rate=best["learning_rate"], clip_grad=True, early_stop_patience=2)

model.eval()
y_pred_val = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        y_pred_val += preds.cpu().tolist()

# Evaluation on Validation Dataset
y_val_np = y_val_tensor.cpu().numpy()
accuracy = accuracy_score(y_val_np, y_pred_val)
precision = precision_score(y_val_np, y_pred_val)
recall = recall_score(y_val_np, y_pred_val)
f1 = f1_score(y_val_np, y_pred_val)
print("\nEvaluation Metrics on Validation Dataset:")
print(f"Accuracy:  {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall:    {recall:.5f}")
print(f"F1-Score:  {f1:.5f}")
print("\nClassification Report:\n", classification_report(y_val_np, y_pred_val))

# Plots
plt.figure(figsize=(10, 6))
plt.plot(tr_accs, label="Train Accuracy", color="orange")
plt.plot(val_accs, label="Validation Accuracy", color="red")
plt.title("Learning Curve for DistilBERT model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(tr_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="green")
plt.title("Loss Curve for DistilBERT model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
