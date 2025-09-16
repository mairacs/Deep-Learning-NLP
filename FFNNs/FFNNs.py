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
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch import nn
from transformers import BertTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Tokenizer optimized for tweets, which handles hashtags, mentions, and emoticons.
tokenizer = TweetTokenizer()


# Text Preprocessing
def preprocess_text(text):
    text = text.lower()                                                 # lowercasing

    text = " ".join(tokenizer.tokenize(text))                           # tokenize Twitter-specific text

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

# Fill NaN values
train_ds = train_ds.fillna("")
val_ds = val_ds.fillna("")
test_ds = test_ds.fillna("")

# Split the data into train set, validation set and test set
x_train, y_train = train_ds["Text"], train_ds["Label"]
x_val, y_val = val_ds["Text"], val_ds["Label"]
x_test = test_ds["Text"]

# Take the sentences of training and validation sets
train_sentences = x_train.tolist()
val_sentences = x_val.tolist()
# Load pre-trained model tokenizer (vocabulary)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

max_len = 0

for sent in train_sentences:
    tokens = bert_tokenizer.tokenize(sent)
    max_len = max(max_len, len(tokens))

print("Max sentence length (in tokens):", max_len)

lengths = [len(bert_tokenizer.tokenize(sent)) for sent in train_sentences]
plt.hist(lengths, bins=50)
plt.title("Average tweet length")
plt.xlabel("Token count")
plt.ylabel("Number of tweets")
plt.grid(True)
plt.show()


# Load GloVe embeddings and convert to word2vec format
glove_input_file = "glove.twitter.27B.200d.txt"
w2v_output_file = "glove_twitter_w2v.txt"
glove2word2vec(glove_input_file, w2v_output_file)
w2v_model = KeyedVectors.load_word2vec_format(w2v_output_file, binary=False)
embedding_dim = w2v_model.vector_size

# Tokenize and embed using GloVe
tokens_train = [tokenizer.tokenize(tweet) for tweet in x_train]
tokens_val = [tokenizer.tokenize(tweet) for tweet in x_val]


# Convert tweet to embedding using the trained Word2Vec model
def tweet_to_vector(tweet, vector_dim, word_model):
    tvector = np.zeros(vector_dim).astype(np.float64)
    words = tokenizer.tokenize(tweet)

    count = 0

    for word in words:
        if word in word_model:
            tvector += word_model[word]
            count += 1

    if count == 0:
        return np.zeros(vector_dim, dtype=np.float64)

    return tvector / count


# Convert all tweets to embeddings
x_train_vec = np.array([tweet_to_vector(tweet, embedding_dim, w2v_model) for tweet in x_train])
x_val_vec = np.array([tweet_to_vector(tweet, embedding_dim, w2v_model) for tweet in x_val])
x_test_vec = np.array([tweet_to_vector(tweet, embedding_dim, w2v_model) for tweet in x_test])

# Save in tensors
x = torch.tensor(x_train_vec, dtype=torch.float64)
y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_val_tensor = torch.tensor(x_val_vec, dtype=torch.float64)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
x_test_tensor = torch.tensor(x_test_vec, dtype=torch.float64)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")


# Create a Neural Network with 3 hidden layers
class Net(nn.Module):
    def __init__(self, D_in, H1, H2, H3, H4, D_out, dropout_rate=0.3):
        super(Net, self).__init__()

        # Activation function is RELU in all layers
        self.activation = nn.ReLU()

        # Layer 1
        self.input_layer = nn.Linear(D_in, H1)
        self.BN_1 = nn.BatchNorm1d(H1)

        # Layer 2
        self.hidden_layer_1 = nn.Linear(H1, H2)
        self.BN_2 = nn.BatchNorm1d(H2)

        # Laywr 3
        self.hidden_layer_2 = nn.Linear(H2, H3)
        self.BN_3 = nn.BatchNorm1d(H3)

        # Layer 4
        # self.hidden_layer_3 = nn.Linear(H3, H4)
        # self.BN_4 = nn.BatchNorm1d(H4)

        # Output layer
        self.output_layer = nn.Linear(H3, D_out)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        h1 = self.input_layer(x)
        h1 = self.BN_1(h1)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)

        h2 = self.hidden_layer_1(h1)
        h2 = self.BN_2(h2)
        h2 = self.activation(h2)
        h2 = self.dropout(h2)

        h3 = self.hidden_layer_2(h2)
        h3 = self.BN_3(h3)
        h3 = self.activation(h3)
        h3 = self.dropout(h3)

        # h4 = self.hidden_layer_3(h3)
        # h4 = self.BN_4(h4)
        # h4 = self.activation(h4)
        # h4 = self.dropout(h4)

        out = self.output_layer(h3)
        return out


# Define layer sizes and model
D_in = x.shape[1]  # size of the input sample
H1, H2, H3, H4 = 512, 256, 128, 64  # size of hidden layers
D_out = 1  # size of the output sample
model = Net(D_in, H1, H2, H3, H4, D_out).double()

# Define Hyperparameters of the model with the Experiments too
# Learning rate
learning_rate = 1e-4
# learning_rate = 1e-3
# learning_rate = 1e-5

# Loss function
loss_func = nn.BCEWithLogitsLoss()
# loss_func = nn.MSELoss(reduction='sum')

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

# Initialise dataloader
dataset = torch.utils.data.TensorDataset(x, y)  # class to represent the data as list of tensors. x=input_features, y=labels
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Function that trains model
def train_model(dataloader, x_val_tensor, y_val_tensor, model, loss_func, optimizer):
    best_val_loss = float("inf")
    # How many iterations will endure the model without improving
    patience = 5
    counter = 0

    # Variables for learning curves
    train_losses_plt = []
    train_accuracies_plt = []
    val_losses_plt = []
    val_accuracies_plt = []

    for epoch in range(50):
        # model in training mode
        model.train()

        batch_losses = []
        # Variables for learning curve of training
        correct_train = 0
        total_train = 0

        # for every batch in dataloader
        for x_batch, y_batch in dataloader:
            # Delete previously stored gradients
            optimizer.zero_grad()

            # forward pass
            y_pred = model(x_batch)

            # count the loss and store it
            loss = loss_func(y_pred, y_batch.float())
            batch_losses.append(loss.item())

            # Perform backpropagation starting from the loss calculated in this epoch
            loss.backward()

            # Update model's weights based on the gradients calculated during backprop
            optimizer.step()

            # Accuracy during training batch
            pred_labels = (torch.sigmoid(y_pred) > 0.5).int()
            # count how many predictions are right
            correct_train += (pred_labels == y_batch.int()).sum().item()
            total_train += y_batch.size(0)

        # Variables for learning curves
        avg_train_loss = sum(batch_losses) / len(dataloader)
        train_losses_plt.append(avg_train_loss)
        train_acc = correct_train / total_train
        train_accuracies_plt.append(train_acc)

        # model in validation dataset
        model.eval()
        with torch.no_grad():
            # forward pass
            val_logits = model(x_val_tensor)
            val_probs = torch.sigmoid(val_logits)
            y_pred_val = (val_probs > 0.5).int()

            # variables for the learning curves
            val_loss = loss_func(val_logits, y_val_tensor).item()
            val_acc = (y_pred_val == y_val_tensor.int()).sum().item() / y_val_tensor.size(0)

        val_losses_plt.append(val_loss)
        val_accuracies_plt.append(val_acc)

        print(f"Epoch {epoch:2}: Train Loss = {avg_train_loss:.5f} | Val Loss = {val_loss:.5f}")

        # Test for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping activated.")
                break
    return train_losses_plt, train_accuracies_plt, val_losses_plt, val_accuracies_plt, model, y_pred_val


# Train the model
train_losses_plt, train_accuracies_plt, val_losses_plt, val_accuracies_plt, model, y_pred_val = train_model(dataloader, x_val_tensor, y_val_tensor, model, loss_func, optimizer)

# Predictions on Test Dataset
model.eval()
with torch.no_grad():
    test_logits = model(x_test_tensor)
    test_probs = torch.sigmoid(test_logits)
    y_test_pred = (test_probs > 0.5).int().squeeze().numpy()

# Create submission file
submission_ds = pd.DataFrame({"ID": test_ds["ID"], "Label": y_test_pred})
submission_ds.to_csv("submission.csv", index=False)

# Convert labels in to NumPy int for metrics
y_val_np = y_val.values.astype(int)
accuracy = accuracy_score(y_val_np, y_pred_val)
precision = precision_score(y_val_np, y_pred_val)
recall = recall_score(y_val_np, y_pred_val)
f1 = f1_score(y_val_np, y_pred_val)
# Classification Report
print("\nEvaluation Metrics on Validation Dataset:")
print(f"Accuracy:  {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall:    {recall:.5f}")
print(f"F1-Score:  {f1:.5f}")
print("\nClassification Report:\n", classification_report(y_val_np, y_pred_val))
# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses_plt, label="loss", color="blue")
plt.plot(train_accuracies_plt, label="accuracy", color="orange")
plt.plot(val_losses_plt, label="val_loss", color="green")
plt.plot(val_accuracies_plt, label="val_accuracy", color="red")
plt.title("Learning curve for Deep Neural Networks")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0.3, 0.9)
plt.legend()
plt.grid(True)
plt.show()

# Optuna Optimization
# Set hyperparameters search ranges for Optuna
H1_r = H2_r = H3_r = H4_r = LR_r = Opt_r = None


# Optuna sample
class Objective:
    def __init__(self, D_in, D_out, x_val_tensor, y_val_tensor, dataset):
        self.D_in = D_in
        self.D_out = D_out
        self.x_val_tensor = x_val_tensor
        self.y_val_tensor = y_val_tensor
        self.dataset = dataset

    def __call__(self, trial):
        # Search for number of neurons per hidden layer
        H1 = trial.suggest_int("H1", H1_r[0], H1_r[1])
        H2 = trial.suggest_int("H2", H2_r[0], H2_r[1])
        H3 = trial.suggest_int("H3", H3_r[0], H3_r[1])
        H4 = trial.suggest_int("H4", H4_r[0], H4_r[1])
        # Search for most suitable learning rate
        learning_rate = trial.suggest_float("learning_rate", LR_r[0], LR_r[1], log=True)
        # Search for most suitable optimizer
        optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "AdamW"])

        # Create model and loss
        loss_func = nn.BCEWithLogitsLoss()
        optuna_model = Net(self.D_in, H1, H2, H3, H4, self.D_out).double()
        optimizer_class = torch.optim.Adam if optimizer_name == "Adam" else torch.optim.AdamW
        optimizer = optimizer_class(optuna_model.parameters(), lr=learning_rate)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=128, shuffle=True)

        train_losses_plt, train_accuracies_plt, val_losses_plt, val_accuracies_plt, optuna_model, y_pred_val = train_model(dataloader,
        x_val_tensor, y_val_tensor, optuna_model, loss_func, optimizer)

        # Evaluate using accuracy on validation set
        y_val_np = self.y_val_tensor.squeeze().numpy().astype(int)
        y_pred_np = y_pred_val.squeeze().numpy().astype(int)
        return accuracy_score(y_val_np, y_pred_np)


# Tests for learning rate
LR_r = [1e-5, 1e-1]
# Tests for number of neurons per hidden layer
H1_r = [8, 512]
H2_r = [8, 512]
H3_r = [8, 512]
H4_r = [8, 512]

D_in = x.shape[1]
D_out = 1
objective = Objective(D_in, D_out, x_val_tensor, y_val_tensor, dataset)

# Create and run Optuna study
num_of_trials = 10
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=num_of_trials)

# Save the best parameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Create and train final model with best parameters
optuna_model = Net(D_in, best_params["H1"], best_params["H2"], best_params["H3"], best_params["H4"], D_out).double()
learning_rate = best_params["learning_rate"]
loss_func = nn.BCEWithLogitsLoss()
optimizer_class = torch.optim.Adam if best_params["optimizer_name"] == "Adam" else torch.optim.AdamW
optimizer = optimizer_class(optuna_model.parameters(), lr=best_params["learning_rate"])
train_losses_plt, train_accuracies_plt, val_losses_plt, val_accuracies_plt, optuna_model, y_pred_val = train_model(dataloader, x_val_tensor, y_val_tensor, optuna_model, loss_func, optimizer)

# Convert labels in to NumPy int for metrics
y_val_np = y_val.values.astype(int)
accuracy = accuracy_score(y_val_np, y_pred_val)
precision = precision_score(y_val_np, y_pred_val)
recall = recall_score(y_val_np, y_pred_val)
f1 = f1_score(y_val_np, y_pred_val)
# Classification report
print("\nEvaluation Metrics on Validation Dataset:")
print(f"Accuracy:  {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall:    {recall:.5f}")
print(f"F1-Score:  {f1:.5f}")
print("\nClassification Report:\n", classification_report(y_val_np, y_pred_val))
# Plot for learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses_plt, label="loss", color="blue")
plt.plot(train_accuracies_plt, label="accuracy", color="orange")
plt.plot(val_losses_plt, label="val_loss", color="green")
plt.plot(val_accuracies_plt, label="val_accuracy", color="red")
plt.title("Learning curve for Deep Neural Networks after Optuna")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0.4, 0.9)
plt.legend()
plt.grid(True)
plt.show()
