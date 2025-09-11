# Copyright (C) 2025 Maira Papadopoulou
# SPDX-License-Identifier: Apache-2.0

# Import basic libraries
import os
import re

import contractions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, learning_curve

# Tokenizer optimized for tweets, which handles hashtags, mentions, and emoticons.
tokenizer = TweetTokenizer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

train_path = os.path.join(DATASET_DIR, "train_dataset.csv")
val_path = os.path.join(DATASET_DIR, "val_dataset.csv")
test_path = os.path.join(DATASET_DIR, "test_dataset.csv")

# Load data sets for kaggle
train_ds = pd.read_csv(train_path)
val_ds = pd.read_csv(val_path)
test_ds = pd.read_csv(test_path)

# EDA
# Total number of tweets
num_tweets = len(train_ds)
num_tweets_val = len(val_ds)

# Total number of words
all_text = train_ds["Text"].dropna()
unique_words = set(" ".join(all_text).split())

# Average number of words and characters per tweet sentiment
train_ds["word_count"] = train_ds["Text"].apply(lambda x: len(str(x).split()))
train_ds["char_count"] = train_ds["Text"].apply(lambda x: len(str(x)))

avg_word_length = train_ds.groupby("Label")["word_count"].mean()
avg_char_length = train_ds.groupby("Label")["char_count"].mean()

labels = pd.Series(["Positive", "Negative"])

plt.figure(figsize=(12, 5))
plt.suptitle(f"Total Words: {len(unique_words)}", fontsize=14, fontweight="bold")
plt.subplot(1, 2, 1)
sns.barplot(x=labels, y=[avg_word_length[1], avg_word_length[0]], palette=["lightgreen", "lightcoral"])
plt.title("Average Word Count per Tweet")
plt.ylabel("Average Word Count")

plt.subplot(1, 2, 2)
sns.barplot(x=labels, y=[avg_char_length[1], avg_char_length[0]], palette=["lightgreen", "lightcoral"])
plt.title("Average Character Count per Tweet")
plt.ylabel("Average Character Count")

plt.tight_layout()
plt.show()

# Total number of positive and negative tweets
num_positive = train_ds["Label"].sum()
num_negative = num_tweets - num_positive
sns.barplot(x=labels, y=[num_positive, num_negative], palette=["lightgreen", "lightcoral"])
plt.suptitle("Total of Positive n' Negative Tweets in Training Set")
plt.title(f"Total Tweets: {num_tweets}", fontsize=14, fontweight="bold")
plt.ylabel("Total")
plt.show()

# Total number of positive and negative tweets
num_positive = val_ds["Label"].sum()
num_negative = num_tweets_val - num_positive
sns.barplot(x=labels, y=[num_positive, num_negative], palette=["lightgreen", "lightcoral"])
plt.suptitle("Total of Positive n' Negative Tweets in Validation Set")
plt.title(f"Total Tweets: {num_tweets_val}", fontsize=14, fontweight="bold")
plt.ylabel("Total")
plt.show()

# Percentage of positive and negative tweets using pie
pos_percentage = (num_positive / num_tweets) * 100
neg_percentage = (num_negative / num_tweets) * 100
plt.figure(figsize=(5, 5))
plt.pie([pos_percentage, neg_percentage], labels=["Positive", "Negative"], autopct="%1.1f%%", colors=["lightgreen", "lightcoral"])
plt.title("Percentage of Positive n Negative Tweets in Validation Set")
plt.show()


# Visualization of top 15 most used words, bigrams, trigrams
def get_ngrams(corpus, n=1, top_n=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words="english").fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
    return words_freq


for n in [1, 2, 3]:
    ngram_data = get_ngrams(all_text, n)
    words, counts = zip(*ngram_data)

    words_series = pd.Series(words)
    counts_series = pd.Series(counts)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=words_series, y=counts_series, palette="Purples_r")
    plt.xticks(rotation=45)
    plt.title(f"Top {n}-grams")
    plt.xlabel(f"{n}-grams")
    plt.ylabel("Frequency")
    plt.show()


# Visualization of top 10 most used words, bigrams, trigrams per positive and negative tweets
positive_text = train_ds[train_ds["Label"] == 1]["Text"].dropna()
negative_text = train_ds[train_ds["Label"] == 0]["Text"].dropna()

for n in [1, 2, 3]:
    pos_ngram_data = get_ngrams(positive_text, n)
    neg_ngram_data = get_ngrams(negative_text, n)

    pos_words, pos_counts = zip(*pos_ngram_data)
    neg_words, neg_counts = zip(*neg_ngram_data)

    pos_words_series = pd.Series(pos_words)
    pos_counts_series = pd.Series(pos_counts)
    neg_words_series = pd.Series(neg_words)
    neg_counts_series = pd.Series(neg_counts)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=pos_words_series, y=pos_counts_series, palette="Greens_r")
    plt.xticks(rotation=45)
    plt.title(f"Top {n}-grams in Positive Tweets")
    plt.xlabel(f"{n}-grams")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.barplot(x=neg_words_series, y=neg_counts_series, palette="Reds_r")
    plt.xticks(rotation=45)
    plt.title(f"Top {n}-grams in Negative Tweets")
    plt.xlabel(f"{n}-grams")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# Text Preprocessing
def preprocess_text(text):
    text = text.lower()                                                 # lowercasing

    text = " ".join(tokenizer.tokenize(text))                           # tokenize Twitter-specific text

    text = contractions.fix(text)                                       # fix contractions like isn't -> is not

    text = re.sub(r"@\w+", "", text)                                    # remove mentions (@username)

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


# Apply the preprocessing function to all data sets
train_ds["Text"] = train_ds["Text"].apply(preprocess_text)
val_ds["Text"] = val_ds["Text"].apply(preprocess_text)
test_ds["Text"] = test_ds["Text"].apply(preprocess_text)

# Partitioning the data
X_train, y_train = train_ds["Text"], train_ds["Label"]
X_val, y_val = val_ds["Text"], val_ds["Label"]
X_test = test_ds["Text"]

# Tried these stopwords firstly, but didn't increase model's accuracy
# stop_wordss = ["the", "is", "are", "was", "were", "not", "have", "to", "this", "that",
#                "there", "be", "so", "you", "do", "does", "did", "of", "just", "and"]

# only these stopwords improved the accuracy of the model, so kept only these
stopwords = ["and", "is", "are", "i"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 3), min_df=3, max_features=50000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Model Development and Evaluation
# Train Logistic Regression model using Grid Search with 5-fold cross-validation
param_grid_search = {"C": [0.1, 0.5, 1, 2],                # Try different regularization values
                     "solver": ["liblinear", "lbfgs"],      # Try different solver types
                     "max_iter": [100, 250, 500]}
grid_search = GridSearchCV(LogisticRegression(), param_grid_search, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# This is an experiment, which I used only Logistic Regression without arguments as a base line
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# Predictions on Validation Dataset
y_pred_val = model.predict(X_val_tfidf)

# Evaluation Metrics
accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)

print("Evaluation Metrics on Validation Dataset:")
print(f"Accuracy: {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1-Score: {f1:.5f}")
print("Classification Report:\n", classification_report(y_val, y_pred_val))


# Predictions on Test Dataset
y_pred_test = model.predict(X_test_tfidf)

# Create submission file
submission_ds = pd.DataFrame({"ID": test_ds["ID"], "Label": y_pred_test})
submission_ds.to_csv("submission.csv", index=False)

# Create learning curves (this code is given)
# Generate learning curve
train_sizes, train_scores, test_scores = learning_curve(
   model, X_train_tfidf, y_train, cv=2, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))

# Compute mean and std of accuracy
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot learnning curve
plt.figure(figsize=(15, 6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

plt.plot(train_sizes, test_mean, label="Validation Score", color="red", marker="s")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve for Logistc Regression")
plt.legend()
plt.show()
