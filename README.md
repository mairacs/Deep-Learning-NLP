# Deep Learning for NLP – Sentiment Classification

## Overview

This repository contains the three assignments of the course **Artificial Intelligence II (ΥΣ19)** at the **Department of Informatics and Telecommunications, National and Kapodistrian University of Athens (NKUA / UoA)**, Spring Semester 2024–2025.  

The common task across all assignments was **sentiment classification on a Twitter dataset**, using progressively more advanced approaches:  
1. [Logistic Regression with TF-IDF](./LogisticRegression)  
2. [Feedforward Neural Networks with Word2Vec embeddings](./FFNNs)  
3. [Transformer-based models (BERT)](./BERT) & [(DistilBERT)](./DistilBERT/)   

In each assignment, the workflow generally included:
- **Data preprocessing**: cleaning tweets, removing URLs/mentions, normalizing text, handling emojis/slangs.
- **Feature extraction or embeddings**:  
  • **TF-IDF** in Assignment 1  
  • **Pre-trained embeddings (GloVe / Word2Vec)** in Assignment 2  
  • **Contextual embeddings via BERT / DistilBERT** in Assignment 3  
- **Model training and evaluation**:  
  • **Hyperparameter tuning** (e.g. regularization, learning rates)  
  • Techniques like **early stopping, dropout, batch size experiments**  
  • Validation using accuracy, precision, recall, F1-score, confusion matrices, and learning curves  
- **Comparative analysis**: assessing how performance improves as models move from classical ML to shallow neural networks to transformer-based architectures.

The dataset splits (`train`, `val`, `test`) are included under [/datasets](./datasets/).  
- The **training set** is used to fit the models and learn parameters.  
- The **validation set** is used during development for hyperparameter tuning, monitoring overfitting, and guiding model improvements.  
- The **test set** remains unseen during training and validation, and is used only for the final evaluation of model performance. 


## Assignments Overview

### **Assignment 1 – Logistic Regression (TF–IDF)**

- **Exploratory Data Analysis (EDA)**: Inspected dataset size, sentiment balance (≈50/50 positive vs. negative), word/character distributions, and frequent n-grams (uni/bi/tri-grams) for both positive and negative tweets. This analysis motivated preprocessing choices (e.g., removing URLs, handling slang).
- **Preprocessing**:  
  • Lowercasing, tokenization with a Twitter-specific tokenizer  
  • Expansion of contractions (isn't → is not)  
  • Removal of mentions, URLs, repeated characters/spaces  
  • Slang normalization (e.g., “cuz” → “because”)  
  • Selective stopword removal (`["and", "is", "are", "i"]`)  
- **Feature Extraction**: **TF-IDF vectorization** with `ngram_range=(1,3)`, `min_df=3`, `max_features=50,000`.  
- **Classifier**: **Logistic Regression**, optimized via **Grid Search with 5-fold cross-validation**. Best configuration: `C=2`, `solver='liblinear'`, `max_iter=100`.  
- **Evaluation**: Accuracy, precision, recall, F1-score, classification reports, and learning curves to monitor overfitting/underfitting.  

📊 **Best performance on validation set:**  
- Accuracy: **0.805**  
- Precision: **0.801**  
- Recall: **0.813**  
- F1-score: **0.807**  

The learning curve showed that, with more data, validation accuracy steadily improved and converged closer to training accuracy, reducing overfitting. Despite its simplicity, the TF-IDF + Logistic Regression pipeline achieved strong baseline results on the Twitter sentiment classification task.  

---

### **Assignment 2 – Feedforward Neural Networks (with GloVe embeddings)**

- **Exploratory Data Analysis**: Checked vocabulary size (~234k unique words) and confirmed balanced sentiment distribution (~50% positive / 50% negative), ensuring no class imbalance.  
- **Preprocessing**:  
  • Lowercasing, Twitter-specific tokenization  
  • Contraction expansion (e.g., *isn’t → is not*)  
  • Removal of mentions, hashtags, URLs, special characters, repeated letters/spaces  
  • Slang normalization (e.g., *luv → love*, *idk → I do not know*)  
- **Embeddings**: Tweets represented via averaged **GloVe embeddings (Twitter, 200d)**, converted to Word2Vec format for compatibility with Gensim.  
- **Model Architecture**: PyTorch **Deep Feedforward Neural Network** with 3 hidden layers (512, 256, 128 neurons), ReLU activations, Dropout (0.3), and Batch Normalization.  
- **Training Setup**:  
  • Loss: `BCEWithLogitsLoss`  
  • Optimizer: **Adam**, learning rate `1e-4`  
  • Batch size: 128  
  • Early stopping (patience=5) to prevent overfitting  
  • Max 50 epochs  
- **Hyperparameter Tuning**:  
  • Manual experiments varying depth, batch size, activation functions, optimizers (SGD, Adam, AdamW), learning rate, and loss functions.  
  • Automated search with **Optuna** confirmed best parameters.  
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices, learning curves. Results reported for both validation and Kaggle test set.  

📊 **Best performance on validation set:** 
- Accuracy: **0.7909**  
- Precision: **0.7918**  
- Recall: **0.7870**  
- F1-score: **0.7894**  

Learning curves showed stable convergence with minimal overfitting thanks to dropout, batch normalization, and early stopping.

---

### **Assignment 3 – Transformers (BERT & DistilBERT)**

- **Preprocessing**:  
  • Lowercasing, tweet-specific tokenization (NLTK TweetTokenizer)  
  • Expansion of contractions (*isn’t → is not*)  
  • Removal of mentions, hashtags, and URLs  
  • Normalization of repeated characters/spaces  
  • Correction of common spelling errors (*amzing → amazing*) and slang normalization (*idk → I do not know*)  

- **Models**:  
  • **BERT (bert-base-uncased)** – 12-layer transformer, fine-tuned for binary classification  
  • **DistilBERT (distilbert-base-uncased)** – lighter 6-layer version retaining ~95% of BERT’s performance  

- **Implementation**: HuggingFace **Transformers** + **PyTorch**. Tweets tokenized with the respective pretrained tokenizers (max length = 60 tokens). Input converted to `input_ids` + `attention_masks` tensors, batched via PyTorch `DataLoader`.  

- **Training Setup**:  
  • Optimizer: **AdamW** with weight decay = 0.01  
  • Learning rate: 2e-5  
  • Batch sizes: 32 for BERT, 16 for DistilBERT  
  • Epochs: 2  
  • Learning rate scheduler: linear decay with warmup  
  • Dropout: default 0.1  
  • Early stopping with patience to avoid overfitting  

- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices, learning curves. Results reported for both validation and Kaggle test set.

📊 **Best performance**:  
- **BERT**: Validation accuracy **85.6%**
- **DistilBERT**: Validation accuracy **84.8%**

Despite being smaller, DistilBERT reached nearly the same performance as BERT, confirming its efficiency for real-world deployment. 

---

## Results Summary

| Model                | Features/Embeddings  | Validation Accuracy | Kaggle Test Accuracy |
|----------------------|-----------------------|---------------------|-----------------------|
| Logistic Regression  | TF-IDF               | ~80.5%              | ~80.2%               |
| FFNNs                | Word2Vec (GloVe 200d)| ~79.1%              | ~78.9%               |
| BERT                 | HuggingFace          | ~85.6%              | ~85.6%               |
| DistilBERT           | HuggingFace          | ~84.8%              | ~85.8%               |


## ⚙️ Installation & Usage

### Requirements
- Python 3.10+
- Core libraries:  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`  
  - `scikit-learn`  
  - `torch`, `gensim`, `optuna`  
  - `transformers`, `nltk`, `contractions`

### Running an assignment
Example: Assignment 1 (Logistic Regression)
```bash
cd LogisticRegression
python LogisticRegression.py
```

(The scripts automatically read datasets from the `datasets/` folder.)


## Acknowledgments
This project was developed as part of the [Artificial Intelligence II (ΥΣ19)](.https://www.di.uoa.gr/en/studies/undergraduate/805) course  
at the Department of Informatics and Telecommunications, National and Kapodistrian University of Athens (NKUA/UoA),  
under the supervision of [Prof. Manolis Koubarakis](.https://cgi.di.uoa.gr/~koubarak/).  

I would also like to thank the teaching staff and the course TAs for their guidance and support throughout the assignments.


## License

This project is licensed under the terms of the `Apache-2.0` License.