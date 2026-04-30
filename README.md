#  Chinese News Classification using BERT

This project demonstrates Chinese news article classification using a fine-tuned BERT model (`bert-base-chinese`) with Hugging Face's `transformers` library. The goal is to classify news articles into categories such as domestic, international, or general content.

---

##  Project Overview

- **Dataset:** Labeled Chinese news articles (`chinese_news.csv`)
- **Model:** `bert-base-chinese` fine-tuned using Hugging Face `Trainer`
- **Task:** Text classification with 3 classes:
  - `国内 (Local)`
  - `国际 (International)`
  - `详细全文 (General Article)`
- **Tools:** PyTorch, Transformers, Datasets, Scikit-learn, Pandas

---

##  Features

- End-to-end pipeline in a single Jupyter Notebook
- Tokenization using BERT tokenizer (handles Chinese)
- Automatic label encoding and mapping
- Train/Validation/Test split with evaluation metrics
- High classification accuracy (~92%)

---

##  Files

- `ChineseNewsClassifier.ipynb` – Full code for preprocessing, training, evaluation, and prediction
- `README.md` – Project documentation

> **Note:** The dataset file `chinese_news.csv` is assumed to be present locally. It is not included in the repo.

---

##  Requirements

Install dependencies via:

```bash
pip install pandas scikit-learn matplotlib datasets transformers torch
````

---

##  Results

* **Validation Accuracy:** \~92.4%
* **Test Accuracy:** \~92.7%
* **Model Output:** Predictions mapped to human-readable labels for interpretability

---

##  Highlights

* Fine-tuned a BERT model on Chinese text for multi-class classification
* Used Hugging Face `Trainer` API for streamlined training
* Evaluated performance using accuracy and F1-score
* Output includes predicted vs true labels for test set
  

