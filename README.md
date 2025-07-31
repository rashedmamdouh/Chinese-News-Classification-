# Chinese News Classification System 📰 中文新闻分类系统

A robust Chinese news classifier supporting multiple modeling pipelines including BERT-based fine‑tuning and traditional ML. Suitable for multilingual AI research & NLP demo projects.

---

## 🧾 Overview 

This project implements Chinese news classification using datasets like **THUCNews**, providing full preprocessing, training, evaluation, and visualization workflows.  

---

## 🔧 Key Features 

- **Text preprocessing**: Chinese tokenization using Jieba, optional SnowNLP :contentReference[oaicite:1]{index=1}  
- **Multiple modeling approaches**:
  - Traditional ML: TF‑IDF + SVM, Naive Bayes, Random Forest :contentReference[oaicite:2]{index=2}  
  - Deep Learning: CNN, RNN/GRU, BiLSTM models on word embedding or character input :contentReference[oaicite:3]{index=3}  
  - Transformer fine‑tuning: BERT (bert-base-chinese), RoBERTa‑Chinese, ERNIE, with task-specific heads :contentReference[oaicite:4]{index=4}  
- **Performance metrics**: accuracy, precision, recall, F1 score, confusion matrices, ROC curves visualization :contentReference[oaicite:5]{index=5}  
- **Fault-tolerant training**: supports checkpointing and resume training workflows :contentReference[oaicite:6]{index=6}  

---

## 📂 Repository Structure 

```

/
├── README.md
├── data/                   # Raw and processed datasets (e.g. THUCNews)
├── src/                    # Core scripts: preprocessing, feature engineering, training, evaluation
├── models/                 # Saved trained models / checkpoints
├── notebooks/              # Jupyter notebooks for experiments and visualization
├── requirements.txt
└── reports/                # Metrics plots: confusion matrices, ROC curves, learning curves

````

---

## 🛠️ Setup & Dependencies 

```bash
git clone https://github.com/rashedmamdouh/Chinese-News-Classification-
cd Chinese-News-Classification
pip install -r requirements.txt
````

## 🚀 Usage Examples

### 1. Traditional ML & Neural Models

```bash
python src/train_ml.py  # TF‑IDF + SVM/RF/NB experiments
python src/train_dl.py  # CNN/RNN/LSTM on embeddings
```

### 2. Transformer-Based Fine-Tuning

```bash
python src/train_transformer.py --model_name bert-base-chinese
```

### 3. Evaluation & Visualization

Produces classification reports, confusion matrix images, ROC curves, and learning curves automatically in `reports/`.

### 4. REST API (optional)

```bash
uvicorn app:app --reload
# or
python app.py
```

Send news text to endpoints like `/predict` to get label and confidence output.

---

## 📊 Evaluation Metrics

* **Traditional models**: accuracy \~0.92+ with SVM/MaxEnt on THUCNews datasets ([GitHub][2], [computer.org][3])
* **Deep learning models**: CNN/RNN baseline often reaches 0.94+, transformer fine‑tuned models reach up to **0.98+ F1** ([GitHub][4], [GitHub][1], [SpringerLink][5])

---

## ⚙️ Customization & Extensions

* Try **ERNIE or RoBERTa‑Chinese** models for improved accuracy over BERT ([SpringerLink][5])
* Incorporate **data augmentation**, **key feature enhancement** (KFE‑CNN) to boost low-resource classification performance (\~98% accuracy) ([MDPI][6])
* Add **multi-label** or **hierarchical news categories**
* Deploy frontend interface with multilingual support (English, 中文, العربية) using Node.js/React
* Add **model distillation** or compression for mobile/embedded usage

---

## 🌏 About the Author | عن المطوّر

**Rashed Mamdouh** – AI & software engineer, native Arabic & English speaker, currently learning Chinese. Focus fields: NLP, Transformers, web–AI integration.

---

## 📌 Future Roadmap

* Add web-based form interface for live classification
* Explore multilingual classification pipelines (e.g. Chinese ↔ Arabic news)
* Introduce metrics dashboards (Streamlit, TensorBoard)
* Integrate sentiment analysis or fake‑news detection modules

---

[1]: https://github.com/weiwenfeng/bert-chinese-news-ai?utm_source=chatgpt.com "中文新闻分类系统 (Chinese News Classification System) - GitHub"
[2]: https://github.com/LiPaoFu/chinese-news-classification/blob/main/README.md?utm_source=chatgpt.com "chinese-news-classification/README.md at main - GitHub"
[3]: https://www.computer.org/csdl/proceedings-article/bigcomp/2018/364901a681/12OmNzy7uN3?utm_source=chatgpt.com "Chinese News Classification - Computer"
[4]: https://github.com/Skura3/Chinese_news_text_classification_?utm_source=chatgpt.com "Skura3/Chinese_news_text_classification_ - GitHub"
[5]: https://link.springer.com/chapter/10.1007/978-981-19-7184-6_8?utm_source=chatgpt.com "Research on Chinese News Text Classification Based on ERNIE Model"
[6]: https://www.mdpi.com/2076-3417/13/9/5399?utm_source=chatgpt.com "Chinese News Text Classification Method via Key Feature Enhancement - MDPI"
