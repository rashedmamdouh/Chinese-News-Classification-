# Chinese News Classification System 📰 中文新闻分类系统

A robust Chinese news classifier supporting multiple modeling pipelines including BERT-based fine‑tuning and traditional ML. Suitable for multilingual AI research & NLP demo projects.

---

## 🧾 Overview 

This project implements Chinese news classification using datasets like **THUCNews**, providing full preprocessing, training, evaluation, and visualization workflows.  

---

## 🔧 Key Features 

- **Text preprocessing**: Chinese tokenization using Jieba, optional SnowNLP 
- **Multiple modeling approaches**:
  - Traditional ML: TF‑IDF + SVM, Naive Bayes, Random Forest
  - Deep Learning: CNN, RNN/GRU, BiLSTM models on word embedding or character input
  - Transformer fine‑tuning: BERT (bert-base-chinese), RoBERTa‑Chinese, ERNIE, with task-specific heads 
- **Performance metrics**: accuracy, precision, recall, F1 score, confusion matrices
- **Fault-tolerant training**: supports checkpointing and resume training workflows 

---

## 🛠️ Setup & Dependencies 

```bash
git clone https://github.com/rashedmamdouh/Chinese-News-Classification-
cd Chinese-News-Classification
pip install -r requirements.txt
````

## 🌏 About the Author

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
