# Smart Article Categorizer

A demonstration pipeline that classifies news articles into six categories—**Tech**, **Finance**, **Healthcare**, **Sports**, **Politics**, and **Entertainment**—using four different embedding methods and a simple Logistic Regression classifier.

---

## 🚀 Features

- **Four Embedding Backends**  
  1. **Word2Vec/GloVe** (average word vectors)  
  2. **BERT** ("[CLS]" token)  
  3. **Sentence‑BERT** (all‑MiniLM‑L6‑v2)  

- **Classification Pipeline**  
  - Train one Logistic Regression per embedding  
  - Evaluate Accuracy, Precision, Recall & F₁ (weighted)  
  - Compare performance across embeddings

- **Command‑Line & Web UI**  
  - `main.py` for terminal‑based inference  
  - `streamlit_app.py` for real‑time browser UI  
  - Confidence tables & bar‑charts  
  - Optional UMAP cluster visualization

- **Flexible Data Loading**  
  - Load & split multiple CSVs  
  - Or pull from HuggingFace datasets (Education, PubMedQA, Sports)

---

## 📁 Directory Structure

```
smart_article_categorizer/
├── data/                    # (optional) your CSV files
├── models/                  # pretrained Word2Vec + trained .joblib classifiers
├── results/                 # performance report & sample data for UI clusters
├── src/
│   ├── data_loader.py       # CSV + HF dataset loading & train/test split
│   ├── embedding_models.py  # Word2Vec, BERT, & Sentence‑BERT embedding funcs
│   ├── train.py             # end‑to‑end training & eval pipeline
│   ├── main.py              # CLI for single‑article classification
│   └── streamlit_app.py     # Streamlit‑based web dashboard
├── requirements.txt         # Python dependencies
└── README.md               # this file
```

---

## ⚙️ Prerequisites

- Python 3.8+  
- (Optional) `git`, `virtualenv`  
- HuggingFace account & `huggingface-cli login` if you load private HF datasets  

---

## 🛠 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your‑username/smart_article_categorizer.git
   cd smart_article_categorizer
   ```

2. **Create & activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Download pretrained Word2Vec**
   Place `GoogleNews‑vectors‑negative300.bin` into `models/` or update its path in `train.py` & `streamlit_app.py`.

---

## 📊 Training & Evaluation

Run the full pipeline on your chosen data source:

```bash
cd src
python train.py
```

This will:
- Load & split your data (CSV or HuggingFace datasets)
- Compute embeddings for each method
- Train & evaluate one LogisticRegression per embedding
- Serialize `.joblib` models into `models/`
- Write `results/embedding_performance.csv` with metrics

---

## 💻 CLI Usage

Classify a single article from the terminal:

```bash
python src/main.py --text "Your article text here."
# or
python src/main.py --file /path/to/article.txt
```

---

## 🌐 Web Dashboard

1. **Run Streamlit**
   ```bash
   cd src
   streamlit run streamlit_app.py
   ```

2. **In your browser**
   - Paste an article into the textarea
   - Click **Classify**
   - View prediction table, confidence bar‑chart
   - (Optional) enable UMAP clustering

---

## 🧹 Clearing HuggingFace Cache

If you've pulled HF datasets and want to reclaim space:

**Manual (Linux/macOS):**
```bash
rm -rf ~/.cache/huggingface/datasets
```

**Programmatic:** Set `cache_dir="../hf_cache"` in your `load_dataset` calls, then after training:
```python
import shutil
shutil.rmtree("../hf_cache", ignore_errors=True)
```

---

## 📈 Analysis & Recommendations

After training, inspect `results/embedding_performance.csv` to see which embedding approach:
- Maximizes F₁‑score
- Balances precision vs. recall
- Offers the fastest runtime for your data size

Use that insight to choose your production embedding.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -m "feat: add bar"`)
4. Push (`git push origin feature/foo`) & open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- HuggingFace for transformers and datasets
- Streamlit for the web interface
- Google for Word2Vec pretrained vectors