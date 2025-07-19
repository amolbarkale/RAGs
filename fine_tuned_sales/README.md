# 🔮 Fine-Tuned Sales Conversion Predictor

A machine learning project that predicts sales call outcomes using fine-tuned BERT embeddings and contrastive learning. The project compares baseline SBERT models with custom fine-tuned BERT models to predict whether a sales call will result in a "won" or "lost" outcome.

## 🎯 Project Overview

This project implements and compares two approaches for sales conversion prediction:

1. **SBERT Baseline**: Uses pre-trained sentence transformers with logistic regression
2. **BERT-Contrastive**: Custom fine-tuned BERT model using contrastive learning for domain-specific embeddings

## 📁 Project Structure

```
fine_tuned_sales/
├── data/
│   └── sales_transcripts.csv          # Sales call transcripts dataset
├── models/
│   ├── bert_contrastive/              # Fine-tuned BERT model directory
│   ├── clf_sbert_base.joblib          # SBERT classifier
│   └── clf_bert_contrastive.joblib    # BERT-contrastive classifier
├── src/
│   ├── baseline.py                    # SBERT baseline training
│   ├── contrastive_finetune_llama.py  # BERT contrastive fine-tuning
│   ├── evaluate.py                    # Model evaluation and comparison
│   ├── main.py                        # Main execution script
│   ├── eda.py                         # Exploratory Data Analysis
│   └── streamlit_app.py               # Web interface for predictions
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd fine_tuned_sales

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Step 1: Train SBERT baseline model
python src/baseline.py

# Step 2: Fine-tune BERT with contrastive learning
python src/contrastive_finetune_llama.py

# Step 3: Train classifiers and evaluate models
python src/evaluate.py
```

### 3. Launch Web Interface

```bash
# Start the Streamlit app
python -m streamlit run src/streamlit_app.py
```

Open your browser to `http://localhost:8501` (or the URL shown in terminal)

## 📊 Detailed Workflow

### Phase 1: Data Preparation
- **Dataset**: Sales call transcripts with binary labels ("won"/"lost")
- **Location**: `data/sales_transcripts.csv`
- **Format**: CSV with columns: `transcript`, `label`

### Phase 2: Baseline Model (`baseline.py`)
```python
# What it does:
1. Loads sales transcript data
2. Splits data into train/test (80/20)
3. Generates embeddings using pre-trained SBERT
4. Trains logistic regression classifier
5. Evaluates performance and saves model
```

**Output**: `models/clf_sbert_base.joblib`

### Phase 3: Contrastive Fine-tuning (`contrastive_finetune_llama.py`)
```python
# What it does:
1. Creates contrastive learning pairs from transcript data
2. Fine-tunes BERT model using contrastive loss
3. Saves fine-tuned model for domain-specific embeddings
```

**Output**: `models/bert_contrastive/` (complete fine-tuned model)

### Phase 4: Evaluation (`evaluate.py`)
```python
# What it does:
1. Loads both SBERT and fine-tuned BERT models
2. Generates embeddings for test data
3. Trains classifiers on embeddings
4. Compares model performance
5. Saves classifiers for web app
```

**Output**: 
- `models/clf_bert_contrastive.joblib`
- Performance comparison metrics

### Phase 5: Web Interface (`streamlit_app.py`)
- Interactive web application for real-time predictions
- Supports both SBERT baseline and BERT-contrastive models
- User-friendly interface for entering sales transcripts

## 🎮 Using the Web Interface

### Model Status Indicators
- ✅ **Green**: Model trained and ready for predictions
- ⚠️ **Yellow**: Model not found, shows training instructions

### Making Predictions
1. **Paste transcript** into the text area
2. **Click "Predict"** button
3. **View results** from both models with confidence scores

### Sample Test Transcripts

**Positive Example (Expected: "won")**:
```
Customer: I've been looking forward to this call. We've been struggling with our current system and really need something more robust. When I saw your demo last week, it looked like exactly what we need. I'd like to move forward with this. Can you send me the contract details?
```

**Negative Example (Expected: "lost")**:
```
Customer: I attended your webinar last month. It was informative, but I'm not sure we really need to make a change. We've been using our current system for three years now, and honestly, it works pretty well for us. We're pretty happy with our current productivity levels.
```

## 🔧 Dependencies

```txt
pandas              # Data manipulation
matplotlib          # Visualization  
sentence-transformers # Pre-trained embeddings
scikit-learn        # Machine learning algorithms
streamlit           # Web interface
joblib              # Model serialization
torch               # Deep learning framework (auto-installed with sentence-transformers)
transformers        # Hugging Face transformers (auto-installed)
```

## 📈 Model Performance

### Evaluation Metrics
Both models are evaluated on:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positive cases  
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy

### Expected Performance
- **SBERT Baseline**: Good general performance using pre-trained embeddings
- **BERT-Contrastive**: Improved domain-specific performance through fine-tuning

## 🛠️ Troubleshooting

### Common Issues

**1. Models not showing as available in web app**
```bash
# Solution: Train the missing models
python src/baseline.py        # For SBERT baseline
python src/evaluate.py        # For BERT-contrastive
```

**2. Import errors**
```bash
# Solution: Install missing dependencies
pip install sentence-transformers streamlit joblib
```

**3. CUDA/GPU issues**
```bash
# Solution: Models work on CPU by default
# GPU acceleration is optional and auto-detected
```

### File Locations
- **Models**: Saved in `models/` directory
- **Classifiers**: `.joblib` files for quick loading
- **Embeddings**: Complete model in `bert_contrastive/` folder

## 🔬 Technical Details

### Contrastive Learning Approach
- **Positive pairs**: Same-label transcript pairs
- **Negative pairs**: Different-label transcript pairs  
- **Loss function**: Contrastive loss to learn discriminative embeddings
- **Architecture**: Fine-tuned BERT-base model

### Model Architecture
```
Input Text → BERT Encoder → [CLS] Token → Classification Head → Prediction
                ↓
            Contrastive Loss (during training)
```

## 🎯 Use Cases

### Business Applications
- **Sales team training**: Identify successful conversation patterns
- **Call quality assessment**: Automated scoring of sales calls
- **Pipeline prediction**: Early prediction of deal outcomes
- **Coaching insights**: Understand what drives conversions

### Technical Applications  
- **NLP research**: Compare pre-trained vs. fine-tuned embeddings
- **Contrastive learning**: Domain adaptation for text classification
- **Model comparison**: Baseline vs. specialized model performance

## 🚀 Future Enhancements

### Model Improvements
- [ ] Add more sophisticated architectures (RoBERTa, DeBERTa)
- [ ] Implement few-shot learning approaches
- [ ] Add multi-class prediction (hot, warm, cold leads)
- [ ] Incorporate audio features from call recordings

### Interface Enhancements  
- [ ] Batch prediction upload
- [ ] Confidence threshold settings
- [ ] Model explanation features (SHAP, attention visualizations)
- [ ] Historical prediction tracking

### Deployment Options
- [ ] Docker containerization
- [ ] REST API endpoints
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] Real-time streaming predictions

## 📝 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📞 Support

For questions, issues, or contributions:
- **Issues**: Open a GitHub issue
- **Features**: Submit a feature request
- **Documentation**: Contribute to documentation improvements

---

**Happy predicting! 🎉**
