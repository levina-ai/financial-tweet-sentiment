# Twitter Financial Sentiment Classification

This repository benchmarks multiple NLP approaches for **financial tweet sentiment classification** with three classes:
- **0 — Bearish**
- **1 — Bullish**
- **2 — Neutral**

It is part of a **Text Mining / NLP course project** and focuses on comparing classic ML baselines vs transformer-based methods under a consistent evaluation setup.
The final selected model achieved the following performance on the **out-of-sample test set**:

- **Precision (macro):** 0.859965  
- **Recall (macro):** 0.884620  
- **Accuracy:** 0.897822  
- **F1-score (macro):** 0.871604  

In the course-wide benchmark, the final submission ranked **3rd overall**.

## What’s inside

### Approaches compared
**Classic + vectorization**
- Bag-of-Words (CountVectorizer) + Logistic Regression / LinearSVC / Random Forest  
- TF-IDF (1–2 grams) + Logistic Regression / LinearSVC / Random Forest / XGBoost  
- Word2Vec (mean pooled tweet embeddings) + Logistic Regression / LinearSVC / Random Forest

**Transformer encoders (embeddings + sklearn classifier)**
- FinBERT / FinancialBERT / Twitter RoBERTa
- Fine-tuned Twitter RoBERTa and FinancialBERT using Hugging Face `Trainer`
- Mean pooling / CLS pooling experiments
- Logistic Regression / LinearSVC / Random Forest / XGBoost on top of frozen embeddings

**Decoder (prompting)**
- A lightweight instruct model used via a strict prompt template (single-word label output)

### Project notebooks
- `notebooks/01_model_testing.ipynb` — full experimentation: EDA, preprocessing, feature building, model benchmarking, tuning
- `notebooks/02_final_model.ipynb` — final selected model + clean evaluation and conclusions

## Data

Expected files:
- `train.csv` with columns: `text`, `label`
- `test.csv` (optional)
