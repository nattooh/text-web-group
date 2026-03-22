# AI vs Human Text Classification

## Overview
This project aims to distinguish between human-written and AI-generated text using a range of machine learning and deep learning models. The task is formulated as a binary text classification problem.

We establish a strong baseline using TF-IDF and Logistic Regression, and extend the analysis to neural models including MLP, BiLSTM, and BERT.

---

## Dataset

The dataset used is **AI Vs Human Text** from Kaggle:

Shayan Gerami. *AI Vs Human Text*. Dataset. Kaggle.  
https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

- ~500,000 samples
- Binary labels:
  - `0` → Human-written
  - `1` → AI-generated
- Column structure:
  - `text`: raw text
  - `generated`: label

⚠️ The dataset is not included in this repository due to its large size.

### Download Instructions
1. Download from Kaggle:
   https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
2. Place the file in:
    dataset/AI_Human.csv


---

## Preprocessing

The dataset is preprocessed using `preprocess.py`:

Steps:
- Chunk-based loading for large file handling
- Removal of missing values
- Light text cleaning:
  - whitespace normalization
  - null character removal
- Label cleaning (ensure binary values)
- Removal of empty text samples
- Deduplication based on text
- Stratified train/validation/test split (80/10/10)

Output files:    
processed_data/
├── full_cleaned.csv
├── train.csv
├── validation.csv
├── test.csv