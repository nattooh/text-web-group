# AI vs Human Text Classification

## Overview
This project aims to distinguish between human-written and AI-generated text using a range of machine learning and deep learning models. The task is formulated as a binary text classification problem.

We establish a strong baseline using TF-IDF and Logistic Regression, and extend the analysis to neural models including BiLSTM, BERT, and DistilBERT.

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

The dataset is preprocessed using `preprocess.py`.

### Steps
- Chunk-based loading to efficiently handle large datasets (~500k rows)
- Removal of missing values in `text` and `generated`
- Light text cleaning:
  - whitespace normalization
  - null character removal
- Label cleaning:
  - ensure numeric values
  - filter to binary labels (`0` = Human, `1` = AI)
- Removal of empty text samples
- Deduplication based on `text`

After cleaning, the dataset is used to generate **multiple balanced subsets** for controlled experiments.

---

### Subset Generation

To study the effect of dataset size on model performance, three subsets are created:

- **5,000 samples**
- **20,000 samples**
- **100,000 samples**

Each subset is:
- **Balanced** (50% Human, 50% AI)
- Randomly sampled using a fixed seed (`random_state=42`)
- Independently split into:
  - **80% Train**
  - **10% Validation**
  - **10% Test**
- Stratified by label to preserve class distribution

---


### Usage

To generate all dataset splits:

```bash
python preprocess.py
```

This will:

Load and clean the full dataset
Remove duplicates
Create 5k, 20k, and 100k balanced subsets
Save train/validation/test splits for each subset

Notes
- The subsets are generated using a fixed random seed and are nested, meaning smaller datasets (e.g., 5k) are subsets of larger ones (e.g., 20k, 100k).
- A fixed random seed ensures reproducibility across runs.
- These subsets are used to evaluate how model performance scales with dataset size.