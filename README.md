# Megaminds-IT-ServicesAssociate-Python-Developer-job-task
A Hybrid Feature-Selection and Autoencoder-Assisted XGBoost Model for Credit Card Fraud Detection


##  Project Overview

Credit card fraud detection is a **highly imbalanced classification problem**, where fraudulent transactions are extremely rare (~0.17%).  
The goal of this project is to build a **robust and reproducible ML pipeline** that can accurately identify fraudulent transactions while minimizing false negatives.

---

##  Dataset

- **Source**: Kaggle – Credit Card Fraud Detection Dataset  
- **Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- **Rows**: 284,807  
- **Columns**: 31  
- **Target Variable**: `Class`  
  - `0` → Normal transaction  
  - `1` → Fraudulent transaction  

 **Note**:  
Due to file size and licensing restrictions, the dataset (`creditcard.csv`) is **NOT included** in this repository.zip file has been added unzip and use it

### Dataset Setup
1. Download the dataset from Kaggle
2. Extract `creditcard.csv`
3. Place it inside:
data/creditcard.csv


---

##  Project Structure

fraud_detection_project/
├── data/
│ └── creditcard.csv # (ignored in GitHub due to its size)
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
├── outputs/
│ ├── plots/ # Saved EDA plots
│ └── metrics/ # Text/CSV summaries
├── src/
│ ├── init.py
│ ├── utils.py
│ ├── preprocess.py
│ ├── feature_selection.py
│ ├── train_baselines.py
│ └── train_proposed.py
├── main.py # Pipeline entry point
├── requirements.txt
|---.gitignore
└── README.md



##  Exploratory Data Analysis (EDA)

EDA is performed using **Jupyter Lab** and includes:

- Dataset structure inspection
- Missing value checks
- Class imbalance analysis
- Transaction amount distribution
- Time feature analysis
- Correlation analysis
- PCA feature distributions

 **Key Insights**
- Dataset is **extremely imbalanced** (~0.17% fraud)
- No missing values
- Fraud transactions tend to have **lower amounts**
- `Time` feature has limited usefulness
- PCA-transformed features dominate

Saved under:
outputs/plots/
outputs/metrics/


##  Machine Learning Pipeline

### Steps:
1. Load dataset
2. Preprocess data (scaling, column handling)
3. Train-validation-test split
4. Hybrid feature selection
5. Train baseline models
6. Train proposed model
7. Evaluate using ROC-AUC

---

##  Models Used

### Baseline Models
- Logistic Regression
- Random Forest
- XGBoost

### Proposed Model
- Neural Network–based model (TensorFlow/Keras)

### Evaluation Metric
- **ROC-AUC Score** (preferred for imbalanced datasets)

---

## Results (AUC Scores)

| Model                | ROC-AUC |
|---------------------|---------|
| Logistic Regression | ~0.95   |
| Random Forest       | ~0.91   |
| XGBoost (Baseline)  | ~0.44   |
| Proposed Model      | ~0.88   |

> Logistic Regression performs exceptionally well due to PCA-transformed features.

---

## How to Run the Project

1️⃣ Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate

```
2️⃣ Install Dependencies

``` bash

pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
python -m ipykernel install --user --name fraud-venv --display-name "Python (fraud-venv)"


```

3️⃣ Run Full ML Pipeline
``` bash
python main.py

```

4️⃣ Run EDA (Jupyter Lab)
``` bash

jupyter lab

```
Open: notebooks/eda.ipynb

Change Kernel to: Python (fraud-venv)

Run: Kernel → Restart & Run All Cells

📁 Outputs
EDA Plots → outputs/plots/*.png

EDA Metrics → outputs/metrics/*.txt

Training Metrics → Printed in terminal

# Megaminds-IT-ServicesAssociate-Python-Developer-job-task
A Hybrid Feature-Selection and Autoencoder-Assisted XGBoost Model for Credit Card Fraud Detection

# Credit Card Fraud Detection – Hybrid ML & Deep Learning Approach

## Project Overview
This project implements a novel fraud detection model named
**HFS-XGBoost-AE**, combining:
- Hybrid Feature Selection
- Autoencoder-based anomaly detection
- XGBoost classification

The goal is to accurately detect rare fraudulent transactions
in highly imbalanced datasets.

---

## Dataset

This project uses the **Credit Card Fraud Detection dataset** from Kaggle.

Due to file size and licensing restrictions, the dataset is **not included** in this repository.

### Download Instructions
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud or i have already uploaded a zip file of size 64MB in data folder
2. Download the ZIP file
3. Extract `creditcard.csv`
4. Place it inside:

---

## Project Structure


fraud_detection_project/
│
├── data/ # Dataset
├── notebooks/ # EDA Notebook
├── src/ # Source Code
├── outputs/ # Models & Plots
├── docs/ # PDF Documents
├── requirements.txt
└── README.md


---

Models Implemented

Logistic Regression

Random Forest

XGBoost

Proposed HFS-XGBoost-AE

Results

The proposed model achieves:

Higher Recall

Higher AUC

Reduced False Positives

Conclusion

The hybrid approach effectively addresses data imbalance
and improves fraud detection performance.

Author - Rajeswari

Prepared for Developer Round – Research & Coding Assessment





