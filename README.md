# Telecom Customer Churn Prediction

**Predict which telecom customers are likely to churn** by walking through a full ML pipeline—from raw data to a saved production-ready model.

---

## Table of Contents

1. [Motivation](#motivation)  
2. [Dataset](#dataset)  
3. [Pipeline Overview](#pipeline-overview)  
4. [Model Details](#model-details)  
5. [Results](#results)  
6. [Usage](#usage)  
7. [Project Structure](#project-structure)  
8. [License](#license)  

---

## Motivation

Customer acquisition in telecom is **5–10×** more expensive than retention. By forecasting churn early, operators can proactively offer promotions or service improvements to at-risk subscribers, reducing revenue loss and improving customer lifetime value.  

---

## Dataset

- **Source:** [Telco Customer Churn on Kaggle][kaggle]  
- **Records:** 7,043 customers over a one-year period  
- **Features (21):**  
  - **Demographics:** gender, SeniorCitizen, Partner, Dependents  
  - **Account:** tenure, Contract, PaperlessBilling, PaymentMethod  
  - **Services:** InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies  
  - **Charges:** MonthlyCharges, TotalCharges  
- **Target:** `Churn` (Yes / No)  

[kaggle]: https://www.kaggle.com/blastchar/telco-customer-churn

---

## Pipeline Overview

1. **01 – Loading & EDA**  
   - Read raw CSV, inspect distributions, churn rates by feature  
2. **02 – Train/Test Split**  
   - Stratified split (80/20) to preserve churn ratio  
3. **03 – Feature Selection**  
   - Univariate tests + tree-based feature importances → selected top 10  
4. **04 – Model Selection**  
   - Compared `LogisticRegression`, `DecisionTreeClassifier`, **`RandomForestClassifier`**, `XGBClassifier`  
   - **Best performer:** Random Forest (Test ROC-AUC: 0.89)  
5. **05 – Model Training**  
   - `GridSearchCV` over `n_estimators=[100,200]`, `max_depth=[10,20,None]`  
   - Final: `n_estimators=200, max_depth=20`  
6. **06 – Evaluation**  
   - Accuracy, Precision, Recall, F1, ROC curve  

---

## Model Details

We serialize our final model to `churn_predictor.pkl`.  To verify exactly which scikit-learn class you’re using:

```python
import pickle

clf = pickle.load(open("churn_predictor.pkl", "rb"))
print(clf)
# ▶️ e.g. RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

## Project Structure  
telcom_churn_prediction/
├── datasets/
│   ├── Telco_Customer_Churn_Dataset.csv
│   ├── cleaned.csv
│   ├── x_train_bal.csv
│   ├── y_train_bal.csv
│   └── x_test.csv, y_test.csv
├── notebooks/
│   ├── 01_loading and EDA.ipynb
│   ├── 02_test_train_split.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_modelselection.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── churn_predictor.pkl    # Final serialized model
├── requirements.txt
└── README.md

