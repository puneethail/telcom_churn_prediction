# Telecom Customer Churn Prediction

This repository implements a complete machine learning pipeline to predict customer churn for a telecom operator using the “Telco Customer Churn” dataset from Kaggle :contentReference[oaicite:0]{index=0}.

## Dataset  
- Source: [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) :contentReference[oaicite:1]{index=1}  
- Records: 7,043 customers with 21 feature columns (demographics, account information, service usage) and a binary `Churn` label (`Yes`/`No`) :contentReference[oaicite:2]{index=2}.  

## Project Structure  
telcom_churn_prediction/
├── datasets/
│ ├── cleaned.csv
│ ├── Telco_Customer_Churn_Dataset.csv
│ ├── x_test.csv
│ ├── x_train_bal.csv
│ ├── y_train_bal.csv 
│ └── y_test.csv
├── notebooks/
│ ├── 01_loading and EDA.ipynb
│ ├── 02_test_train_split.ipynb
│ ├── 03_feature_selectiom.ipynb
│ ├── 04_modelselection.ipynb
│ ├── 05_model_training.ipynb
│ └── 06_model_evaluation.ipynb
├──churn_predictor.pkl
├── requirements.txt
└── README.md
