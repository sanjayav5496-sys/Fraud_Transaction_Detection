📘 Project Overview

This repository contains code, notebooks, and artifacts to build a Fraud Transaction Detection system. The objective is to classify financial transactions as fraudulent or genuine using a reproducible machine learning pipeline:

Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction

This README describes everything required to run, reproduce, extend, and deploy the project.

🔑 Key Features

End-to-end ML pipeline (preprocessing → training → inference)

Multiple model experiments (Logistic Regression, Random Forest, XGBoost)

Handling of class imbalance (SMOTE / undersampling options)

Model serialization (.pkl) for fast predictions

Jupyter notebooks for EDA & experiments

Clear instructions to keep private data (CSV) off the repo

🧩 Data (How to prepare / where to put it)

Keep the real dataset private and place it locally in data/ (e.g., data/transactions.csv).

If you need example data for demos, create a small data/sample_transactions.csv with a few sanitized rows and commit that.

Expected columns (example):

transaction_id,step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFraud


Adapt to your dataset (the scripts include robust loading logic).

🚀 How to Run
1) Quick predict (use saved model)

If models/final_model.pkl exists:

python src/prediction.py --input data/sample_input.json


prediction.py loads the model, applies preprocessing pipelines, and prints/classifies incoming transactions.

2) Train models (from raw CSV)

To train and evaluate models:

python src/model_train.py --data-path data/transactions.csv --out models/


This script:

Loads CSV

Applies preprocessing & feature engineering

Trains several models (configurable)

Evaluates using stratified CV / holdout set

Saves the best model to models/final_model.pkl

Writes metrics to results/classification_report.txt

3) Run EDA notebook

Open notebooks/EDA.ipynb and run the cells to explore distributions, imbalance and correlations.

🧪 Preprocessing & Feature Engineering (what the code does)

src/data_preprocessing.py and src/feature_engineering.py handle:

Missing value imputation

Timestamp extraction (hour/day features)

One-hot / target encoding for categorical vars

Scaling of numeric features (StandardScaler / MinMax)

Handling imbalance: SMOTE / undersampling options

Optional custom features like transaction frequency, aggregated sender/receiver stats

🧠 Models Included

Logistic Regression (baseline)

Random Forest Classifier

XGBoost Classifier

(Optional) LightGBM, Neural Nets

Model selection criteria: Precision, Recall (emphasis on Recall for fraud detection), F1-score, ROC-AUC.

📊 Evaluation & Metrics

The training script writes a full classification report and confusion matrix to results/ including:

Accuracy

Precision

Recall (Sensitivity)

F1-score

ROC-AUC

Recommendation: For fraud detection prefer higher Recall on the fraud class (minimize false negatives). Use precision/recall tradeoffs and threshold tuning for deployment.

🔁 Cross Validation & Hyperparameter Tuning

Use stratified K-Fold due to class imbalance.

Hyperparameter tuning can be done with GridSearchCV or RandomizedSearchCV.

Save the best hyperparameters in results/best_params.json.

🧩 Deployment & Real-time Considerations

For real-time inference:

Wrap prediction.py in an API using Flask or FastAPI.

Use the serialized pipeline (preprocessing + model) to ensure consistent transformations.

Implement logging, rate-limiting and circuit-breaker for production APIs.

Consider batching predictions for throughput if volume is high.
🧾 Results & Logging

Save evaluation metrics to results/classification_report.txt.

Save confusion matrix and ROC curve plots to results/.

Example: results/confusion_matrix.png, results/roc_curve.png.


