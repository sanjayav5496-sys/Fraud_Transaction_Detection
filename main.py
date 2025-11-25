import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve

# Load data
data = pd.read_csv('AIML Dataset.csv')  

# One-hot encoding for 'type'
X = data.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)
X = pd.get_dummies(X, columns=['type'])
y = data['isFraud']

# Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Plot 1: Class distribution
plt.figure(figsize=(4,4))
sns.countplot(x=y)
plt.title('Class Distribution (isFraud)')
plt.show()

# Plot 2: Feature distribution by class for key features
for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig']:
    plt.figure(figsize=(6,4))
    sns.histplot(data, x=col, hue='isFraud', bins=50, log_scale=(False, True), element='step')
    plt.title(f'Distribution of {col} by Fraud Class')
    plt.show()

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
val_pred_rf = rf.predict(X_val)
test_pred_rf = rf.predict(X_test)
val_proba_rf = rf.predict_proba(X_val)[:,1]
test_proba_rf = rf.predict_proba(X_test)[:,1]

# XGBoost
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train, y_train)
val_pred_xgb = xgb.predict(X_val)
test_pred_xgb = xgb.predict(X_test)
val_proba_xgb = xgb.predict_proba(X_val)[:,1]
test_proba_xgb = xgb.predict_proba(X_test)[:,1]

def print_metrics(model_name, true, pred, dataset_name):
    print(f"\n{model_name} - {dataset_name} Set:")
    print("Precision:", precision_score(true, pred))
    print("Recall:", recall_score(true, pred))
    print("F1-Score:", f1_score(true, pred))

print_metrics('Random Forest', y_val, val_pred_rf, "Validation")
print_metrics('Random Forest', y_test, test_pred_rf, "Test")
print_metrics('XGBoost', y_val, val_pred_xgb, "Validation")
print_metrics('XGBoost', y_test, test_pred_xgb, "Test")

# Confusion matrix (for Random Forest on test set)
cm = confusion_matrix(y_test, test_pred_rf)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix (Random Forest)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC & Precision-Recall curves (for Random Forest)
fpr, tpr, _ = roc_curve(y_test, test_proba_rf)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title('ROC Curve (Random Forest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, test_proba_rf)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.title('Precision-Recall Curve (Random Forest)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Feature importance (Random Forest)
importances = rf.feature_importances_
feat_names = X_train.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,5))
sns.barplot(x=importances[indices][:10], y=feat_names[indices][:10])
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Feature importance (XGBoost)
xgb_importances = xgb.feature_importances_
indices = np.argsort(xgb_importances)[::-1]
plt.figure(figsize=(8,5))
sns.barplot(x=xgb_importances[indices][:10], y=feat_names[indices][:10])
plt.title('Top 10 Feature Importances (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

import joblib

# For Random Forest (if your model object is named rf)
joblib.dump(rf, 'rf_model.pkl')

# For XGBoost (if your model object is named xgb)
joblib.dump(xgb, 'xgb_model.pkl') 

