import pandas as pd
import joblib

rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

columns = [
    'step',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'type_CASH_IN',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER'
]

print("Choose Transaction Type:")
print("1. CASH_IN")
print("2. CASH_OUT")
print("3. DEBIT")
print("4. PAYMENT")
print("5. TRANSFER")
trans_choice = int(input("Enter the number (1-5): "))
types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
trans_type = types[trans_choice - 1]

step = int(input("Step (integer, e.g. 1): "))
amount = float(input("Transaction Amount: "))
oldbalanceOrg = float(input("Old Balance (sender): "))
newbalanceOrig = float(input("New Balance (sender): "))
oldbalanceDest = float(input("Old Balance (receiver): "))
newbalanceDest = float(input("New Balance (receiver): "))

input_dict = {
    'step': step,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'type_CASH_IN': 1 if trans_type == 'CASH_IN' else 0,
    'type_CASH_OUT': 1 if trans_type == 'CASH_OUT' else 0,
    'type_DEBIT': 1 if trans_type == 'DEBIT' else 0,
    'type_PAYMENT': 1 if trans_type == 'PAYMENT' else 0,
    'type_TRANSFER': 1 if trans_type == 'TRANSFER' else 0
}

input_df = pd.DataFrame([input_dict], columns=columns)

print("Choose model for prediction:")
print("1. Random Forest")
print("2. XGBoost")
model_choice = int(input("Enter 1 or 2: "))

if model_choice == 1:
    prediction = rf_model.predict(input_df)[0]
    print("\nRandom Forest Result:", "FRAUDULENT 🚨" if prediction == 1 else "NOT Fraudulent ✅")
else:
    prediction = xgb_model.predict(input_df)[0]
    print("\nXGBoost Result:", "FRAUDULENT 🚨" if prediction == 1 else "NOT Fraudulent ✅")


