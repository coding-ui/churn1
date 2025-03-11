# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv("E:\\zproject\\chrun\\chrunnew\\customer-churn-prediction\\backend\\dataset.csv")

# Drop customerID column as it is not required for modelling
df = df.drop(columns=["customerID"])

# Replace empty values in TotalCharges with 0 and convert to float
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

# Label encoding of target column
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
# df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})

# Label encoding of categorical features
object_columns = df.select_dtypes(include="object").columns
encoders = {}

for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save the encoders to a pickle file
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Split the features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)

# Save the trained model as a pickle file
model_data = {"model": rfc, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# Load the saved model and feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Example input data for prediction
input_data = {

  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 6,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 91,
  "TotalCharges": 531,

}
# Convert input data to DataFrame
input_data_df = pd.DataFrame([input_data])

# Load encoders and transform categorical features
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

# Display result
result = "Churn" if prediction[0] == 1 else "No Churn"
probability_percent = pred_prob[0][1] * 100  # Probability of Churn in percentage

print(f"Result: {result}")
print(f"Probability of Churn: {probability_percent:.2f}%")