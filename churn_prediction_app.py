import streamlit as st
import pandas as pd
import pickle

# Load the saved model and encoders
# @st.cache(allow_output_mutation=True)
def load_model_and_encoders():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], encoders, model_data["features_names"]

model, encoders, feature_names = load_model_and_encoders()

# Hide anchor link for all headers
st.markdown(
    """
    <style>
        h1 a, h2 a, h3 a {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)






# Streamlit app
st.title("Customer Churn Prediction")

# Input fields for user data
st.header("Enter Customer Details")

# Row 1
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male","Female" ])
import streamlit as st

with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])

    if SeniorCitizen == "Yes":
        SeniorCitizen = 1
    else:
        SeniorCitizen = 0

        
     
with col3:
    Partner = st.selectbox("Partner", ["Yes", "No"])

# Row 2
col4, col5, col6 = st.columns(3)
with col4:
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
with col5:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    
with col6:
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])

# Row 3
col7, col8, col9 = st.columns(3)
with col7:
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
with col8:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with col9:
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

# Row 4
col10, col11, col12 = st.columns(3)
with col10:
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
with col11:
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
with col12:
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

# Row 5
col13, col14, col15 = st.columns(3)
with col13:
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
with col14:
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
with col15:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

# Row 6
col16, col17, col18 = st.columns(3)
with col16:
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
with col17:
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
with col18:
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)

# Row 7 (Full width)
col19,col20,col21 = st.columns(3)
with col19:
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)

# Create a dictionary for the input data
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical features using the loaded encoders
for column in input_df.columns:
    if column in encoders:
        input_df[column] = encoders[column].transform(input_df[column])

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction result
    churn_status = "Churn" if prediction[0] == 1 else "No Churn"
    st.subheader("Prediction Result")
    st.write(f"**Churn Status:** {churn_status}")
    st.write(f"**Probability of Churn:** {prediction_proba[0][1] * 100:.2f}%")