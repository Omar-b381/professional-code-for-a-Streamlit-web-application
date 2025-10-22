import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model
# Make sure 'rf_churn_model.pkl' is in the same directory
try:
    model = joblib.load('rf_churn_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'rf_churn_model.pkl' not found.")
    st.stop() # Stop the app if the model isn't found

# 2. Define the feature order (CRITICAL)
# This MUST be the same order as the data used for training
# Based on our previous discussion:
MODEL_COLUMNS = [
    'Complaints',
    'Status',
    'Seconds_of_Use',
    'Subscription_Length',
    'Frequency_of_use',
    'Call_Failure',
    'Distinct_Called_Numbers',
    'Customer_Value',
    'Frequency_of_SMS',
    'Age_Group',
    'Age',
    'Charge_Amount',
    'Tariff_Plan'
]

# 3. Set up the Streamlit page
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🚀", layout="wide")
st.title("🚀 Proactive Customer Churn Prediction")
st.write("""
This app uses a Random Forest model (Accuracy: 98.6%) to predict if a customer 
is likely to churn (leave the service). Please input the customer's details 
in the sidebar to get a prediction.
""")

# 4. Create the input widgets in the sidebar
st.sidebar.header("Customer Input Features")

# --- Helper function to create inputs ---
def user_inputs():
    st.sidebar.markdown("### Customer Experience")
    complaints = st.sidebar.slider("Complaints (0 = No, 1 = Yes)", 0, 1, 0)
    status = st.sidebar.selectbox("Customer Status (1 = Active, 2 = Non-active)", [1, 2])
    call_failure = st.sidebar.number_input("Number of Call Failures", min_value=0, value=0)

    st.sidebar.markdown("### Customer Usage")
    seconds_of_use = st.sidebar.number_input("Total Seconds of Use", min_value=0.0, value=150.0)
    frequency_of_use = st.sidebar.number_input("Total Number of Calls", min_value=0, value=5)
    distinct_called_numbers = st.sidebar.number_input("Distinct Called Numbers", min_value=0, value=4)
    frequency_of_sms = st.sidebar.number_input("Total Number of SMS", min_value=0, value=0)

    st.sidebar.markdown("### Customer Value & Plan")
    subscription_length = st.sidebar.number_input("Subscription Length (months)", min_value=1, value=3)
    customer_value = st.sidebar.number_input("Customer Value (e.g., $)", min_value=0.0, value=200.0)
    # Based on data dictionary: "ordinal attribute (0: lowest amount, 9: highest amount)"
    charge_amount = st.sidebar.slider("Charge Amount (Category)", 0, 9, 1)
    # Based on data dictionary: "binary (1: Pay as you go, 2: contractual)"
    tariff_plan = st.sidebar.selectbox("Tariff Plan", [1, 2])

    st.sidebar.markdown("### Customer Demographics")
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
    # Based on data dictionary: "ordinal attribute (1: younger age, 5: older age)"
    age_group = st.sidebar.slider("Age Group (Category)", 1, 5, 2)

    # Put all inputs into a dictionary
    data = {
        'Complaints': complaints,
        'Status': status,
        'Seconds_of_Use': seconds_of_use,
        'Subscription_Length': subscription_length,
        'Frequency_of_use': frequency_of_use,
        'Call_Failure': call_failure,
        'Distinct_Called_Numbers': distinct_called_numbers,
        'Customer_Value': customer_value,
        'Frequency_of_SMS': frequency_of_sms,
        'Age_Group': age_group,
        'Age': age,
        'Charge_Amount': charge_amount,
        'Tariff_Plan': tariff_plan
    }
    
    # Convert dictionary to DataFrame with the correct column order
    input_df = pd.DataFrame([data])
    input_df = input_df[MODEL_COLUMNS] # Ensure order
    return input_df

# Get user input
input_df = user_inputs()

# 5. Display the input data (optional)
st.subheader("Customer Data Input:")
st.dataframe(input_df)

# 6. Create a prediction button
if st.button("Predict Customer Churn", key="predict_button"):
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        churn_prob = probability[0][1] * 100
        st.error(f"Prediction: **Customer will CHURN** (Probability: {churn_prob:.2f}%) 😡", icon="🚨")
        st.warning("Recommendation: This customer is at high risk. Escalate to the retention team immediately.")
    else:
        no_churn_prob = probability[0][0] * 100
        st.success(f"Prediction: **Customer will STAY** (Probability: {no_churn_prob:.2f}%) 😊", icon="✅")
        st.info("Recommendation: Customer is loyal. Consider offering a rewards or loyalty bonus.")