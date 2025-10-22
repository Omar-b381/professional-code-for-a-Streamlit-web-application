import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------------------------------
# 1. Load the Saved Model
# --------------------------------------------------------------------------
# This file MUST be in the same folder as your app.py
MODEL_FILE = 'rf_churn_model.pkl'

try:
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_FILE}' not found.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --------------------------------------------------------------------------
# 2. Define the correct Feature Order (CRITICAL)
# --------------------------------------------------------------------------
# This MUST be the exact order of columns your model was trained on.
# We are using the names with spaces, as you provided.
TRAINING_COLUMNS_ORDER = [
    'Call Failure',
    'Complaints',
    'Subscription Length',
    'Charge Amount',
    'Seconds of Use',
    'Frequency of use',
    'Frequency of SMS',
    'Distinct Called Numbers',
    'Age Group',
    'Tariff Plan',
    'Status',
    'Age',
    'Customer Value'
]

# --------------------------------------------------------------------------
# 3. Function to Get User Inputs
# --------------------------------------------------------------------------
def get_user_inputs():
    """
    Creates sidebar widgets to get input for all 13 features.
    Returns a DataFrame with the user's inputs.
    """
    
    st.sidebar.header("Customer Input Features")

    # Create inputs for each feature
    # Note: The dictionary keys MUST match the names in TRAINING_COLUMNS_ORDER
    
    call_failure = st.sidebar.number_input("Number of Call Failures", min_value=0, value=0)
    complaints = st.sidebar.slider("Complaints (0 = No, 1 = Yes)", 0, 1, 0)
    subscription_length = st.sidebar.number_input("Subscription Length (months)", min_value=1, value=3)
    
    # Assuming 'Charge Amount' is the 0-9 ordinal scale
    charge_amount = st.sidebar.slider("Charge Amount (Category)", 0, 9, 1)
    
    seconds_of_use = st.sidebar.number_input("Total Seconds of Use", min_value=0.0, value=150.0, format="%.2f")
    frequency_of_use = st.sidebar.number_input("Total Number of Calls", min_value=0, value=5)
    frequency_of_sms = st.sidebar.number_input("Total Number of SMS", min_value=0, value=0)
    distinct_called_numbers = st.sidebar.number_input("Distinct Called Numbers", min_value=0, value=4)
    
    # Assuming 'Age Group' is the 1-5 ordinal scale
    age_group = st.sidebar.slider("Age Group (Category)", 1, 5, 2)
    
    # Assuming 'Tariff Plan' is 1 or 2
    tariff_plan = st.sidebar.selectbox("Tariff Plan (1 = PayGo, 2 = Contractual)", [1, 2])
    
    # Assuming 'Status' is 1 or 2
    status = st.sidebar.selectbox("Customer Status (1 = Active, 2 = Non-active)", [1, 2])
    
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=25)
    customer_value = st.sidebar.number_input("Customer Value ($)", min_value=0.0, value=200.0, format="%.2f")

    # Collect data into a dictionary
    data = {
        'Call Failure': call_failure,
        'Complaints': complaints,
        'Subscription Length': subscription_length,
        'Charge Amount': charge_amount,
        'Seconds of Use': seconds_of_use,
        'Frequency of use': frequency_of_use,
        'Frequency of SMS': frequency_of_sms,
        'Distinct Called Numbers': distinct_called_numbers,
        'Age Group': age_group,
        'Tariff Plan': tariff_plan,
        'Status': status,
        'Age': age,
        'Customer Value': customer_value
    }
    
    # Convert dictionary to a DataFrame
    input_df = pd.DataFrame([data])
    
    return input_df

# --------------------------------------------------------------------------
# 4. Main Application Interface
# --------------------------------------------------------------------------

# Set up the page configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🚀", layout="wide")

# Title and description
st.title("🚀 Proactive Customer Churn Prediction")
st.write("""
This app uses a Random Forest model (Accuracy: 98.6%) to predict if a customer 
is likely to churn. Please input the customer's details 
in the sidebar to get a prediction.
""")

# --- Get inputs and re-order columns ---
input_df = get_user_inputs()

# Re-order the DataFrame to match the model's training order
try:
    input_data_ordered = input_df[TRAINING_COLUMNS_ORDER]
except KeyError as e:
    st.error(f"Error in column names: {e}. Check your TRAINING_COLUMNS_ORDER list in the code.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while re-ordering columns: {e}")
    st.stop()

# Display the final input data
st.subheader("Customer Data (Ready for Prediction):")
st.dataframe(input_data_ordered)

# --- Prediction Logic ---
if st.button("Predict Customer Churn", key="predict_button"):
    
    try:
        # Make prediction
        prediction = model.predict(input_data_ordered)
        probability = model.predict_proba(input_data_ordered)

        st.subheader("Prediction Result:")
        
        if prediction[0] == 1:
            churn_prob = probability[0][1] * 100
            st.error(f"Prediction: **Customer will CHURN** (Probability: {churn_prob:.2f}%) 😡", icon="🚨")
            st.warning("Recommendation: This customer is at high risk. Escalate to the retention team immediately.")
        else:
            no_churn_prob = probability[0][0] * 100
            st.success(f"Prediction: **Customer will STAY** (Probability: {no_churn_prob:.2f}%) 😊", icon="✅")
            st.info("Recommendation: Customer is loyal. Consider offering a rewards or loyalty bonus.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")