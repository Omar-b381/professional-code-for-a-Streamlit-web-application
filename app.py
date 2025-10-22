import streamlit as st
import pandas as pd

def get_user_inputs():
    """
    Creates sidebar widgets to get input for all 13 features.
    """
    
    st.sidebar.header("Customer Input Features")

    # --- Create inputs for each feature ---
    
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

    # --- Collect data into a dictionary ---
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
    
    # --- Convert to DataFrame ---
    input_df = pd.DataFrame([data])
    
    return input_df

# --- How to use the function in your app ---

# 1. Get the inputs
input_data = get_user_inputs()

# 2. ⚠️ CRITICAL STEP: Re-order columns
# The model MUST receive columns in the exact order it was trained on.
# Create a list of your columns in the correct training order.
TRAINING_COLUMNS_ORDER = [
    'Complaints', 'Status', 'Seconds_of_Use', 'Subscription_Length',
    'Frequency_of_use', 'Call_Failure', 'Distinct_Called_Numbers',
    'Customer_Value', 'Frequency_of_SMS', 'Age_Group', 'Age',
    'Charge_Amount', 'Tariff_Plan'
    # ^^^ THIS ORDER MUST MATCH YOUR MODEL'S TRAINING DATA ^^^
]

# Re-order the DataFrame
try:
    input_data_ordered = input_data[TRAINING_COLUMNS_ORDER]
except KeyError as e:
    st.error(f"Error in column names: {e}. Check your TRAINING_COLUMNS_ORDER list.")
    st.stop()


# 3. Display and Predict
st.subheader("User Input (Ready for Model):")
st.dataframe(input_data_ordered)

if st.button("Predict"):
    # Now you can safely pass 'input_data_ordered' to your model
    # prediction = model.predict(input_data_ordered)
    # st.success(f"Prediction: {prediction[0]}")
    pass # Placeholder for your prediction logic