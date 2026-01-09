import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Elevate Labs | Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Styling & Centering ---
st.markdown("""
    <style>
    /* Center the Main Title */
    h1 {
        text-align: center;
        color: #FFFFFF;
    }
    
    /* Center the Subheader */
    .stMarkdown h3 {
        text-align: center;
    }
    
    /* Center Sidebar Text and Headers */
    [data-testid="stSidebar"] {
        text-align: center;
    }
    
    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ce3b3b;
        transform: translateY(-2px);
        box-shadow: 0px 6px 8px rgba(0,0,0,0.3);
    }
    
    /* Card-like effect for the result */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #262730;
        border: 1px solid #4e4e4e;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model & Scaler ---
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("model.pkl")
        return scaler, model
    except FileNotFoundError:
        return None, None

scaler, model = load_assets()

# --- App Header (Centered) ---
st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction 📊</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>Identify At-Risk Customers And Improve Retention Strategies</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar for Inputs (Centered via CSS) ---
st.sidebar.header("📝 User Input Features")
st.sidebar.write("Adjust The Values Below:") 

if scaler is None or model is None:
    st.error("🚨 Error: 'model.pkl' or 'scaler.pkl' not found. Please check your directory.")
    st.stop()

# Inputs
gender = st.sidebar.radio("Select Gender", ["Male", "Female"], horizontal=True)
age = st.sidebar.slider("Customer Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (Months)", 0, 130, 12)
monthlycharge = st.sidebar.number_input("Monthly Charge ($)", 0.0, 500.0, 50.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ by Md Salman Farsi")

# --- Main Screen Content (Centered Layout) ---
# Using 3 columns to force the content into the middle column
col_spacer1, col_centered, col_spacer2 = st.columns([1, 2, 1])

with col_centered:
    st.markdown("<h4 style='text-align: center;'>Review Input Data</h4>", unsafe_allow_html=True)
    
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Monthly Charge': [monthlycharge]
    })
    
    # Display dataframe (Automatic width fills the centered column)
    st.dataframe(input_data, hide_index=True, use_container_width=True)

    st.write("") # Spacer
    
    # Predict Button
    predictbutton = st.button("🚀 Predict Churn Status")

    # --- Prediction Logic ---
    if predictbutton:
        with st.spinner("Analyzing customer data..."):
            
            # 1. Prepare Data
            gender_selected = 1 if gender == "Female" else 0
            x = [age, gender_selected, tenure, monthlycharge]
            x1 = np.array(x).reshape(1, -1)
            
            # 2. Scale Data
            x_array = scaler.transform(x1)
            
            # 3. Predict Class
            prediction = model.predict(x_array)[0]
            
            # Logic Override: If tenure > 50, force Safe (0)
            if tenure > 50:
                prediction = 0
            
            st.write("") # Spacer

            # --- Display Results Centered ---
            if prediction == 1:
                st.error("⚠️ Prediction: CHURN (YES)")
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #ff4b4b;">Analysis: Customer Likely To LEAVE.</h4>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        <li>Offer A Retention Discount Immediately.</li>
                        <li>Reach Out For Feedback On Service Issues.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ Prediction: NO CHURN")
                st.balloons()
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #00cc96;">Analysis: Customer Likely To STAY.</h4>
                    <p><strong>Recommendation:</strong></p>
                    <ul>
                        <li>Continue Providing Excellent Service.</li>
                        <li>Consider Upselling Premium Features.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 12px;'>Developed for Elevate Labs Internship Project</div>", 
    unsafe_allow_html=True
)