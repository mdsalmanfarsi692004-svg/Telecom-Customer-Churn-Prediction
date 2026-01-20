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

# --- Custom CSS (Styles for Matching the Image) ---
st.markdown("""
    <style>
    /* 1. Global Center Alignment */
    h1, h2, h3, h4, h5, h6, p, div, span {
        text-align: center;
    }

    /* 2. Sidebar Alignment */
    [data-testid="stSidebar"] {
        text-align: center !important;
    }
    
    /* 3. Radio Button Center Fix */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center !important;
        gap: 15px;
        width: 100%;
    }
    
    /* 4. Button Styling */
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 45px;
    }
    .stButton button:hover {
        background-color: #ce3b3b;
    }
    
    /* 5. Footer Fix */
    .footer-text {
        width: 100%;
        text-align: center;
        font-size: 11px !important;
        white-space: nowrap !important;
        color: grey;
        margin-top: 20px;
    }
    
    /* 6. Result Card Styling (Exact Match to Image) */
    .result-card {
        background-color: #1E1E1E; /* Dark Card Background */
        border: 1px solid #333;
        border-radius: 10px;
        padding: 25px;
        margin-top: 15px;
        text-align: center; /* Text Center */
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .result-title-churn {
        color: #FF4B4B; /* Red Text */
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-title-safe {
        color: #00CC96; /* Green Text */
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .recommendation-title {
        color: white;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    
    /* Bullets ko Center mein rakhne ka trick */
    ul {
        display: inline-block; /* Block ko center karega */
        text-align: left; /* Text ko left karega taaki bullets align rahein */
        color: #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("model.pkl")
        return scaler, model
    except FileNotFoundError:
        return None, None

scaler, model = load_assets()

# --- HEADER ---
st.markdown("<h1>📊 Customer Churn Prediction 📊</h1>", unsafe_allow_html=True)
st.markdown("<h3>Identify At-Risk Customers And Improve Retention Strategies</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="140">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.write("") 

st.sidebar.header("📝 User Input Features")
st.sidebar.write("Adjust The Values Below:") 

if scaler is None or model is None:
    st.error("🚨 Error: Files not found!")
    st.stop()

# Inputs
st.sidebar.markdown("<b>Select Gender</b>", unsafe_allow_html=True)
gender = st.sidebar.radio("Select Gender", ["Male", "Female"], horizontal=True, label_visibility="collapsed")

age = st.sidebar.slider("Customer Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (Months)", 0, 130, 12)
monthlycharge = st.sidebar.number_input("Monthly Charge ($)", 0.0, 500.0, 50.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="footer-text">Made with ❤️ by Md Salman Farsi</div>', unsafe_allow_html=True)


# --- MAIN CONTENT ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h4>Review Input Data</h4>", unsafe_allow_html=True)
    
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Monthly Charge': [monthlycharge]
    })
    
    st.dataframe(input_data, hide_index=True, use_container_width=True)
    st.write("") 
    
    # --- BUTTON ---
    b1, b2, b3 = st.columns([2, 1.5, 2]) 
    with b2:
        predictbutton = st.button("🚀 Predict Churn Status", use_container_width=True)

    # --- Prediction Logic & Result Display ---
    if predictbutton:
        with st.spinner("Analyzing..."):
            gender_selected = 1 if gender == "Female" else 0
            x = [age, gender_selected, tenure, monthlycharge]
            x1 = np.array(x).reshape(1, -1)
            x_array = scaler.transform(x1)
            prediction = model.predict(x_array)[0]
            
            if tenure > 50:
                prediction = 0
            
            st.write("") 

            # --- RESULT SECTION (Exact Match) ---
            if prediction == 1:
                # 1. Red Alert Bar
                st.error("⚠️ Prediction: CHURN (YES)")
                
                # 2. Dark Card (Centered)
                st.markdown("""
                <div class="result-card">
                    <div class="result-title-churn">Analysis: Customer Likely To LEAVE.</div>
                    <div class="recommendation-title">Recommended Actions:</div>
                    <ul>
                        <li>Offer A Retention Discount Immediately.</li>
                        <li>Reach Out For Feedback On Service Issues.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 1. Green Alert Bar
                st.success("✅ Prediction: NO CHURN")
                
                # 2. Dark Card (Centered)
                st.markdown("""
                <div class="result-card">
                    <div class="result-title-safe">Analysis: Customer Likely To STAY.</div>
                    <div class="recommendation-title">Recommendation:</div>
                    <ul>
                        <li>Continue Providing Excellent Service.</li>
                        <li>Consider Upselling Premium Features.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# --- Bottom Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey; font-size: 12px;'>Developed for Elevate Labs Internship Project</div>", unsafe_allow_html=True)