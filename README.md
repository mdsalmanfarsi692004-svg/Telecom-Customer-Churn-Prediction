# 📊 Telecom Customer Churn Prediction System

## 🚀 Project Overview
In the highly competitive telecommunications sector, acquiring a new customer is **5-7x more expensive** than retaining an existing one. This project addresses the challenge of customer attrition by developing a Machine Learning framework capable of identifying high-risk customers with precision.

The system utilizes a **Random Forest Classifier** to analyze behavioral data—such as contract duration, monthly charges, and tenure—to forecast churn probability.

## 🛠️ Tech Stack
* **Python**: Core programming language for data processing.
* **Scikit-Learn**: Used for building and tuning the classification model.
* **Streamlit**: Framework used to deploy the predictive model as an interactive web application.
* **Pandas & NumPy**: For Data Preprocessing and ETL pipelines.

## 🌟 Key Features
* **Real-time Prediction**: User inputs customer details (Gender, Senior Citizen status, Tenure, Monthly Charges) and receives instant risk assessments.
* **Actionable Insights**:
    * 🔴 **High Risk (Churn):** Identifies users likely to leave and suggests retention discounts.
    * 🟢 **Safe (No Churn):** Identifies loyal customers and suggests upselling premium features or loyalty rewards.
* **Interactive Interface**: A user-friendly dashboard that allows non-technical staff to make data-driven decisions.
  
## 📸 Screenshots
<img width="1912" height="923" alt="No Churn Output" src="https://github.com/user-attachments/assets/0178e36e-4023-4b29-a68b-66bd45db0eb6" />
<img width="1912" height="923" alt="No Churn Output" src="https://github.com/user-attachments/assets/59a4bb44-f43f-439f-811b-d46d6cd1e0ae" />

## 📊 Methodology
1.  **Data Preprocessing**: Handled missing values, scaled numerical features, and applied One-Hot Encoding to categorical variables.
2.  **Model Selection**: Trained multiple algorithms and selected the **Random Forest Classifier** for its superior performance in handling non-linear data.
3.  **Deployment**: The final model is integrated into a Streamlit app (`App.py`) for live demonstrations.

## 📂 How to Run Locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run App.py
