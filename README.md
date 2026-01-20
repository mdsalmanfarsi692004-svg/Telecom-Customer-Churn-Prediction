# 📊 Telecom Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)

## 🚀 Project Overview
In the highly competitive telecommunications sector, acquiring a new customer is **5-7x more expensive** than retaining an existing one. This project addresses the challenge of customer attrition by developing a Machine Learning framework capable of identifying high-risk customers with precision.

The system utilizes a **Random Forest Classifier** to analyze behavioral data—such as contract duration, monthly charges, and tenure—to forecast churn probability.

---

## 📸 Screenshots

### 🏠 Home Page
<img width="1910" height="918" alt="home_page" src="https://github.com/user-attachments/assets/b23996b4-5911-4e4e-9e99-fe66bd22de67" />

### 🟢 Safe (No Churn)
<img width="1890" height="908" alt="no_churn" src="https://github.com/user-attachments/assets/8f152825-5f6b-48e3-913f-471f4973ce67" />

### 🔴 High Risk (Churn Alert)
<img width="1887" height="910" alt="churn" src="https://github.com/user-attachments/assets/841f43b5-15a8-4da9-9a1e-098c8b20d838" />

---

## 🛠️ Tech Stack
* **Python**: Core programming language for data processing.
* **Scikit-Learn**: Used for building and tuning the classification model.
* **Streamlit**: Framework used to deploy the predictive model as an interactive web application.
* **Pandas & NumPy**: For Data Preprocessing and ETL pipelines.

---

## 🌟 Key Features
* **Real-time Prediction**: User inputs customer details (Gender, Age, Tenure, Monthly Charges) and receives instant risk assessments.
* **Actionable Insights**:
    * 🔴 **High Risk (Churn):** Identifies users likely to leave and suggests retention discounts.
    * 🟢 **Safe (No Churn):** Identifies loyal customers and suggests upselling premium features.
* **Interactive Interface**: A user-friendly dashboard built with Streamlit for data-driven decision-making.

---

## 📂 Project Structure
text
├── images/             # Screenshots of the application
├── notebooks/          # Jupyter Notebook with ML model training
├── reports/            # Project PDF report and PPT presentation
├── app.py              # Main Streamlit application
├── churn_data.csv      # Dataset used for the project
├── model.pkl           # Trained Random Forest model
├── scaler.pkl          # Data scaling object
└── requirements.txt    # Python dependencies

## 📊 Methodology
1) Data Preprocessing: Handled missing values, scaled numerical features, and applied One-Hot Encoding to categorical variables.

2) Model Selection: Trained multiple algorithms and selected the Random Forest Classifier for its superior performance in handling non-linear data.

3) Deployment: The final model is integrated into a Streamlit app (app.py) for live demonstrations.

## 💻 How to Run Locally
1) Clone the repository:
   git clone [https://github.com/mdsalmanfarsi692004-svg/Telecom-Customer-Churn-Prediction.git](https://github.com/mdsalmanfarsi692004-svg/Telecom-Customer-Churn-Prediction.git)
2) Navigate to the directory:
   cd Telecom-Customer-Churn-Prediction
3) Install dependencies:
   pip install -r requirements.txt
4) Run the application:
   streamlit run app.py

## 🔗 **Live App:** [Click Here to View](https://telecom-customer-churn-prediction.streamlit.app/)

👨‍💻 Developed by
Md Salman Farsi Elevate Labs Internship Project
