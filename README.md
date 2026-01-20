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

---

## 📸 Screenshots

### 🏠 Home Page
<img width="1910" height="918" alt="home_page" src="https://github.com/user-attachments/assets/b23996b4-5911-4e4e-9e99-fe66bd22de67" />

### 🟢 Safe (No Churn)
<img width="1890" height="908" alt="no_churn" src="https://github.com/user-attachments/assets/8f152825-5f6b-48e3-913f-471f4973ce67" />

### 🔴 High Risk (Churn Alert)
<img width="1887" height="910" alt="churn" src="https://github.com/user-attachments/assets/841f43b5-15a8-4da9-9a1e-098c8b20d838" />

---

## 📊 Methodology
1. **Data Preprocessing**: Handled missing values, scaled numerical features, and applied One-Hot Encoding to categorical variables.
2. **Model Selection**: Trained multiple algorithms and selected the **Random Forest Classifier** for its superior performance in handling non-linear data.
3. **Deployment**: The final model is integrated into a Streamlit app (`App.py`) for live demonstrations.

## 🎥 Live Demo
▶️ **[Watch Project Video](https://drive.google.com/file/d/1cvDgQc5vMkSRffzcHhDd8Z8I_4n93s5y/view?usp=drivesdk)**

---

## 💻 How to Run Locally

Run the following commands in your terminal to set up the project:

```bash
# Clone the repository
git clone [https://github.com/mdsalmanfarsi692004-svg/Telecom-Customer-Churn-Prediction.git](https://github.com/mdsalmanfarsi692004-svg/Telecom-Customer-Churn-Prediction.git)

# Navigate to the directory
cd Telecom-Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run App.py

Developed by Md Salman Farsi for Elevate Labs
