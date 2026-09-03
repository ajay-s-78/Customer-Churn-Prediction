# Customer Churn Prediction System

A Machine Learning web application that predicts customer churn for a telecommunications company and provides customer risk analysis through an interactive Flask dashboard.

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses Machine Learning to predict whether a customer is likely to churn based on their demographic, service, contract, tenure, and billing information.

The application provides:

- Customer churn prediction
- Churn probability and risk level
- Customer analytics dashboard
- Customer-level risk analysis
- Model training and comparison
- Prediction history using SQLite
- Interactive Flask web interface

---

## 🚀 Features

### 🔮 Churn Prediction

Enter customer information and the system predicts:

- Churn / No Churn
- Churn probability
- Risk level
- Recommendation

Risk levels are classified as:

- **High Risk** — 70% or above
- **Medium Risk** — 40% to 69.99%
- **Low Risk** — below 40%

### 📊 Dashboard

The dashboard provides customer and churn analytics with tenure-based filtering.

### 👥 Customer Analysis

The customer page calculates churn probabilities for customers and assigns each customer a risk category.

### 📈 Analytics

The application provides analytics based on:

- Churn distribution
- Contract type
- Internet service
- Payment method
- Customer tenure
- Monthly charges
- Customer demographics
- Service usage

### 💾 Prediction History

Prediction results are stored in a local SQLite database for later reference.

---

## 🤖 Machine Learning Pipeline

The training pipeline performs the following steps:

1. Load the Telco Customer Churn dataset
2. Clean the data
3. Convert numeric fields
4. Handle missing values
5. Remove duplicate records
6. Encode categorical features
7. Split data into training and testing sets
8. Train multiple classification models
9. Evaluate model performance
10. Select the best model using ROC AUC
11. Save the trained model
12. Save the preprocessing information

The project uses an **80/20 stratified train-test split**. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## 🧠 Machine Learning Models

Four classification algorithms are trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
- Classification Report

The best-performing model is selected according to **ROC AUC**. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Joblib

### Data Processing
- Pandas
- NumPy

### Data Visualization
- Matplotlib
- Seaborn

### Frontend
- HTML
- CSS
- JavaScript

### Database
- SQLite

The Python dependencies and versions are defined in `requirements.txt`. :contentReference[oaicite:6]{index=6}

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── customers.html
│   ├── predict.html
│   ├── analytics.html
│   └── about.html
│
├── app.py
├── database.py
├── predict.py
├── train_model.py
├── utils.py
├── model.pkl
├── preprocessing_pipeline.pkl
├── requirements.txt
├── package.json
├── package-lock.json
├── .gitignore
└── README.md
