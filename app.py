from flask import Flask, render_template, request
import joblib
import pandas as pd

from predict import (
    predict_churn,
    model,
    label_encoders,
    feature_names
)
from database import save_prediction,create_database
from utils import (
    DATASET_PATH,
    validate_input,
    get_dataset_stats,
    get_analytics_data
)

app = Flask(__name__)
create_database()
# Load model and preprocessing pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")


# ===========================
# Home Page
# ===========================
@app.route("/")
def home():
    return render_template("index.html")


# ===========================
# Dashboard
# ===========================
@app.route("/dashboard")
def dashboard():

    tenure_filter = request.args.get("tenure", "all")

    df = pd.read_csv(DATASET_PATH)

    if tenure_filter != "all":

        start, end = map(int, tenure_filter.split("-"))

        df = df[
            (df["tenure"] >= start) &
            (df["tenure"] <= end)
        ]

    stats = get_dataset_stats(df)

    analytics = get_analytics_data(
        df,
        predictor=predict_churn
    )

    return render_template(
        "dashboard.html",
        stats=stats,
        analytics=analytics,
        selected_tenure=tenure_filter
    )
# ===========================
# All Customers
# ===========================

@app.route("/customers")
def customers():

    df = pd.read_csv(DATASET_PATH)

    customers_data = []

    # Get preprocessing information
    label_encoders = pipeline["label_encoders"]
    feature_names = pipeline["feature_names"]

    # Prepare customer data
    prediction_df = df[feature_names].copy()

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for col in numeric_columns:
        if col in prediction_df.columns:
            prediction_df[col] = pd.to_numeric(
                prediction_df[col],
                errors="coerce"
            )
    prediction_df = prediction_df.fillna(0)

    # Apply the same encoders used during training
    for column, encoder in label_encoders.items():

        if column in prediction_df.columns:

            prediction_df[column] = encoder.transform(
                prediction_df[column].astype(str)
            )

    # Keep the same feature order
    prediction_df = prediction_df[feature_names]

    # Predict all customers
    probabilities = model.predict_proba(
        prediction_df
    )[:, 1] * 100

    # Create customer records
    for i, probability in enumerate(probabilities):

        probability = round(float(probability), 2)

        if probability >= 70:
            risk = "High"
        elif probability >= 40:
            risk = "Medium"
        else:
            risk = "Low"

        customers_data.append({
            "customerID": str(df.iloc[i]["customerID"]),
            "contract": str(df.iloc[i]["Contract"]),
            "tenure": int(df.iloc[i]["tenure"]),
            "monthlyCharges": round(
                float(df.iloc[i]["MonthlyCharges"]),
                2
            ),
            "churn": str(df.iloc[i]["Churn"]),
            "risk": risk,
            "probability": probability
        })

    return render_template(
        "customers.html",
        customers=customers_data
    )


## ===========================
# Prediction Page
# ===========================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None
    error = None

    if request.method == "POST":

        # Get form data
        form_data = request.form.to_dict()

        # =========================================
        # Convert gender
        # =========================================

        form_data["gender"] = {
            "male": "Male",
            "female": "Female"
        }.get(
            form_data.get("gender"),
            form_data.get("gender")
        )

        # =========================================
        # Convert HTML form fields
        # =========================================

        form_data["SeniorCitizen"] = (
            1 if form_data.get("senior") == "yes" else 0
        )

        form_data["Partner"] = (
            "Yes"
            if form_data.get("partner") == "yes"
            else "No"
        )

        form_data["Dependents"] = (
            "Yes"
            if form_data.get("dependents") == "yes"
            else "No"
        )

        form_data["Contract"] = {
            "month": "Month-to-month",
            "one": "One year",
            "two": "Two year"
        }.get(
            form_data.get("contract"),
            "Month-to-month"
        )

        form_data["MonthlyCharges"] = form_data.get(
            "monthly",
            "0"
        )

        form_data["TotalCharges"] = form_data.get(
            "total",
            "0"
        )

        form_data["PaymentMethod"] = {
            "electronic": "Electronic check",
            "mailed": "Mailed check",
            "auto_card": "Credit card (automatic)",
            "auto_bank": "Bank transfer (automatic)"
        }.get(
            form_data.get("payment"),
            "Electronic check"
        )

        form_data["PaperlessBilling"] = (
            "Yes"
            if form_data.get("paperless") == "yes"
            else "No"
        )

        form_data["InternetService"] = {
            "fiber": "Fiber optic",
            "dsl": "DSL",
            "none": "No"
        }.get(
            form_data.get("internet"),
            "No"
        )

        form_data["OnlineSecurity"] = (
            "Yes"
            if form_data.get("onlinesecurity") == "yes"
            else "No"
        )

        form_data["TechSupport"] = (
            "Yes"
            if form_data.get("support") == "yes"
            else "No"
        )

        form_data["StreamingTV"] = (
            "Yes"
            if form_data.get("streamingtv") == "yes"
            else "No"
        )

        # =========================================
        # Fields not present in current form
        # =========================================

        form_data["PhoneService"] = "Yes"
        form_data["MultipleLines"] = "No phone service"
        form_data["OnlineBackup"] = "No"
        form_data["DeviceProtection"] = "No"
        form_data["StreamingMovies"] = "No"

        # =========================================
        # New customer ID
        # =========================================

        form_data["customerID"] = "NEW_CUSTOMER"

        # =========================================
        # Validate
        # =========================================

        valid, response = validate_input(form_data)

        if valid:

            try:

                # =====================================
                # Predict
                # =====================================

                result = predict_churn(response)

                print("Prediction result:", result)

                # =====================================
                # Save prediction to SQLite
                # =====================================

                if result["success"]:

                    print("Saving prediction to database...")

                    save_prediction(
                        customer_id=response.get(
                            "customerID",
                            "NEW_CUSTOMER"
                        ),
                        gender=response.get(
                            "gender",
                            "Unknown"
                        ),
                        tenure=int(
                            response.get(
                                "tenure",
                                0
                            )
                        ),
                        monthly_charges=float(
                            response.get(
                                "MonthlyCharges",
                                0
                            )
                        ),
                        prediction=result["prediction"],
                        confidence=result["confidence"]
                    )

                    print(
                        "Prediction saved successfully!"
                    )

            except Exception as e:

                error = str(e)

                print("ERROR:", error)

        else:

            error = response

            print(
                "VALIDATION ERROR:",
                response
            )

    # =========================================
    # ALWAYS return the prediction page
    # =========================================

    return render_template(
        "predict.html",
        result=result,
        error=error
    )
# ===========================
# Analytics
# ===========================
@app.route("/analytics")
def analytics():

    tenure_filter = request.args.get("tenure", "all")

    df = pd.read_csv(DATASET_PATH)

    if tenure_filter != "all":

        start, end = map(
            int,
            tenure_filter.split("-")
        )

        df = df[
            (df["tenure"] >= start) &
            (df["tenure"] <= end)
        ]

    analytics_data = get_analytics_data(
        df=df
    )

    return render_template(
        "analytics.html",
        analytics=analytics_data,
        selected_tenure=tenure_filter
    )


# ===========================
# About
# ===========================
@app.route("/about")
def about():
    return render_template("about.html")


# ===========================
# Run Application
# ===========================
if __name__ == "__main__":
    app.run(debug=True)