import joblib
import pandas as pd


# ============================================================
# Load trained model and preprocessing pipeline
# ============================================================

model = joblib.load("model.pkl")
pipeline = joblib.load("preprocessing_pipeline.pkl")

label_encoders = pipeline["label_encoders"]
feature_names = pipeline["feature_names"]


# ============================================================
# Preprocess customer input
# ============================================================

def preprocess_input(customer_data):

    # Create one-row DataFrame
    df = pd.DataFrame([customer_data])

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Replace invalid/missing numeric values
    for column in numeric_columns:

        if column in df.columns:

            df[column] = df[column].fillna(0)

    # --------------------------------------------------------
    # Apply the encoders used during training
    # --------------------------------------------------------

    for column, encoder in label_encoders.items():

        if column not in df.columns:
            continue

        value = str(df[column].iloc[0])

        # Check value against training categories
        if value not in encoder.classes_:

            raise ValueError(
                f"Invalid value for {column}: '{value}'. "
                f"Expected one of: "
                f"{list(encoder.classes_)}"
            )

        df[column] = encoder.transform(
            df[column].astype(str)
        )

    # --------------------------------------------------------
    # Check that all model features exist
    # --------------------------------------------------------

    missing_features = [
        column
        for column in feature_names
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Keep exactly the same feature order as training
    # --------------------------------------------------------

    df = df[feature_names]

    return df


# ============================================================
# Predict customer churn
# ============================================================

def predict_churn(customer_data):

    try:

        print("========================================")
        print("Starting churn prediction...")
        print("Customer data received:")
        print(customer_data)

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_data = preprocess_input(
            customer_data
        )

        print("Preprocessing successful.")

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        prediction = model.predict(
            processed_data
        )[0]

        print("Prediction value:", prediction)

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            processed_data
        )[0]

        # Probability of churn
        churn_probability = float(
            probabilities[1]
        )

        # Probability of staying
        stay_probability = float(
            probabilities[0]
        )

        # Convert churn probability to percentage
        confidence = round(
            churn_probability * 100,
            2
        )

        print(
            "Churn probability:",
            confidence,
            "%"
        )

        print(
            "Stay probability:",
            round(stay_probability * 100, 2),
            "%"
        )

        # ----------------------------------------------------
        # Determine prediction
        # ----------------------------------------------------

        prediction_text = (
            "Yes"
            if prediction == 1
            else "No"
        )

        # ----------------------------------------------------
        # Determine risk level
        # ----------------------------------------------------

        # HIGH RISK
        if confidence >= 70:

            risk = "High"

            recommendation = (
                "Customer has a high churn risk. "
                "Offer discounts, loyalty rewards, "
                "or a retention plan."
            )

        # MEDIUM RISK
        elif confidence >= 40:

            risk = "Medium"

            recommendation = (
                "Customer has a moderate churn risk. "
                "Consider offering a targeted discount, "
                "service upgrade, or proactive support."
            )

        # LOW RISK
        else:

            risk = "Low"

            recommendation = (
                "Customer is likely to stay. "
                "Continue providing quality service "
                "and maintain customer satisfaction."
            )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = {

            "success": True,

            "prediction": prediction_text,

            "confidence": confidence,

            "risk": risk,

            "recommendation": recommendation

        }

        print("Prediction result:")
        print(result)

        print("========================================")

        return result

    # ========================================================
    # Prediction error
    # ========================================================

    except Exception as e:

        print("========================================")
        print("PREDICTION ERROR:")
        print(str(e))
        print("========================================")

        return {

            "success": False,

            "error": str(e)

        }