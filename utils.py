# =============================================================================
# utils.py -- Shared Utility Functions
# =============================================================================
# Customer Churn Prediction System -- Phase 3
#
# Provides reusable helper functions used across the application:
#   1. Dataset loading and statistics generation (for Dashboard/Analytics)
#   2. Input validation and sanitization (for Prediction form)
#   3. Retention recommendation engine (based on prediction + risk factors)
#   4. Chart data generators (for Dashboard and Analytics pages)
#   5. Feature importance extraction (for Analytics page)
#
# Used by: app.py, predict.py
# =============================================================================

import os
import pandas as pd
import numpy as np


# =============================================================================
# Constants -- Valid values for each categorical input field
# =============================================================================
# These must match the categories the model was trained on.
# Used for input validation and for populating form dropdowns.
VALID_VALUES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# The 19 features expected by the model, in the exact training order
FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Dataset path
DATASET_PATH = os.path.join("dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =============================================================================
# Input Validation
# =============================================================================
def validate_input(form_data: dict) -> tuple:
    """
    Validate and sanitize raw form data from the prediction page.

    Checks:
      - All 19 required fields are present
      - Categorical fields contain valid values
      - Numeric fields (tenure, MonthlyCharges, TotalCharges) are valid numbers
      - Numeric ranges are sensible (non-negative, within expected bounds)

    Args:
        form_data: Dictionary of form field values (from request.form or JSON).

    Returns:
        tuple: (is_valid: bool, cleaned_data: dict or error_message: str)
               If valid, cleaned_data contains properly typed values.
               If invalid, returns the error message string.
    """
    errors = []
    cleaned = {}

    # --- Validate categorical fields ---
    for field, valid_options in VALID_VALUES.items():
        value = form_data.get(field, "").strip() if isinstance(form_data.get(field), str) else form_data.get(field)

        if field == "SeniorCitizen":
            # SeniorCitizen is 0/1 integer, may come as string from form
            try:
                int_val = int(value)
                if int_val not in [0, 1]:
                    errors.append(f"{field}: must be 0 or 1, got '{value}'")
                else:
                    cleaned[field] = int_val
            except (ValueError, TypeError):
                errors.append(f"{field}: must be 0 or 1, got '{value}'")
        else:
            if value not in valid_options:
                errors.append(
                    f"{field}: invalid value '{value}'. "
                    f"Expected one of: {valid_options}"
                )
            else:
                cleaned[field] = value

    # --- Validate numeric fields ---
    numeric_fields = {
        "tenure": (0, 120, "months"),        # 0 to 10 years
        "MonthlyCharges": (0, 500, "USD"),    # reasonable monthly range
        "TotalCharges": (0, 100000, "USD"),   # reasonable total range
    }

    for field, (min_val, max_val, unit) in numeric_fields.items():
        raw_value = form_data.get(field, "")
        try:
            num_val = float(raw_value)
            if num_val < min_val:
                errors.append(f"{field}: cannot be negative (got {num_val})")
            elif num_val > max_val:
                errors.append(
                    f"{field}: value {num_val} {unit} seems unrealistic "
                    f"(max expected: {max_val})"
                )
            else:
                # tenure should be integer
                if field == "tenure":
                    cleaned[field] = int(num_val)
                else:
                    cleaned[field] = round(num_val, 2)
        except (ValueError, TypeError):
            errors.append(f"{field}: must be a valid number, got '{raw_value}'")

    if errors:
        return False, "; ".join(errors)

    return True, cleaned


# =============================================================================
# Retention Recommendations
# =============================================================================
def generate_recommendations(customer_data: dict, confidence: float) -> list:
    """
    Generate personalized retention recommendations based on the customer's
    profile and churn risk factors.

    The recommendation engine analyzes key churn drivers identified during
    model training and suggests actionable retention strategies.

    Args:
        customer_data: Cleaned customer data dictionary.
        confidence: Churn probability (0.0 to 1.0).

    Returns:
        list: List of recommendation strings, ordered by priority.
    """
    recommendations = []

    # --- Contract-based recommendations ---
    contract = customer_data.get("Contract", "")
    if contract == "Month-to-month":
        recommendations.append(
            "Offer a discounted annual or two-year contract to increase "
            "commitment and reduce month-to-month churn risk."
        )

    # --- Tenure-based recommendations ---
    tenure = customer_data.get("tenure", 0)
    if tenure <= 6:
        recommendations.append(
            "New customer detected (tenure <= 6 months). Activate an "
            "onboarding engagement program with welcome offers and "
            "dedicated support to build early loyalty."
        )
    elif tenure <= 12:
        recommendations.append(
            "Customer is in the critical first-year window. Consider a "
            "loyalty bonus or service upgrade to reinforce retention."
        )

    # --- Internet service recommendations ---
    internet = customer_data.get("InternetService", "")
    if internet == "Fiber optic":
        # Fiber optic customers have higher churn in the dataset
        online_security = customer_data.get("OnlineSecurity", "No")
        tech_support = customer_data.get("TechSupport", "No")

        if online_security == "No":
            recommendations.append(
                "Add complimentary Online Security service. Fiber optic "
                "customers without security features show higher churn rates."
            )
        if tech_support == "No":
            recommendations.append(
                "Offer a free Tech Support trial. Fiber optic users often "
                "need technical assistance, and providing it reduces frustration."
            )

    # --- Payment method recommendations ---
    payment = customer_data.get("PaymentMethod", "")
    if payment == "Electronic check":
        recommendations.append(
            "Encourage switching from Electronic Check to automatic payment "
            "(bank transfer or credit card). Electronic check users have "
            "significantly higher churn rates due to manual payment friction."
        )

    # --- Paperless billing ---
    paperless = customer_data.get("PaperlessBilling", "")
    if paperless == "Yes" and contract == "Month-to-month":
        recommendations.append(
            "Paperless billing + month-to-month contract is a high-risk "
            "combination. Offer a small discount for switching to a longer "
            "contract term."
        )

    # --- Monthly charges ---
    monthly = customer_data.get("MonthlyCharges", 0)
    if monthly > 80:
        recommendations.append(
            "High monthly charges detected (>${:.0f}/mo). Review the "
            "customer's plan for potential bundle discounts or loyalty "
            "pricing to improve perceived value.".format(monthly)
        )

    # --- Streaming services without protection ---
    streaming_tv = customer_data.get("StreamingTV", "No")
    streaming_movies = customer_data.get("StreamingMovies", "No")
    device_protection = customer_data.get("DeviceProtection", "No")

    if (streaming_tv == "Yes" or streaming_movies == "Yes") and device_protection == "No":
        recommendations.append(
            "Customer uses streaming services but lacks Device Protection. "
            "Offer a bundled entertainment + protection package at a discount."
        )

    # --- Senior citizen specific ---
    senior = customer_data.get("SeniorCitizen", 0)
    if senior == 1:
        recommendations.append(
            "Senior citizen customer. Consider offering a senior-specific "
            "plan with simplified billing, priority support, and tailored "
            "service bundles."
        )

    # --- High confidence churn ---
    if confidence > 0.8:
        recommendations.insert(0,
            "CRITICAL: Churn probability exceeds 80%. Immediate intervention "
            "required -- assign a dedicated account manager and offer a "
            "personalized retention package within 48 hours."
        )
    elif confidence > 0.6:
        recommendations.insert(0,
            "WARNING: Elevated churn risk detected. Schedule a proactive "
            "outreach call within the next week to address potential concerns."
        )

    # Fallback if no specific recommendations
    if not recommendations:
        recommendations.append(
            "Customer profile appears stable. Continue standard engagement "
            "and monitor for any changes in usage patterns."
        )

    return recommendations


# =============================================================================
# Dataset Statistics (for Dashboard)
# =============================================================================
def get_dataset_stats(df=None) -> dict:
    """
    Load the dataset and compute summary statistics for the Dashboard page.

    Returns:
        dict: Contains KPI values, distribution data for charts, and
              segmentation data. Returns empty dict if dataset not found.
    """

    if df is None:
        if not os.path.exists(DATASET_PATH):
            return {}

        df = pd.read_csv(DATASET_PATH)

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0.0)
    total = len(df)
    churn_yes = int((df["Churn"] == "Yes").sum())
    churn_no = int((df["Churn"] == "No").sum())

    stats = {
        # --- KPI Cards ---
        "total_customers": total,
        "churn_count": churn_yes,
        "retained_count": churn_no,
        "churn_rate": round(churn_yes / total * 100, 2),
        "avg_tenure": round(df["tenure"].mean(), 1),
        "avg_monthly_charges": round(df["MonthlyCharges"].mean(), 2),
        "avg_total_charges": round(df["TotalCharges"].mean(), 2),
        "senior_pct": round(df["SeniorCitizen"].mean() * 100, 1),

        # --- Chart Data: Churn Distribution ---
        "churn_labels": ["No Churn", "Churn"],
        "churn_values": [churn_no, churn_yes],

        # --- Chart Data: Contract Distribution ---
        "contract_labels": df["Contract"].value_counts().index.tolist(),
        "contract_values": df["Contract"].value_counts().values.tolist(),

        # --- Chart Data: Internet Service ---
        "internet_labels": df["InternetService"].value_counts().index.tolist(),
        "internet_values": df["InternetService"].value_counts().values.tolist(),

        # --- Chart Data: Payment Method ---
        "payment_labels": df["PaymentMethod"].value_counts().index.tolist(),
        "payment_values": df["PaymentMethod"].value_counts().values.tolist(),

        # --- Chart Data: Gender Distribution ---
        "gender_labels": df["gender"].value_counts().index.tolist(),
        "gender_values": df["gender"].value_counts().values.tolist(),

        # --- Chart Data: Churn by Contract Type ---
        "churn_by_contract": _churn_by_category(df, "Contract"),

        # --- Chart Data: Churn by Internet Service ---
        "churn_by_internet": _churn_by_category(df, "InternetService"),

        # --- Chart Data: Churn by Payment Method ---
        "churn_by_payment": _churn_by_category(df, "PaymentMethod"),

        # --- Chart Data: Tenure Distribution ---
        "tenure_churned": df[df["Churn"] == "Yes"]["tenure"].tolist(),
        "tenure_retained": df[df["Churn"] == "No"]["tenure"].tolist(),

        # --- Chart Data: Monthly Charges Distribution ---
        "charges_churned": df[df["Churn"] == "Yes"]["MonthlyCharges"].tolist(),
        "charges_retained": df[df["Churn"] == "No"]["MonthlyCharges"].tolist(),
    }

    return stats


def _churn_by_category(df: pd.DataFrame, column: str) -> dict:
    """
    Calculate churn rate for each category in a given column.

    Args:
        df: DataFrame with 'Churn' column.
        column: Categorical column name to group by.

    Returns:
        dict: {"labels": [...], "churn_rates": [...], "counts": [...]}
    """
    grouped = df.groupby(column)["Churn"].apply(
        lambda x: round((x == "Yes").mean() * 100, 2)
    )
    counts = df[column].value_counts()

    return {
        "labels": grouped.index.tolist(),
        "churn_rates": grouped.values.tolist(),
        "counts": [int(counts[label]) for label in grouped.index],
    }


# =============================================================================
# Analytics Data (for Analytics page)
# =============================================================================
def get_analytics_data(df=None, predictor=None):

    if df is None:
        if not os.path.exists(DATASET_PATH):
            return {}

        df = pd.read_csv(DATASET_PATH)

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0.0)

    df["Churn_Binary"] = (
        df["Churn"] == "Yes"
    ).astype(int)

    analytics = {}

    # ===============================
    # Analytics Summary Cards
    # ===============================

    total_customers = len(df)

    churned = int(df["Churn_Binary"].sum())
    retained = total_customers - churned

    avg_tenure = round(
        df["tenure"].mean(), 1
    ) if total_customers > 0 else 0

    avg_monthly = round(
        df["MonthlyCharges"].mean(), 1
    ) if total_customers > 0 else 0

    analytics["summary"] = {
        "avgTenure": avg_tenure,
        "avgMonthly": avg_monthly,
        "churned": churned,
        "retained": retained
    }

    # --- Correlation Matrix ---

    numeric_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn_Binary"
    ]

    corr_matrix = df[numeric_cols].corr().round(3)

    analytics["correlation"] = {
        "labels": numeric_cols,
        "values": corr_matrix.values.tolist(),
    }


    # --- Correlation Matrix (numeric columns) ---
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn_Binary"]
    corr_matrix = df[numeric_cols].corr().round(3)
    analytics["correlation"] = {
        "labels": numeric_cols,
        "values": corr_matrix.values.tolist(),
    }

    # --- Customer Segmentation by Tenure ---
    tenure_bins = [0, 6, 12, 24, 48, 72, 120]
    tenure_labels = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo", "72+ mo"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True)

    tenure_seg = df.groupby("tenure_group", observed=True).agg(
        count=("Churn", "count"),
        churn_rate=("Churn_Binary", lambda x: round(x.mean() * 100, 2)),
        avg_monthly=("MonthlyCharges", lambda x: round(x.mean(), 2)),
    ).reset_index()

    analytics["tenure_segmentation"] = {
        "labels": tenure_seg["tenure_group"].astype(str).tolist(),
        "counts": tenure_seg["count"].tolist(),
        "churn_rates": tenure_seg["churn_rate"].tolist(),
        "avg_charges": tenure_seg["avg_monthly"].tolist(),
    }

    # --- Churn by Service Combinations ---
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity",
        "OnlineBackup", "TechSupport", "DeviceProtection",
    ]
    service_churn = {}
    for col in service_cols:
        grouped = df.groupby(col)["Churn_Binary"].mean() * 100
        service_churn[col] = {
            "labels": grouped.index.tolist(),
            "churn_rates": grouped.round(2).values.tolist(),
        }
    analytics["service_churn"] = service_churn

    # --- Monthly Charges vs Churn (binned) ---
    charge_bins = [0, 20, 40, 60, 80, 100, 120]
    charge_labels = ["$0-20", "$20-40", "$40-60", "$60-80", "$80-100", "$100-120"]
    df["charge_group"] = pd.cut(
        df["MonthlyCharges"], bins=charge_bins, labels=charge_labels, right=True
    )
    charge_churn = df.groupby("charge_group", observed=True).agg(
        count=("Churn", "count"),
        churn_rate=("Churn_Binary", lambda x: round(x.mean() * 100, 2)),
    ).reset_index()

    analytics["charges_segmentation"] = {
        "labels": charge_churn["charge_group"].astype(str).tolist(),
        "counts": charge_churn["count"].tolist(),
        "churn_rates": charge_churn["churn_rate"].tolist(),
    }

    # --- Senior Citizen Analysis ---
    senior_data = df.groupby("SeniorCitizen").agg(
        count=("Churn", "count"),
        churn_rate=("Churn_Binary", lambda x: round(x.mean() * 100, 2)),
    ).reset_index()

    analytics["senior_analysis"] = {
        "labels": ["Non-Senior", "Senior"],
        "counts": senior_data["count"].tolist(),
        "churn_rates": senior_data["churn_rate"].tolist(),
    }

   
      # -----------------------------
    # Dashboard - Churn Trend
    # -----------------------------

    analytics["churnTrend"] = {
        "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "churn": [22, 24, 25, 27, 26, 28, 30, 29, 27, 26, 25, 26],
        "retained": [78, 76, 75, 73, 74, 72, 70, 71, 73, 74, 75, 74]
    }
    # -----------------------------
    # Churn vs Retention
    # -----------------------------

    total_customers = len(df)

    if total_customers > 0:
        churned = int(df["Churn_Binary"].sum())
        retained = total_customers - churned

        churn_percent = round(
            (churned / total_customers) * 100, 1
        )

        retained_percent = round(
            (retained / total_customers) * 100, 1
        )
    else:
        churn_percent = 0
        retained_percent = 0

    analytics["churnTrend"] = {
        "labels": ["Churned", "Retained"],
        "churn": [churn_percent, 0],
        "retained": [0, retained_percent]
    }

  
    # -----------------------------
    # Dashboard - Risk Distribution
    # -----------------------------

    risk_score = pd.Series(0.0, index=df.index)

    # Contract risk
    risk_score += df["Contract"].eq("Month-to-month") * 25
    risk_score += df["Contract"].eq("One year") * 8

    # Tenure risk
    risk_score += (df["tenure"] <= 12) * 20
    risk_score += ((df["tenure"] > 12) & (df["tenure"] <= 24)) * 10

    # Monthly charge risk
    risk_score += (df["MonthlyCharges"] >= 80) * 20
    risk_score += (
        (df["MonthlyCharges"] >= 60) &
        (df["MonthlyCharges"] < 80)
    ) * 10

    # Support/security risk
    risk_score += df["TechSupport"].eq("No") * 10
    risk_score += df["OnlineSecurity"].eq("No") * 10

    # Paperless billing
    risk_score += df["PaperlessBilling"].eq("Yes") * 5

    low_risk = int((risk_score < 35).sum())
    medium_risk = int(
        ((risk_score >= 35) & (risk_score < 60)).sum()
    )
    high_risk = int((risk_score >= 60).sum())

    analytics["riskDistribution"] = {
        "labels": ["Low", "Medium", "High"],
        "data": [low_risk, medium_risk, high_risk]
    }
    # -----------------------------
    # Dashboard - Contract Chart
    # -----------------------------
    contract = df.groupby("Contract")["Churn_Binary"].mean() * 100

    analytics["churnByContract"] = {
        "labels": contract.index.tolist(),
        "data": contract.round(2).tolist()
    }

    # -----------------------------
    # Dashboard - Tenure Chart
    # -----------------------------
    analytics["tenure"] = {
        "labels": tenure_seg["tenure_group"].astype(str).tolist(),
        "retained": (
            tenure_seg["count"] -
            (tenure_seg["count"] * tenure_seg["churn_rate"] / 100)
        ).round().astype(int).tolist(),
        "churned": (
            tenure_seg["count"] *
            tenure_seg["churn_rate"] / 100
        ).round().astype(int).tolist()
    }
    

       # -----------------------------
    # Dashboard - Top At-Risk Customers
    # -----------------------------

    analytics["topRisk"] = {
        "customerID": [],
        "contract": [],
        "tenure": [],
        "monthlyCharges": [],
        "risk": [],
        "probability": []
    }

    if predictor is not None and len(df) > 0:

        # Get the actual model and preprocessing objects
        predict_globals = predictor.__globals__

        model = predict_globals["model"]
        label_encoders = predict_globals["label_encoders"]
        feature_names = predict_globals["feature_names"]

        # Prepare filtered customers
        prediction_df = df[feature_names].copy()

        numeric_columns = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        for col in numeric_columns:
            prediction_df[col] = pd.to_numeric(
                prediction_df[col],
                errors="coerce"
            )

        # Apply same encoders as predict.py
        for column, encoder in label_encoders.items():

            if column in prediction_df.columns:

                prediction_df[column] = encoder.transform(
                    prediction_df[column].astype(str)
                )

        # Same feature order as the trained model
        prediction_df = prediction_df[feature_names]

        # Predict ALL customers at once
        probabilities = model.predict_proba(
            prediction_df
        )[:, 1]

        predictions = []

        for i, probability in enumerate(probabilities):

            probability = round(
                float(probability) * 100,
                2
            )

            if probability >= 70:
                risk = "High"
            elif probability >= 40:
                risk = "Medium"
            else:
                risk = "Low"

            predictions.append({
                "customerID": str(df.iloc[i]["customerID"]),
                "contract": str(df.iloc[i]["Contract"]),
                "tenure": int(df.iloc[i]["tenure"]),
                "monthlyCharges": round(
                    float(df.iloc[i]["MonthlyCharges"]),
                    2
                ),
                "risk": risk,
                "probability": probability
            })

        # Highest probability first
        predictions.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        # Top 10
        predictions = predictions[:10]

        analytics["topRisk"] = {
            "customerID": [
                x["customerID"] for x in predictions
            ],
            "contract": [
                x["contract"] for x in predictions
            ],
            "tenure": [
                x["tenure"] for x in predictions
            ],
            "monthlyCharges": [
                x["monthlyCharges"] for x in predictions
            ],
            "risk": [
                x["risk"] for x in predictions
            ],
            "probability": [
                x["probability"] for x in predictions
            ]
        }

    return analytics
# =============================================================================
# Feature Importance (for Analytics page)
# =============================================================================
def get_feature_importance(model, feature_names: list) -> dict:
    """
    Extract feature importance from the trained model.

    For Logistic Regression, uses absolute coefficient values.
    For tree-based models, uses the built-in feature_importances_.

    Args:
        model: Trained sklearn model object.
        feature_names: List of feature names matching model input order.

    Returns:
        dict: {"features": [...], "importances": [...]} sorted descending.
    """
    try:
        if hasattr(model, "feature_importances_"):
            # Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Linear models (Logistic Regression) -- use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            return {"features": feature_names, "importances": [1.0] * len(feature_names)}

        # Normalize to 0-100 scale for display
        if importances.max() > 0:
            importances = (importances / importances.max() * 100).round(2)

        # Sort by importance (descending)
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importances = [float(importances[i]) for i in sorted_indices]

        return {
            "features": sorted_features,
            "importances": sorted_importances,
        }
    except Exception:
        return {"features": feature_names, "importances": [0.0] * len(feature_names)}


# =============================================================================
# Model Metrics Formatter
# =============================================================================
def format_model_metrics(pipeline: dict) -> dict:
    """
    Format model metrics from the saved pipeline for display on the
    Dashboard and About pages.

    Args:
        pipeline: Loaded preprocessing_pipeline.pkl dictionary.

    Returns:
        dict: Formatted metrics with percentage strings and labels.
    """
    raw_metrics = pipeline.get("model_metrics", {})

    return {
        "model_name": pipeline.get("best_model_name", "Unknown"),
        "training_date": pipeline.get("training_date", "N/A"),
        "accuracy": round(raw_metrics.get("Accuracy", 0) * 100, 2),
        "precision": round(raw_metrics.get("Precision", 0) * 100, 2),
        "recall": round(raw_metrics.get("Recall", 0) * 100, 2),
        "f1_score": round(raw_metrics.get("F1 Score", 0) * 100, 2),
        "roc_auc": round(raw_metrics.get("ROC AUC", 0) * 100, 2),
    }
