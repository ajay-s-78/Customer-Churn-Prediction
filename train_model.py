# =============================================================================
# train_model.py -- Complete ML Training Pipeline
# =============================================================================
# Customer Churn Prediction System -- Phase 2
#
# This script performs the full end-to-end machine learning pipeline:
#   1. Load & inspect the Telco Customer Churn dataset
#   2. Clean data (missing values, type conversions, duplicates)
#   3. Encode categorical features using LabelEncoder
#   4. Split into 80/20 train/test sets (stratified)
#   5. Train 4 classifiers with optimized hyperparameters
#   6. Evaluate with Accuracy, Precision, Recall, F1, ROC AUC
#   7. Print comparison table and confusion matrices
#   8. Select the best model (by ROC AUC) and save as model.pkl
#   9. Save the preprocessing pipeline (label encoders + feature names)
#
# Usage:  python train_model.py
# Output: model.pkl, preprocessing_pipeline.pkl
# =============================================================================

import sys
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
DATASET_PATH = os.path.join("dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_OUTPUT_PATH = "model.pkl"
PIPELINE_OUTPUT_PATH = "preprocessing_pipeline.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def print_header(title: str) -> None:
    """Print a formatted section header for console output."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# STEP 1: Load Dataset
# =============================================================================
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset from CSV.

    Args:
        path: Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: Raw dataset.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
    """
    print_header("STEP 1: Loading Dataset")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            f"Please place the CSV file in the dataset/ directory."
        )

    df = pd.read_csv(path)
    print(f"  [OK] Loaded dataset from: {path}")
    print(f"  [OK] Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


# =============================================================================
# STEP 2: Display Dataset Information
# =============================================================================
def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Display comprehensive dataset information including shape,
    data types, missing values, and class distribution.

    Args:
        df: The loaded DataFrame.
    """
    print_header("STEP 2: Dataset Information")

    # Basic shape
    print_subheader("Shape")
    print(f"  Rows:    {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")

    # Column data types
    print_subheader("Column Data Types")
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        print(f"  {col:<22} | Type: {dtype_str:<10} | Unique: {unique_count:<6} | Nulls: {null_count}")

    # Target variable distribution
    print_subheader("Target Variable Distribution (Churn)")
    churn_counts = df["Churn"].value_counts()
    total = len(df)
    for label, count in churn_counts.items():
        pct = count / total * 100
        print(f"  {label:<5} : {count:>5} ({pct:.1f}%)")

    # Statistical summary for numeric columns
    print_subheader("Numeric Columns — Statistical Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe().round(2).to_string())


# =============================================================================
# STEP 3: Data Cleaning
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning operations:
      1. Convert TotalCharges from string to numeric (coerce errors to NaN)
      2. Fill NaN TotalCharges with 0.0 (these are tenure=0 new customers)
      3. Remove duplicate rows
      4. Drop the customerID column (not a predictive feature)

    Args:
        df: Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    print_header("STEP 3: Data Cleaning")

    # --- 3a. Convert TotalCharges to numeric ---
    # TotalCharges is loaded as string because 11 rows contain blank spaces
    # These blanks correspond to customers with tenure=0 (brand new customers)
    print_subheader("3a. Converting TotalCharges to Numeric")
    blank_count = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    print(f"  [OK] Found {blank_count} blank TotalCharges entries (tenure=0 customers)")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    nan_count = df["TotalCharges"].isnull().sum()
    print(f"  [OK] Converted to numeric -- {nan_count} NaN values created")

    # --- 3b. Handle missing values ---
    # Fill NaN TotalCharges with 0.0 since these are new customers with no charges yet
    print_subheader("3b. Handling Missing Values")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    remaining_nulls = df.isnull().sum().sum()
    print(f"  [OK] Filled NaN TotalCharges with 0.0")
    print(f"  [OK] Remaining null values in entire dataset: {remaining_nulls}")

    # --- 3c. Remove duplicates ---
    print_subheader("3c. Removing Duplicates")
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"  [OK] Checked {initial_rows} rows -- removed {removed} duplicate(s)")
    print(f"  [OK] Remaining rows: {len(df)}")

    # --- 3d. Remove customerID ---
    # customerID is a unique identifier, not a predictive feature
    print_subheader("3d. Removing customerID Column")
    df = df.drop(columns=["customerID"])
    print(f"  [OK] Dropped 'customerID' -- {len(df.columns)} columns remaining")

    # --- Summary ---
    print_subheader("Cleaning Summary")
    print(f"  Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Data types:")
    for dtype_name, count in df.dtypes.value_counts().items():
        print(f"    {str(dtype_name):<12}: {count} columns")

    return df


# =============================================================================
# STEP 4: Encode Categorical Variables
# =============================================================================
def encode_features(df: pd.DataFrame) -> tuple:
    """
    Encode all categorical (object/string) columns using LabelEncoder.
    Also encodes the target variable 'Churn' (No=0, Yes=1).

    We use LabelEncoder here because:
      - Tree-based models (Decision Tree, Random Forest, Gradient Boosting)
        handle ordinal-encoded features natively
      - Logistic Regression will still converge with label-encoded features
        for this dataset's category sizes (2-4 unique values per column)

    Args:
        df: Cleaned DataFrame with categorical columns.

    Returns:
        tuple: (encoded_df, label_encoders_dict, feature_names_list)
    """
    print_header("STEP 4: Encoding Categorical Variables")

    # Identify categorical columns (object / string types)
    # Use both 'object' and 'string' dtypes to catch all text columns across
    # different pandas versions (pandas 3.x uses StringDtype by default)
    cat_cols_obj = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols_str = df.select_dtypes(include=["string"]).columns.tolist()
    categorical_columns = list(dict.fromkeys(cat_cols_obj + cat_cols_str))  # deduplicate, preserve order
    print(f"  Found {len(categorical_columns)} categorical columns to encode:")

    # Store label encoders for each column (needed for prediction preprocessing)
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        original_values = sorted(df[col].unique().tolist())
        df[col] = le.fit_transform(df[col].astype(str))
        encoded_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        label_encoders[col] = le
        print(f"\n  {col}:")
        for original, encoded in encoded_mapping.items():
            print(f"    {original:<30} -> {encoded}")

    # Separate features and target
    feature_names = [col for col in df.columns if col != "Churn"]

    print_subheader("Encoding Summary")
    print(f"  [OK] Encoded {len(categorical_columns)} categorical columns")
    print(f"  [OK] Feature columns: {len(feature_names)}")
    print(f"  [OK] Target column: Churn (No=0, Yes=1)")
    print(f"  [OK] All columns now numeric:")
    print(f"    {df.dtypes.value_counts().to_string()}")

    return df, label_encoders, feature_names


# =============================================================================
# STEP 5: Split Dataset
# =============================================================================
def split_dataset(df: pd.DataFrame, feature_names: list) -> tuple:
    """
    Split the dataset into training (80%) and testing (20%) sets.
    Uses stratified splitting to maintain class distribution in both sets.

    Args:
        df: Fully encoded DataFrame.
        feature_names: List of feature column names (excludes 'Churn').

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print_header("STEP 5: Splitting Dataset (80% Train / 20% Test)")

    X = df[feature_names]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # Maintain class distribution in both splits
    )

    print(f"  [OK] Total samples:    {len(X)}")
    print(f"  [OK] Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  [OK] Testing samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Verify stratification preserved class distribution
    print_subheader("Class Distribution Verification")
    train_churn_pct = y_train.mean() * 100
    test_churn_pct = y_test.mean() * 100
    print(f"  Training set churn rate: {train_churn_pct:.2f}%")
    print(f"  Testing set churn rate:  {test_churn_pct:.2f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 6: Train Models
# =============================================================================
def train_models(X_train, y_train) -> dict:
    """
    Train four classification models with tuned hyperparameters:
      1. Logistic Regression — linear baseline with regularization
      2. Decision Tree — interpretable single-tree model
      3. Random Forest — bagging ensemble of decision trees
      4. Gradient Boosting — sequential boosting ensemble

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        dict: {model_name: trained_model_object}
    """
    print_header("STEP 6: Training Models")

    # Define models with optimized hyperparameters
    # -------------------------------------------------------------------------
    # Design decisions for hyperparameters:
    #   - Logistic Regression: max_iter=1000 ensures convergence on this dataset;
    #     C=1.0 provides moderate regularization
    #   - Decision Tree: max_depth=10 prevents overfitting while capturing
    #     key decision boundaries; min_samples_split=5 avoids tiny leaf nodes
    #   - Random Forest: 200 trees for stable predictions; max_depth=15 allows
    #     deeper trees since the ensemble averages out overfitting
    #   - Gradient Boosting: 200 estimators with learning_rate=0.1 balances
    #     learning speed and generalization; max_depth=5 keeps individual
    #     trees shallow (standard for boosting)
    # -------------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,  # Use all CPU cores for parallel training
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,  # Stochastic gradient boosting for regularization
            random_state=RANDOM_STATE,
        ),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training: {name}...", end="", flush=True)
        start_time = datetime.now()
        model.fit(X_train, y_train)
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f" [OK] Done ({elapsed:.2f}s)")
        trained_models[name] = model

    print(f"\n  [OK] All {len(trained_models)} models trained successfully!")
    return trained_models


# =============================================================================
# STEP 7: Evaluate Models
# =============================================================================
def evaluate_models(trained_models: dict, X_test, y_test) -> dict:
    """
    Evaluate all trained models on the test set using 5 metrics:
      - Accuracy:  Overall correctness
      - Precision: Of predicted churners, how many actually churned
      - Recall:    Of actual churners, how many were correctly identified
      - F1 Score:  Harmonic mean of Precision and Recall
      - ROC AUC:   Area under ROC curve (probability-based ranking quality)

    Also generates confusion matrices and classification reports.

    Args:
        trained_models: Dict of {name: model} from train_models().
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        dict: {model_name: {metric_name: score, ..., 'confusion_matrix': cm}}
    """
    print_header("STEP 7: Evaluating Models")

    results = {}

    for name, model in trained_models.items():
        print_subheader(name)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of churn (class 1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Store results
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "confusion_matrix": cm,
        }

        # Print individual model results
        print(f"  Accuracy:   {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  ROC AUC:    {roc_auc:.4f}")

        # Confusion Matrix
        print(f"\n  Confusion Matrix:")
        print(f"                    Predicted No  Predicted Yes")
        print(f"    Actual No  :    {cm[0][0]:>8}      {cm[0][1]:>8}")
        print(f"    Actual Yes :    {cm[1][0]:>8}      {cm[1][1]:>8}")

        # Classification Report
        print(f"\n  Classification Report:")
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        # Add manual indentation for cleaner console output
        for line in report.strip().split("\n"):
            print(f"    {line}")

    return results


# =============================================================================
# STEP 8: Compare Models & Select Best
# =============================================================================
def compare_and_select_best(results: dict, trained_models: dict) -> tuple:
    """
    Print a side-by-side comparison table of all models and select
    the best-performing model based on ROC AUC score.

    ROC AUC is chosen as the selection criterion because:
      - It evaluates the model's ability to rank predictions correctly
      - It's threshold-independent (unlike Accuracy, Precision, Recall)
      - It handles class imbalance better than accuracy (26.5% churn rate)

    Args:
        results: Evaluation results from evaluate_models().
        trained_models: Dict of trained model objects.

    Returns:
        tuple: (best_model_name, best_model_object)
    """
    print_header("STEP 8: Model Comparison & Selection")

    # Build comparison table
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    model_names = list(results.keys())

    # Print table header
    print(f"\n  {'Metric':<14}", end="")
    for name in model_names:
        print(f"  {name:>20}", end="")
    print()
    print(f"  {'-' * 14}", end="")
    for _ in model_names:
        print(f"  {'-' * 20}", end="")
    print()

    # Print each metric row with best value highlighted
    for metric in metrics:
        scores = [results[name][metric] for name in model_names]
        best_score = max(scores)
        print(f"  {metric:<14}", end="")
        for score in scores:
            marker = " *" if score == best_score else "  "
            print(f"  {score:>17.4f}{marker}", end="")
        print()

    # Select best model by ROC AUC
    best_name = max(results, key=lambda name: results[name]["ROC AUC"])
    best_score = results[best_name]["ROC AUC"]
    best_model = trained_models[best_name]

    print(f"\n  {'-' * (14 + 22 * len(model_names))}")

    print(f"\n  >> BEST MODEL: {best_name}")
    print(f"     ROC AUC Score: {best_score:.4f}")
    print(f"     Accuracy: {results[best_name]['Accuracy']:.4f} ({results[best_name]['Accuracy']*100:.2f}%)")
    print(f"     F1 Score: {results[best_name]['F1 Score']:.4f}")

    return best_name, best_model


# =============================================================================
# STEP 9: Save Model & Preprocessing Pipeline
# =============================================================================
def save_model_and_pipeline(
    best_name: str,
    best_model,
    label_encoders: dict,
    feature_names: list,
    results: dict,
) -> None:
    """
    Save the best model and preprocessing pipeline to disk using joblib.

    Two files are saved:
      1. model.pkl — The trained classifier (for prediction)
      2. preprocessing_pipeline.pkl — Label encoders, feature names, and
         metadata needed to preprocess new input data identically to
         how the training data was processed

    Args:
        best_name: Name of the best model.
        best_model: Trained model object.
        label_encoders: Dict of {column_name: fitted_LabelEncoder}.
        feature_names: List of feature column names in correct order.
        results: Evaluation results dict.
    """
    print_header("STEP 9: Saving Model & Preprocessing Pipeline")

    # --- Save the trained model ---
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    model_size = os.path.getsize(MODEL_OUTPUT_PATH) / 1024.0  # KB
    print(f"  [OK] Model saved: {MODEL_OUTPUT_PATH} ({model_size:.1f} KB)")

    # --- Save the preprocessing pipeline ---
    # This pipeline contains everything needed to transform raw input data
    # into the same encoded format the model expects
    pipeline = {
        "label_encoders": label_encoders,
        "feature_names": feature_names,
        "best_model_name": best_name,
        "model_metrics": {
            metric: results[best_name][metric]
            for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        },
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "encoding_maps": {
            col: dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
            for col, le in label_encoders.items()
        },
    }

    joblib.dump(pipeline, PIPELINE_OUTPUT_PATH)
    pipeline_size = os.path.getsize(PIPELINE_OUTPUT_PATH) / 1024.0  # KB
    print(f"  [OK] Pipeline saved: {PIPELINE_OUTPUT_PATH} ({pipeline_size:.1f} KB)")

    # Print saved pipeline details
    print_subheader("Pipeline Contents")
    print(f"  - Label encoders for {len(label_encoders)} columns")
    print(f"  - Feature names ({len(feature_names)} features)")
    print(f"  - Best model: {best_name}")
    print(f"  - Model metrics snapshot")
    print(f"  - Encoding maps for prediction preprocessing")
    print(f"  - Training timestamp: {pipeline['training_date']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Execute the complete training pipeline end-to-end.
    """
    start_time = datetime.now()

    print("\n" + "=" * 70)
    print("  CUSTOMER CHURN PREDICTION -- MODEL TRAINING PIPELINE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Step 1: Load dataset
    df = load_dataset(DATASET_PATH)

    # Step 2: Display dataset information
    display_dataset_info(df)

    # Step 3: Clean data
    df = clean_data(df)

    # Step 4: Encode categorical variables
    df, label_encoders, feature_names = encode_features(df)

    # Step 5: Split into train/test sets
    X_train, X_test, y_train, y_test = split_dataset(df, feature_names)

    # Step 6: Train all models
    trained_models = train_models(X_train, y_train)

    # Step 7: Evaluate all models
    results = evaluate_models(trained_models, X_test, y_test)

    # Step 8: Compare and select best model
    best_name, best_model = compare_and_select_best(results, trained_models)

    # Step 9: Save model and preprocessing pipeline
    save_model_and_pipeline(best_name, best_model, label_encoders, feature_names, results)

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print_header("TRAINING COMPLETE")
    print(f"  Total execution time: {elapsed:.2f} seconds")
    print(f"  Best model: {best_name}")
    print(f"  ROC AUC: {results[best_name]['ROC AUC']:.4f}")
    print(f"  Files saved:")
    print(f"    - {MODEL_OUTPUT_PATH}")
    print(f"    - {PIPELINE_OUTPUT_PATH}")
    print(f"\n  Next step: Run 'python app.py' to start the Flask application.\n")


if __name__ == "__main__":
    main()
