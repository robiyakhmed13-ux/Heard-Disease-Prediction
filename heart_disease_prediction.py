# =============================================================================
# Heart Disease Prediction using Logistic Regression
# Author: [Your Name]
# Dataset: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the heart disease dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['target'].value_counts()}")
    print("  0 = Healthy heart | 1 = Defective heart (has disease)")
    return df


# =============================================================================
# 2. EDA
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Visualise target distribution, age vs disease, and correlation heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Heart Disease Dataset – EDA", fontsize=16)

    # Target distribution
    sns.countplot(x='target', data=df, ax=axes[0],
                  palette=['steelblue', 'salmon'])
    axes[0].set_title("Heart Disease Distribution")
    axes[0].set_xticklabels(['Healthy (0)', 'Disease (1)'])
    axes[0].set_ylabel("Count")

    # Age vs target
    sns.boxplot(x='target', y='age', data=df, ax=axes[1],
                palette=['steelblue', 'salmon'])
    axes[1].set_title("Age vs Heart Disease")
    axes[1].set_xticklabels(['Healthy', 'Disease'])

    # Max heart rate vs disease
    sns.boxplot(x='target', y='thalach', data=df, ax=axes[2],
                palette=['steelblue', 'salmon'])
    axes[2].set_title("Max Heart Rate vs Heart Disease")
    axes[2].set_xticklabels(['Healthy', 'Disease'])

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")

    # Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues', square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Heatmap saved as 'correlation_heatmap.png'")


# =============================================================================
# 3. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns='target', axis=1)
    Y = df['target']
    print(f"Features: {X.shape} | Target: {Y.shape}")
    return X, Y


# =============================================================================
# 4. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.1, random_state=1):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 5. Model Training
# =============================================================================

def train_model(X_train, Y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    print("Model training complete.")
    return model


# =============================================================================
# 6. Model Evaluation
# =============================================================================

def evaluate_model(model, X_train, Y_train, X_test, Y_test) -> None:
    """Accuracy, classification report, and confusion matrix."""
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    print(f"\nTraining Accuracy : {accuracy_score(Y_train, train_preds):.4f}")
    print(f"Test     Accuracy : {accuracy_score(Y_test,  test_preds):.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(Y_test, test_preds,
                                target_names=['Healthy', 'Has Disease']))

    # Confusion matrix
    cm = confusion_matrix(Y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Healthy', 'Has Disease'],
                yticklabels=['Healthy', 'Has Disease'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'")


# =============================================================================
# 7. Predictive System
# =============================================================================

def predict_heart_disease(model, input_data: tuple) -> str:
    """
    Predict heart disease for a single patient.

    Parameters
    ----------
    input_data : tuple of 13 values:
        (age, sex, cp, trestbps, chol, fbs, restecg,
         thalach, exang, oldpeak, slope, ca, thal)

    sex  : Male=1, Female=0
    cp   : chest pain type (0-3)
    fbs  : fasting blood sugar > 120 mg/dl (1=True, 0=False)
    """
    arr = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(arr)
    if prediction[0] == 0:
        return "💚 Prediction: Healthy Heart — No disease detected."
    else:
        return "❤️‍🩹 Prediction: Defective Heart — Disease likely present."


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "heart_disease_data.csv"   # update path if needed

    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())
    print("\nStatistical summary:\n", df.describe())

    plot_eda(df)

    X, Y = split_features_target(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    model = train_model(X_train, Y_train)
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # Sample prediction
    # (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    sample = (48, 0, 2, 130, 275, 0, 1, 139, 0, 0.2, 2, 0, 2)
    result = predict_heart_disease(model, sample)
    print(f"\nSample Prediction:\n{result}")
