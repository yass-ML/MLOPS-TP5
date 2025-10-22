import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib



def build_logistic_model():
    
    df = pd.read_csv("breast-cancer.csv") # https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?resource=download
    y = df["diagnosis"].map({"M":1, "B":0})  # Map Malignant to 1, Benign to 0
    X = df.drop(columns=["diagnosis"])

    # Train logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "logistic_regression.joblib")
    print(f"\nLogistic regression model saved to: logistic_regression.joblib")
    
    # Show model info
    print(f"\nModel Information:")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    print(f"  Coefficients: {model.coef_[0]}")
    print(f"  Classes: {model.classes_}")
    print(f"  Training accuracy: {model.score(X, y):.2%}")


if __name__ == "__main__":
    build_logistic_model()
