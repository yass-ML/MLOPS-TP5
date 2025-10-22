import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib



def build_decision_tree_model():
    
    df = pd.read_csv("breast-cancer.csv") # https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset?resource=download
    y = df["diagnosis"].map({"M":1, "B":0})  # Map Malignant to 1, Benign to 0
    X = df.drop(columns=["diagnosis"])

    # Train logistic regression
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "tree_clf.joblib")
    print(f"\nDecision tree classifier model saved to: tree_clf.joblib")
    
    # Show model info
    print(f"\nModel Information:")
    print(f"  Feature importances: {model.feature_importances_}")
    print(f"  Classes: {model.classes_}")
    print(f"  Training accuracy: {model.score(X, y):.2%}")


if __name__ == "__main__":
    build_decision_tree_model()
