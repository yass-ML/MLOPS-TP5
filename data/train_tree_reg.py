import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib



def build_decision_tree_model():
    
    df = pd.read_csv('houses.csv')
    X = df[['size', 'nb_rooms', 'garden']]
    y = df['price']

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "tree_reg.joblib")
    print(f"\nDecision tree regressor model saved to: tree_reg.joblib")
    

if __name__ == "__main__":
    build_decision_tree_model()
