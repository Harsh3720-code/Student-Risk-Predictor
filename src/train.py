import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    print("Loading preprocessed data...")

    data_dir = "../data/preprocessed"

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    # Ensure model output delivery exists
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)

    # --------------------------
    # Train Logistic Regression
    # --------------------------

    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    y_pred_lr = log_reg.predict(X_test)
    print("\nLogistic Regression Evaluation:")
    print(classification_report(y_test, y_pred_lr))

    joblib.dump(log_reg, os.path.join(model_dir, "log_reg.pkl"))
    print("Saved Logistic Regression model successfully.")

    # --------------------------
    # Train Random Forest
    # --------------------------

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    print("\nRandom Forest Evaluation:")
    print(classification_report(y_test, y_pred_rf))

    joblib.dump(rf, os.path.join(model_dir, "rf.pkl"))
    print("Saved Random Forest model successfully.")

    print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()