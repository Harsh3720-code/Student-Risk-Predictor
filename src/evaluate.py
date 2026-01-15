import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

def main():
    print("Loading processed data & models...")

    data_dir = "../data/preprocessed"
    model_dir = "../models"

    # Load data
    X_test = np.load(os.path.join(data_dir, "X_test_processed.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    # Safety: force correct shape
    y_test = np.array(y_test).reshape(-1).astype(int)

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape, "unique:", np.unique(y_test))

    # Load models
    log_reg = joblib.load(os.path.join(model_dir, "log_reg.pkl"))
    rf = joblib.load(os.path.join(model_dir, "rf.pkl"))

    # -----------------------
    # Evaluate Logistic Regression
    # -----------------------
    print("\nEvaluating Logistic Regression...")
    y_pred_lr = log_reg.predict(X_test)
    y_pred_rf = rf.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    auc_lr = roc_auc_score(y_test, y_pred_lr)
    print("AUC Score:", auc_lr)

    # -----------------------
    # Evaluate Random Forest
    # -----------------------
    print("\nEvaluating Random Forest...")
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    auc_rf = roc_auc_score(y_test, y_pred_rf)
    print("AUC Score:", auc_rf)

    # -----------------------
    # ROC Curve Plot
    # -----------------------

    print("\nComputing ROC curves...")
    # fpr = false positive rate and tpr = true positive rate (Recall)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, fpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label=f"Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)

    plt.show()

    # -----------------------
    # Threshold Tuning (Random Forest)
    # -----------------------
    print("\nThreshold tuning for Random Forest...")

    threshold_to_test = [0.5,0.4,0.3,0.25,0.2]

    for t in threshold_to_test:
        print("\n--- Threshold = {t} ---")

        y_pred_tuned = (y_proba_rf >= t).astype(int)

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_tuned))
        print("Classification Report:")
        print(classification_report(y_test, y_pred_tuned))

    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()






