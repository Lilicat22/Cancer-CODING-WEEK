# src/evaluate_model.py
import pandas as pd
import joblib
import shap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model():

    # Charger les données
    X_test = pd.read_csv("../data/X_test_cleaned.csv")
    y_test = pd.read_csv("../data/y_test_cleaned.csv")

    y_test = y_test.squeeze()

    # Charger le modèle
    model = joblib.load("../models/svm_model.pkl")

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    # Métriques
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # SHAP
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[1], X_test)


if __name__ == "__main__":
    evaluate_model()
