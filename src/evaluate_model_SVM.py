"""SVM evaluation utilities.

Ce module expose des fonctions pour évaluer et expliquer un modèle SVM.

Usage :
    python -m src.evaluate_model_SVM
"""

from __future__ import annotations

import os
import joblib
import shap
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from .data_processing_SVM import load_data
except ImportError:  # pragma: no cover
    from data_processing_SVM import load_data


def evaluate_model(model, X_test, y_test, save_package: bool = False):
    """Calcule les métriques d'évaluation et retourne les prédictions."""

    # Le modèle SVM est entraîné sur un sous-ensemble de colonnes.
    X_test = X_test[["Age", "Number of sexual partners"]]
    y_pred = model.predict(X_test)

    if save_package:
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        package = {
            "model": model,
            "accuracy": accuracy,
            "confusion_matrix": cm,
        }

        base_path = os.path.dirname(os.path.dirname(__file__))
        package_path = os.path.join(base_path, "models", "svm_package.pkl")
        os.makedirs(os.path.dirname(package_path), exist_ok=True)
        joblib.dump(package, package_path)

    return y_pred


def shap_analysis(model, X_train, X_test, nsamples: int = 100):
    """Retourne un explainer SHAP (sans calculer les valeurs SHAP par défaut)."""

    # Alignement des colonnes avec celles utilisées pendant l'entraînement.
    X_train = X_train[["Age", "Number of sexual partners"]]
    X_test = X_test[["Age", "Number of sexual partners"]]

    # Shap KernelExplainer rejette parfois les objets sklearn Pipeline car il tente
    # d'écrire dans `feature_names_in_`. Enveloppons la prédiction dans une
    # fonction simple pour éviter cela.
    def predict_proba(X):
        return model.predict_proba(X)

    explainer = shap.KernelExplainer(predict_proba, X_train)
    # Si besoin, on peut calculer les valeurs :
    # shap_values = explainer.shap_values(X_test, nsamples=nsamples)

    return explainer


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "models", "svm_model.pkl")
    model = joblib.load(model_path)

    y_pred = evaluate_model(model, X_test, y_test, save_package=True)
    _ = shap_analysis(model, X_train, X_test)

    print(f"SVM evaluated – {len(y_pred)} predictions made.")


if __name__ == "__main__":
    main()
