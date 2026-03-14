"""SVM training script.

Ce module expose `train_svm` pour être utilisé par les tests et autres scripts.

Usage :
    python -m src.train_model_SVM
"""

from __future__ import annotations

import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from .data_processing_SVM import load_data
except ImportError:  # pragma: no cover
    from data_processing_SVM import load_data


def train_svm(X_train, y_train, save_model: bool = True):
    """Entraîne un SVM et sauve le modèle si demandé."""

    # Sélection des features
    X_train = X_train[["Age", "Number of sexual partners"]]

    svm_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    svm_pipeline.fit(X_train, y_train)

    if save_model:
        base_path = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(base_path, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "svm_model.pkl")
        joblib.dump(svm_pipeline, model_file)
        print("Model sauvegardé dans :", model_file)

    return svm_pipeline


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    train_svm(X_train, y_train)


if __name__ == "__main__":
    main()
