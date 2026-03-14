"""XGBoost training script.

Ce module expose `train_xgboost` pour être utilisé par les tests et autres scripts.

Usage :
    python -m src.train_model_XGBoost
"""

from __future__ import annotations

import joblib
from pathlib import Path
import pandas as pd
import xgboost as xgb


def load_data():
    """Charge les jeux de données déjà splittés ou crée le split si nécessaire."""

    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"

    xtrain_path = data_dir / "X_train_cleaned.csv"
    ytrain_path = data_dir / "y_train_cleaned.csv"
    xtest_path = data_dir / "X_test_cleaned.csv"
    ytest_path = data_dir / "y_test_cleaned.csv"

    if xtrain_path.exists() and ytrain_path.exists() and xtest_path.exists() and ytest_path.exists():
        X_train = pd.read_csv(xtrain_path)
        y_train = pd.read_csv(ytrain_path).squeeze()
        X_test = pd.read_csv(xtest_path)
        y_test = pd.read_csv(ytest_path).squeeze()
    else:
        raise FileNotFoundError(
            "Fichiers de données absents. Exécutez d'abord le prétraitement pour générer les fichiers de train/test."
        )

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, save_model: bool = True):
    """Entraîne un modèle XGBoost et le sauvegarde."""

    selected_features = [
        "Schiller",
        "Age",
        "Hormonal Contraceptives",
        "Num of pregnancies",
        "Number of sexual partners"
    ]

    X_train = X_train[selected_features]

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    if save_model:
        root_dir = Path(__file__).resolve().parent.parent
        models_dir = root_dir / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "xgboost_model.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Modèle sauvegardé dans : {model_path}")

    return model


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    train_xgboost(X_train, y_train)


if __name__ == "__main__":
    main()
