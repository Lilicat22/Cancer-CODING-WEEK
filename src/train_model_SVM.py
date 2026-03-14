
# src/train_model.py

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
from data_processing_SVM import load_data


def train_svm(X_train, y_train):
    """
    Entraine le modèle SVM avec les features Age et Number of sexual partners
    et le sauvegarde
    """

    # Sélection des features
    X_train = X_train[["Age", "Number of sexual partners"]]

    svm_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True))
    ])

    svm_pipeline.fit(X_train, y_train)

    # chemin du dossier models
    base_path = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_path, "models")

    os.makedirs(model_path, exist_ok=True)

    model_file = os.path.join(model_path, "svm_model.pkl")

    joblib.dump(svm_pipeline, model_file)

    print("Model sauvegardé dans :", model_file)

    return svm_pipeline


# Chargement des données
X_train, X_test, y_train, y_test = load_data()

# Entraînement du modèle
train_svm(X_train, y_train)