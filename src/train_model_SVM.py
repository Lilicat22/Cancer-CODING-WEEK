
# src/train_model.py

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os


def train_svm(X_train, y_train):

    """
    Entraine le modèle SVM et le sauvegarde
    """

    svm_pipeline = Pipeline([
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