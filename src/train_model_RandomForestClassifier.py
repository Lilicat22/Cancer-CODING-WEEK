import os
import joblib

from sklearn.ensemble import RandomForestClassifier

try:
    from .data_processing_RandomForestClassifier import load_data
except ImportError:  # pragma: no cover
    from data_processing_RandomForestClassifier import load_data


def train_model():

    X_train, X_test, y_train, y_test = load_data()

    # Sélection des features utilisées pour l'entraînement
    X_train = X_train[["Age", "Number of sexual partners"]]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    base_path = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_path, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_random_forest.pkl")
    joblib.dump(model, model_path)

    return model


if __name__ == "__main__":
    train_model()