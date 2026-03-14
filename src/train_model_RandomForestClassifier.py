import joblib

from sklearn.ensemble import RandomForestClassifier

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

    joblib.dump(model, "model_random_forest.pkl")

    return model


if __name__ == "__main__":
    train_model()