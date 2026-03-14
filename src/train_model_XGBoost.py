import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

# chemins du projet
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
data_dir = root_dir / "data"
models_dir = root_dir / "models"

# créer le dossier models s'il n'existe pas
models_dir.mkdir(exist_ok=True)

# Charger les données
X_train = pd.read_csv(data_dir / "X_train_cleaned.csv")
y_train = pd.read_csv(data_dir / "y_train_cleaned.csv").squeeze()

X_test = pd.read_csv(data_dir / "X_test_cleaned.csv")
y_test = pd.read_csv(data_dir / "y_test_cleaned.csv").squeeze()

# 🔹 Features sélectionnées
selected_features = [
    "Schiller",
    "Age",
    "Hormonal Contraceptives",
    "Num of pregnancies",
    "Number of sexual partners"
]

# Garder uniquement ces colonnes
X_train = X_train[selected_features]
X_test = X_test[selected_features]

print("Features utilisées :", X_train.columns.tolist())

# Création du modèle
model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,
    random_state=42
)

# Entraînement
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("\nClassification report :")
print(classification_report(y_test, y_pred))

# 🔹 Sauvegarde du modèle dans models
model_path = models_dir / "xgboost_model.pkl"

joblib.dump(model, model_path)

print(f"\n✅ Modèle sauvegardé dans : {model_path}")