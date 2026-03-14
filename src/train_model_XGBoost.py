import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ...existing code...
script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent
data_dir = root_dir / "data"

# Charger les données nettoyées
X_train = pd.read_csv(data_dir / 'X_train_cleaned.csv')
y_train = pd.read_csv(data_dir / 'y_train_cleaned.csv').squeeze()
X_test = pd.read_csv(data_dir / 'X_test_cleaned.csv')
y_test = pd.read_csv(data_dir / 'y_test_cleaned.csv').squeeze()

# Créer le modèle XGBoost
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)

# Entraîner le modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Sauvegarder le modèle pour réutilisation (streamlit/flask)
model.save_model(data_dir / 'xgboost_model.json')
joblib.dump(model, data_dir / 'xgboost_model.joblib')

print("Modèle sauvegardé dans data/xgboost_model.json et data/xgboost_model.joblib")
