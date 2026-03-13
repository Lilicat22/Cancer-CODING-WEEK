# La prédiction nécesite ces bibliothèques et 1 à 6
import numpy as np
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Chargement de la dataset-1
data = pd.read_csv("data/data.csv")

#Convertir en numérique (les valeurs non numériques deviennent NaN)
data = data.apply(pd.to_numeric, errors='coerce')

# Calculer le pourcentage de valeurs manquantes
missing_ratio = data.isnull().mean()

# Supprimer les colonnes avec plus de 60% de valeurs manquantes
data = data.loc[:, missing_ratio < 0.6]

# Remplacer les valeurs manquantes par la médiane-2
data = data.fillna(data.median())

print("Dataset nettoyé :", data.shape)

# Variables explicatives-3
X = data.drop("Biopsy", axis=1)

# Variable cible-4
y = data["Biopsy"]

# Séparation des données-5
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Création du modèle-6
model = RandomForestClassifier(n_estimators=200,random_state=42)

# Entraînement du modèle
model.fit(X_train, y_train)


# Sauvegarde du modèle
import joblib
joblib.dump(model, "rf_model.pkl")
print("Modèle sauvegardé dans rf_model.pkl")
