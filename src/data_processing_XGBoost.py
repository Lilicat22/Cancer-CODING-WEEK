import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
import numpy as np

# 1. LOCALISATION AUTOMATIQUE
# __file__ est le chemin vers ce script
script_dir = Path(__file__).resolve().parent
# On remonte d'un niveau pour sortir de 'src' et arriver à la racine 'Cancer-CODING-WEEK'
root_dir = script_dir.parent
data_dir = root_dir / "data"

print("--- 🔍 DIAGNOSTIC DES CHEMINS ---")
print(f"📍 Dossier du script : {script_dir}")
print(f"📁 Racine du projet  : {root_dir}")
print(f"📦 Dossier data visé : {data_dir}")

# 2. VÉRIFICATION RÉELLE DU CONTENU
if data_dir.exists():
    print(f"✅ Dossier 'data' trouvé !")
    print(f"📄 Fichiers vus : {os.listdir(data_dir)}")
else:
    print(f"❌ Erreur : Le dossier 'data' n'existe pas à l'adresse : {data_dir}")
    exit(1)  # Arrêter si le dossier n'existe pas

# 3. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES BRUTES
raw_data_path = data_dir / 'risk_factors_cervical_cancer.csv'
if not raw_data_path.exists():
    print(f"❌ Erreur : Le fichier 'risk_factors_cervical_cancer.csv' n'existe pas dans {data_dir}")
    exit(1)

# Charger les données brutes
data = pd.read_csv(raw_data_path)

# Remplacer les '?' par NaN et remplir avec la médiane
data = data.replace('?', np.nan)
data = data.fillna(data.median(numeric_only=True))

# Assumer que la colonne cible est 'Biopsy' (ajuster si nécessaire)
if 'Biopsy' not in data.columns:
    print("❌ Erreur : Colonne 'Biopsy' non trouvée dans les données.")
    exit(1)

X = data.drop("Biopsy", axis=1)
y = data["Biopsy"]

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarder les données nettoyées
X_train.to_csv(data_dir / 'X_train_cleaned.csv', index=False)
y_train.to_csv(data_dir / 'y_train_cleaned.csv', index=False)
X_test.to_csv(data_dir / 'X_test_cleaned.csv', index=False)
y_test.to_csv(data_dir / 'y_test_cleaned.csv', index=False)

print("✅ Données nettoyées et sauvegardées avec succès !")

# 4. CHARGEMENT SÉCURISÉ (maintenant que les fichiers existent)
try:
    # On construit le chemin complet vers chaque fichier
    X_train = pd.read_csv(data_dir / 'X_train_cleaned.csv')
    print("✅ X_train chargé avec succès !")
    
    # Si le premier passe, on peut charger les autres
    Y_train = pd.read_csv(data_dir / 'y_train_cleaned.csv').squeeze()
    X_test = pd.read_csv(data_dir / 'X_test_cleaned.csv')
    Y_test = pd.read_csv(data_dir / 'y_test_cleaned.csv').squeeze()
    
    print("\n🚀 TOUT EST PRÊT ! Voici un aperçu des données :")
    print(X_train.head())

except Exception as e:
    print(f"\n💥 ÉCHEC FINAL : {e}")
    
import numpy as np

def optimize_memory(df):

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            else:

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2

    print(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")

    return df