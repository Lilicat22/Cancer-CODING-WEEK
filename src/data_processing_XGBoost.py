import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# chemin des données
data_dir = Path("data")
raw_path = data_dir / "risk_factors_cervical_cancer.csv"


def load_data():
    """Charge le dataset brut et sépare X et y"""
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {raw_path}")

    df = pd.read_csv(raw_path)

    X = df.drop(columns=["Biopsy"])
    y = df["Biopsy"]

    return X, y


# chemins des fichiers splittés
xtrain_path = data_dir / "X_train_cleaned.csv"
ytrain_path = data_dir / "y_train_cleaned.csv"
xtest_path = data_dir / "X_test_cleaned.csv"
ytest_path = data_dir / "y_test_cleaned.csv"


if xtrain_path.exists() and ytrain_path.exists() and xtest_path.exists() and ytest_path.exists():

    print("ℹ️ Fichiers splittés déjà présents. Chargement...")

    X_train = pd.read_csv(xtrain_path)
    y_train = pd.read_csv(ytrain_path).squeeze()

    X_test = pd.read_csv(xtest_path)
    y_test = pd.read_csv(ytest_path).squeeze()

else:

    print("ℹ️ Chargement du dataset brut")

    X, y = load_data()

    print("ℹ️ Création du train/test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(xtrain_path, index=False)
    y_train.to_csv(ytrain_path, index=False)

    X_test.to_csv(xtest_path, index=False)
    y_test.to_csv(ytest_path, index=False)

    print("✅ Données sauvegardées avec succès")