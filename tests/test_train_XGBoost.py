import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from pathlib import Path

def train_xgboost():
    # 1. Chemins des fichiers (ajusté selon vos dossiers précédents)
    data_dir = Path.cwd() / "data" / "data"
    
    # 2. Chargement des données nettoyées
    X_train = pd.read_csv(data_dir / 'X_train_cleaned.csv')
    y_train = pd.read_csv(data_dir / 'y_train_cleaned.csv').squeeze()
    X_test = pd.read_csv(data_dir / 'X_test_cleaned.csv')
    y_test = pd.read_csv(data_dir / 'y_test_cleaned.csv').squeeze()

    print(f"Échantillon d'entraînement : {X_train.shape[0]} patientes")

    # 3. Gestion du déséquilibre des classes (85% vs 15%) 
    # Calcul du poids pour donner plus d'importance à la classe minoritaire (At risk)
    # Formule : scale_pos_weight = count(negative) / count(positive)
    counter_pos = sum(y_train == 1)
    counter_neg = sum(y_train == 0)
    scale_weight = counter_neg / counter_pos

    # 4. Configuration du modèle XGBoost [cite: 39]
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_weight, # Technique de pondération 
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # 5. Entraînement
    print("Entraînement du modèle XGBoost en cours...")
    model.fit(X_train, y_train)

    # 6. Évaluation sur le jeu de test 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*30)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*30)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    trained_model = train_xgboost()
    