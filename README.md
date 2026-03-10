


# Projet Cancer du Col de l'Utérus
Application d'aide à la décision médicale pour évaluer le risque de cancer du col de l'utérus à partir de facteurs comportementaux et médicaux. Utilise un modèle de machine learning explicable avec SHAP.
## Équipe
- Lilicat22 : Structure, coordination, tests
- Lehvnxx : Data Processing & EDA
- Wilfried-23 : Modèle Random Forest
- JaurXs : Modèle XGBoost
- Hansem01-boot : Modèle SVM & Interface

## Structure du projet
```
cervical-cancer-risk/
├── .github/workflows/    # CI/CD
├── notebooks/            # EDA
├── src/                  # Code source
│   ├── data_processing.py
│   ├── train_model.py
│   └── explain.py
├── app/                  # Interface
│   └── app.py
├── tests/                # Tests
│   └── test_data_processing.py
├── requirements.txt      # Dépendances
└── README.md             # Documentation

```

## Installation
```bash
git clone https://github.com/Lilicat22/Cancer-CODING-WEEK.git
cd Cancer-CODING-WEEK
pip install -r requirements.txt
```

## Utilisation

```bash
# Entraîner les modèles
python src/train_model.py

# Lancer l'application
streamlit run app/app.py
```


## Branches
- `main` : version stable
- `feature/eda-p2` : pour Lehvnxx (EDA)
- `feature/random-forest-p3` : pour Wilfried-23 (Random Forest)
- `feature/xgboost-p4` : pour JaurXs (XGBoost)
- `feature/svm-interface-p5` : pour Hansem01-boot (SVM + Interface)
- `feature/tests-p1` : pour Lilicat22 (tests et coordination)


## Comment faire :

### 1. Ouvre le fichier
```bash
nano README.md
```
