

```markdown
# Projet Cancer du Col de l'Utérus
Application d'aide à la décision médicale pour évaluer le risque de cancer du col de l'utérus à partir de facteurs comportementaux et médicaux. Utilise un modèle de machine learning explicable avec SHAP.
## Équipe
- Personne 1 (Lilicat22) : Structure, coordination, tests
- Personne 2 : Data Processing & EDA
- Personne 3 : Modèle Random Forest
- Personne 4 : Modèle XGBoost
- Personne 5 : Modèle SVM & Interface

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

Utilisation

```bash
# Entraîner les modèles
python src/train_model.py

# Lancer l'application
streamlit run app/app.py
```

Branches

· main : version stable
· feature/eda : Personne 2
· feature/random-forest : Personne 3
· feature/xgboost : Personne 4
· feature/svm-interface : Personne 5

```

---

## Comment faire :

### 1. Ouvre le fichier
```bash
nano README.md
```
---

