# Projet Cancer du Col de l'Utérus

Outil d'aide à la décision permettant d'évaluer le risque de cancer du col de l'utérus à partir de données cliniques et comportementales. Le projet propose plusieurs modèles de machine learning (Random Forest, XGBoost, SVM) et utilise SHAP pour fournir des explications locales/globales.

---

## � Équipe

- **Lilicat22** : Structure, coordination, tests
- **Lehvnxx** : Data Processing & EDA
- **Wilfried-23** : Modèle Random Forest
- **JaurXs** : Modèle XGBoost
- **Hansem01-boot** : Modèle SVM & Interface

---

## 🔍 Structure du projet

```
Cancer-CODING-WEEK/
├── app/                        # Interface (Streamlit, en cours)
├── data/                       # Données sources + données nettoyées / splittées
├── models/                     # Modèles entraînés + packages d'évaluation (metrics + explainers)
├── notebooks/                  # Notebooks d'EDA & SHAP
├── src/                        # Code source
│   ├── data_processing_*.py    # Chargement / prétraitement des données
│   ├── train_model_*.py        # Entraînement des modèles (SVM / RF / XGBoost)
│   ├── evaluate_model_*.py     # Évaluation & explication (SHAP)
│   └── ...
├── tests/                      # Tests unitaires (pytest)
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation (ce fichier)
```

---

## ✅ Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🧩 Données attendues

Le projet s'appuie sur des fichiers CSV nettoyés et splittés :

- `data/X_train_cleaned.csv`
- `data/y_train_cleaned.csv`
- `data/X_test_cleaned.csv`
- `data/y_test_cleaned.csv`

Si ces fichiers ne sont pas présents, ils peuvent être générés à partir du jeu de données brut (`data/risk_factors_cervical_cancer.csv`) en exécutant le script de prétraitement (actuellement implémenté dans `src/data_processing_XGBoost.py`).

---

## 🧠 Entraînement des modèles

Les scripts d'entraînement sont importables (pas de side-effect à l'import), et ils écrivent les modèles dans `models/`.

### SVM

```bash
python -m src.train_model_SVM
```

### Random Forest

```bash
python -m src.train_model_RandomForestClassifier
```

### XGBoost

```bash
python -m src.train_model_XGBoost
```

---

## 📊 Évaluation et explication (SHAP)

Chaque script d'évaluation charge le modèle entraîné depuis `models/`, calcule des métriques et génère un package dédié (prédictions, confusion matrix, explainer SHAP) dans `models/`.

### SVM

```bash
python -m src.evaluate_model_SVM
```

### Random Forest

```bash
python -m src.evaluate_model_RandomForestClassifier
```

### XGBoost

```bash
python -m src.evaluate_model_XGBoost
```

---

## 🧪 Tests

Pour exécuter la suite de tests :

```bash
python -m pytest
```

Les tests couvrent le pipeline SVM (prétraitement, entraînement, évaluation) et la validation des fonctions de chargement des données.

---

## 🌟 Exemple d'usage

1. **Préparer les données** : Assurez-vous que les fichiers CSV nettoyés sont présents dans `data/`.

2. **Entraîner un modèle** :
   ```bash
   python -m src.train_model_SVM
   ```

3. **Évaluer le modèle** :
   ```bash
   python -m src.evaluate_model_SVM
   ```

4. **Analyser avec SHAP** : Les explainers sont sauvegardés dans les packages d'évaluation (`models/*_package.pkl`).

---

## 📈 Métriques attendues

- **Accuracy** : ~85-95% selon le modèle et les données.
- **Confusion Matrix** : Disponible dans les packages d'évaluation.
- **SHAP Values** : Pour expliquer les prédictions individuelles.

---

## 🛠️ Branches

- `main` : Version stable
- `feature/eda-p2` : Pour Lehvnxx (EDA)
- `feature/random-forest-p3` : Pour Wilfried-23 (Random Forest)
- `feature/xgboost-p4` : Pour JaurXs (XGBoost)
- `feature/svm-interface-p5` : Pour Hansem01-boot (SVM + Interface)
- `feature/tests-p1` : Pour Lilicat22 (tests et coordination)

---

## 🤝 Contribuer

- Créez une branche dédiée à votre fonctionnalité.
- Ajoutez/modifiez des tests si nécessaire.
- Exécutez `python -m pytest` avant de soumettre une PR.

---

## 📚 Ressources

- [SHAP Documentation](https://github.com/slundberg/shap)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

## ⚠️ Notes

- Les modèles sauvegardés (fichiers `.pkl`) sont placés dans `models/`.
- Les scripts `src/train_model_*.py` sont conçus pour être réutilisés dans des pipelines ou dans des tests.
- Pour ajouter un nouveau modèle, créez :
  - `src/train_model_<nom>.py`
  - `src/evaluate_model_<nom>.py`
  - (optionnel) un test `tests/test_<nom>_model.py`

---

## 🧑‍🤝‍🧑 Contribuer

- Créez une branche dédiée à votre fonctionnalité.
- Ajoutez/modifiez des tests si nécessaire.
- Exécutez `python -m pytest` avant de soumettre une PR.

---

## 📚 Ressources

- SHAP : https://github.com/slundberg/shap
- scikit-learn : https://scikit-learn.org/
