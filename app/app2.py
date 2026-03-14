import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(page_title="Risque Cancer du Col", layout="wide")

# ============================================
# CHARGEMENT DES RESSOURCES (avec cache)
# ============================================
@st.cache_resource
def load_models():
    """Charge tous les modèles disponibles et le scaler."""
    models = {}
    # Chemins vers les modèles (à adapter selon vos fichiers)
    model_paths = {
        'Random Forest': 'app/models/rf_model.pkl',
        'XGBoost': 'app/models/xgb_model.pkl',
        'SVM': 'app/models/svm_model.pkl'
    }
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            # Si le modèle n'existe pas, on crée un modèle factice pour tester
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy='prior')
            # On l'entraîne avec des données bidon (juste pour avoir predict_proba)
            X_dummy = np.random.rand(100, 35)
            y_dummy = np.array([0]*85 + [1]*15)
            dummy.fit(X_dummy, y_dummy)
            models[name] = dummy
    scaler = joblib.load('app/scaler.pkl') if os.path.exists('app/scaler.pkl') else None
    return models, scaler

@st.cache_data
def load_feature_names():
    """Charge la liste des noms de features."""
    if os.path.exists('app/feature_names.txt'):
        with open('app/feature_names.txt', 'r') as f:
            return [line.strip() for line in f]
    else:
        # Fallback : 35 features génériques
        return [f'feature_{i}' for i in range(35)]

@st.cache_data
def load_model_metrics():
    """Charge les métriques des modèles (pour admin)."""
    if os.path.exists('app/model_metrics.json'):
        with open('app/model_metrics.json', 'r') as f:
            return json.load(f)
    else:
        # Métriques factices
        return {
            'Random Forest': {'ROC-AUC': 0.98, 'Accuracy': 0.95, 'Precision': 0.92, 'Recall': 0.88, 'F1': 0.90},
            'XGBoost': {'ROC-AUC': 0.97, 'Accuracy': 0.94, 'Precision': 0.91, 'Recall': 0.87, 'F1': 0.89},
            'SVM': {'ROC-AUC': 0.96, 'Accuracy': 0.93, 'Precision': 0.90, 'Recall': 0.85, 'F1': 0.87}
        }

def load_validations():
    """Charge les validations des médecins depuis un fichier CSV."""
    if os.path.exists('app/validations.csv'):
        return pd.read_csv('app/validations.csv')
    else:
        return pd.DataFrame(columns=['timestamp', 'doctor', 'model', 'comment', 'rating', 'prediction'])

def save_validation(doctor, model, comment, rating, prediction):
    """Sauvegarde une validation dans le CSV."""
    df = load_validations()
    new_row = pd.DataFrame({
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'doctor': [doctor],
        'model': [model],
        'comment': [comment],
        'rating': [rating],
        'prediction': [prediction]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('app/validations.csv', index=False)

# ============================================
# GESTION DES RÔLES
# ============================================
st.sidebar.title("👤 Profil utilisateur")
role = st.sidebar.selectbox("Choisissez votre rôle", ["Visiteur", "Médecin", "Administrateur"])

# Si médecin, on demande son nom (pour les validations)
doctor_name = None
if role == "Médecin":
    doctor_name = st.sidebar.text_input("Votre nom (pour validation)", value="Dr. X")

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
models, scaler = load_models()
feature_names = load_feature_names()
model_metrics = load_model_metrics()
validations = load_validations()

# ============================================
# TITRE PRINCIPAL
# ============================================
st.title("🔬 Aide à la décision médicale - Cancer du col de l'utérus")
st.markdown("---")

# ============================================
# FORMULAIRE DE SAISIE
# ============================================
st.header("📋 Caractéristiques de la patiente")

# On répartit les 35 features en deux colonnes pour un affichage compact
col1, col2 = st.columns(2)

input_data = {}
with col1:
    for i, feat in enumerate(feature_names[:18]):
        # Déterminer le type de champ selon la feature (binaire ou continue)
        if any(x in feat for x in ['STDs', 'Dx', 'Hinselmann', 'Schiller', 'Citology']):
            input_data[feat] = st.selectbox(f"{feat}", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui", key=f"in_{i}")
        else:
            input_data[feat] = st.number_input(f"{feat}", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"in_{i}")
with col2:
    for i, feat in enumerate(feature_names[18:]):
        if any(x in feat for x in ['STDs', 'Dx', 'Hinselmann', 'Schiller', 'Citology']):
            input_data[feat] = st.selectbox(f"{feat}", [0, 1], format_func=lambda x: "Non" if x==0 else "Oui", key=f"in_{i+18}")
        else:
            input_data[feat] = st.number_input(f"{feat}", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"in_{i+18}")

# ============================================
# SÉLECTION DU MODÈLE
# ============================================
st.header("🤖 Choix du modèle")
model_choice = st.selectbox("Sélectionnez un modèle à utiliser", list(models.keys()))

# ============================================
# BOUTON DE PRÉDICTION
# ============================================
if st.button("🔮 Prédire", type="primary"):
    # Création du DataFrame à partir des saisies
    df_input = pd.DataFrame([input_data])
    # S'assurer de l'ordre des colonnes
    df_input = df_input[feature_names]
    
    # Appliquer le scaler si nécessaire (pour SVM notamment)
    model = models[model_choice]
    if scaler is not None and model_choice == 'SVM':
        X_scaled = scaler.transform(df_input)
        proba = model.predict_proba(X_scaled)[0, 1]
    else:
        proba = model.predict_proba(df_input)[0, 1]
    
    pred_class = 1 if proba >= 0.5 else 0
    
    # Affichage du résultat
    st.subheader("📊 Résultat de la prédiction")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Probabilité de risque", f"{proba:.2%}")
    with col_res2:
        if pred_class == 1:
            st.error("⚠️ **Risque élevé** (Biopsy positive)")
        else:
            st.success("✅ **Risque faible** (Biopsy négative)")
    
    # Barre de progression
    st.progress(proba)
    
    # ============================================
    # EXPLICATIONS SHAP
    # ============================================
    st.subheader("🔍 Explication de la prédiction (SHAP)")
    # Création de l'explainer selon le type de modèle
    if 'XGBoost' in model_choice or 'Random Forest' in model_choice:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)
        # Si classification binaire, shap_values est une liste, on prend la classe 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
    else:
        # Pour SVM, on utilise KernelExplainer (plus lent, on prend un échantillon)
        st.warning("SHAP pour SVM peut être lent. Utilisation d'un échantillon.")
        # On crée un échantillon d'arrière-plan (ici on prend les données d'entraînement factices)
        background = np.random.rand(50, 35)  # à remplacer par de vraies données si possible
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(df_input)[1]  # pour la classe 1
        expected_value = explainer.expected_value[1]
    
    # Force plot
    fig, ax = plt.subplots(figsize=(10, 2))
    shap.force_plot(expected_value, shap_values[0], df_input.iloc[0, :], feature_names=feature_names, matplotlib=True, show=False, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary plot (feature importance locale)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, df_input, feature_names=feature_names, show=False)
    st.pyplot(fig2)
    
    # ============================================
    # FONCTIONNALITÉS SPÉCIFIQUES AU RÔLE
    # ============================================
    if role == "Médecin" and doctor_name:
        st.markdown("---")
        st.subheader("🩺 Validation médicale")
        with st.form("validation_form"):
            rating = st.slider("Évaluation du modèle pour ce cas (1 = pas fiable, 5 = très fiable)", 1, 5, 3)
            comment = st.text_area("Commentaire (optionnel)")
            submitted = st.form_submit_button("Soumettre la validation")
            if submitted:
                save_validation(doctor_name, model_choice, comment, rating, pred_class)
                st.success("Merci pour votre retour !")
        
        # Affichage des validations précédentes
        st.subheader("📝 Avis des médecins sur ce modèle")
        if not validations.empty:
            # Filtrer par modèle si on veut
            st.dataframe(validations[validations['model'] == model_choice][['timestamp', 'doctor', 'rating', 'comment']])
        else:
            st.info("Aucune validation pour l'instant.")
    
    elif role == "Administrateur":
        st.markdown("---")
        st.subheader("📈 Performances des modèles")
        # Afficher les métriques stockées
        metrics_df = pd.DataFrame(model_metrics).T
        st.dataframe(metrics_df)
        
        # Option : voir toutes les validations
        st.subheader("📋 Toutes les validations")
        if not validations.empty:
            st.dataframe(validations)
        else:
            st.info("Aucune validation.")

# ============================================
# PIED DE PAGE
# ============================================
st.markdown("---")
st.caption("Application développée dans le cadre du projet de décision médicale. Les modèles sont explicables via SHAP.")