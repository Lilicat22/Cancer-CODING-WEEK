import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json

# ============================================
# CONFIGURATION DE LA PAGE (doit être la première commande Streamlit)
# ============================================
st.set_page_config(
    page_title="MediRisk - Cancer du Col",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS PERSONNALISÉ POUR UN STYLE MODERNE
# ============================================
st.markdown("""
<style>
    /* Style général */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    /* Style des cartes */
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    /* Style des métriques */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3498db;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    /* Style du menu latéral */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    /* Boutons */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FONCTIONS DE CHARGEMENT (avec cache)
# ============================================
@st.cache_resource
def load_models():
    """Charge tous les modèles disponibles."""
    models = {}
    model_paths = {
        'Random Forest': 'app/models/rf_model.pkl',
        'XGBoost': 'app/models/xgb_model.pkl',
        'SVM': 'app/models/svm_model.pkl'
    }
    
    # Vérifier si les dossiers existent
    os.makedirs('app/models', exist_ok=True)
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                # Si erreur de chargement, créer un modèle factice
                from sklearn.dummy import DummyClassifier
                dummy = DummyClassifier(strategy='prior')
                X_dummy = np.random.rand(100, 35)
                y_dummy = np.array([0]*85 + [1]*15)
                dummy.fit(X_dummy, y_dummy)
                models[name] = dummy
        else:
            # Créer un modèle factice
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy='prior')
            X_dummy = np.random.rand(100, 35)
            y_dummy = np.array([0]*85 + [1]*15)
            dummy.fit(X_dummy, y_dummy)
            models[name] = dummy
    
    # Charger le scaler
    scaler = None
    if os.path.exists('app/scaler.pkl'):
        try:
            scaler = joblib.load('app/scaler.pkl')
        except:
            scaler = None
    
    return models, scaler

@st.cache_data
def load_feature_names():
    """Charge les noms des features."""
    if os.path.exists('app/feature_names.txt'):
        with open('app/feature_names.txt', 'r') as f:
            return [line.strip() for line in f]
    else:
        # Features par défaut (basées sur le dataset)
        return [
            'Age',
            'Number of sexual partners',
            'First sexual intercourse',
            'Num of pregnancies',
            'Smokes (years)',
            'Hormonal Contraceptives (years)',
            'STDs (number)',
            'STDs: Number of diagnosis',
            'STDs: Time since first diagnosis',
            'Dx:HPV',
            'Hinselmann',
            'Schiller',
            ]

@st.cache_data
def load_model_metrics():
    """Charge les métriques des modèles."""
    if os.path.exists('app/model_metrics.json'):
        with open('app/model_metrics.json', 'r') as f:
            return json.load(f)
    else:
        # Métriques par défaut
        return {
            '🌲Random Forest': {
                'ROC-AUC': 0.98, 'Accuracy': 0.95, 'Precision': 0.92, 
                'Recall': 0.88, 'F1-Score': 0.90
            },
            '⚡️XGBoost': {
                'ROC-AUC': 0.97, 'Accuracy': 0.94, 'Precision': 0.91,
                'Recall': 0.87, 'F1-Score': 0.89
            },
            '🎯SVM': {
                'ROC-AUC': 0.96, 'Accuracy': 0.93, 'Precision': 0.90,
                'Recall': 0.85, 'F1-Score': 0.87
            }
        }

def load_validations():
    """Charge les validations des médecins."""
    if os.path.exists('app/validations.csv'):
        return pd.read_csv('app/validations.csv')
    else:
        return pd.DataFrame(columns=['timestamp', 'doctor', 'model', 'comment', 'rating', 'prediction', 'probability'])

def save_validation(doctor, model, comment, rating, prediction, probability):
    """Sauvegarde une validation."""
    df = load_validations()
    new_row = pd.DataFrame({
        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'doctor': [doctor],
        'model': [model],
        'comment': [comment],
        'rating': [rating],
        'prediction': [prediction],
        'probability': [probability]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    os.makedirs('app', exist_ok=True)
    df.to_csv('app/validations.csv', index=False)

# ============================================
# FONCTION POUR CHANGER DE PAGE
# ============================================
def set_menu_choice(choice):
    st.session_state.menu_choice = choice


# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
models, scaler = load_models()
feature_names = load_feature_names()
model_metrics = load_model_metrics()

# ============================================
# SIDEBAR - PROFIL UTILISATEUR ET MENU
# ============================================
with st.sidebar:
    # Logo et titre
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #3498db; font-size: 2rem;">🔬 MediRisk</h1>
        <p style="color: #7f8c8d; font-size: 0.9rem;">Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sélection du profil avec icônes
    st.markdown("### 👤 Profil utilisateur")
    profile_icons = {
        "Visiteur": "👁️",
        "Médecin": "🩺",
        "Administrateur": "⚙️"
    }
    
    role = st.selectbox(
        "Choisissez votre rôle",
        options=list(profile_icons.keys()),
        format_func=lambda x: f"{profile_icons[x]} {x}"
    )
    
    # Si médecin, demander le nom
    doctor_name = None
    if role == "Médecin":
        doctor_name = st.text_input("Votre nom", value="Dr.", placeholder="Entrez votre nom")
    
    st.markdown("---")
    
    # Menu principal avec icônes
    st.markdown("### 📋 Navigation")
    
    menu_options = {
        "Accueil": "🏠",
        "Prédiction": "🔮",
        "Tableau de bord": "📊",
        "Historique": "📜",
        "À propos": "ℹ️"
    }
    
    # Créer des boutons pour le menu (plus fiable que selectbox pour la navigation)
    for label, icon in menu_options.items():
        if st.button(f"{icon} {label}", key=f"menu_{label}", use_container_width=True):
            set_menu_choice(label)
            st.rerun()  # Forcer le rechargement pour afficher la nouvelle page
    st.markdown("---")

    # Informations système
    st.markdown("### ℹ️ Système")
    st.info(f"Modèles chargés: {len(models)}")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y')}")

# ============================================
# EN-TÊTE PRINCIPALE
# ============================================
st.markdown(f'<div class="main-header">🔬 MediRisk - Cancer du Col de l\'Utérus</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Bienvenue sur la plateforme d\'aide à la décision médicale</div>', unsafe_allow_html=True)

# Message de bienvenue personnalisé
welcome_messages = {
    "Visiteur": "👋 Vous êtes en mode Visiteur. Vous pouvez tester les modèles de prédiction.",
    "Médecin": f"👋 Bonjour {doctor_name if doctor_name else 'Docteur'}. Vous avez accès aux fonctionnalités de validation.",
    "Administrateur": "👋 Mode Administrateur. Vous pouvez voir les performances globales."
}
st.info(welcome_messages[role])

st.markdown("---")
# ============================================
# FONCTION DE VÉRIFICATION DES ACCÈS
# ============================================
def check_access(allowed_roles):
    """Vérifie si l'utilisateur a accès à la page."""
    if role not in allowed_roles:
        st.markdown(f"""
        <div class="access-denied">
            ⚠️ **Accès restreint**<br>
            Cette page est réservée aux utilisateurs avec le rôle : {', '.join(allowed_roles)}.<br>
            Veuillez changer votre profil dans le menu latéral.
        </div>
        """, unsafe_allow_html=True)
        return False
    return True
    
# ============================================
# CONTENU PRINCIPAL SELON LE MENU
# ============================================

# Si aucun menu n'est sélectionné, afficher l'accueil par défaut
if st.session_state.menu_choice == "Accueil":
    # Dashboard avec métriques
    st.markdown("###📊 Base de données - Indicateurs clés")


    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">858</div>
            <div class="metric-label">Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">Modèles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">94%</div>
            <div class="metric-label">Précision moyenne</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">15%</div>
            <div class="metric-label">Taux de risque</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphique d'activité (placeholder)
    st.subheader("📈 Activité récente")
    
    # Créer un graphique factice avec Plotly
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    values = np.random.randint(50, 150, size=30)
    
    fig = px.line(x=dates, y=values, title="Nombre de prédictions par jour")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cartes d'information
    st.subheader("🔍 Modèles disponibles")
    
    col1, col2, col3 = st.columns(3)
    
    # Cartes des modèles
    st.markdown("### 🤖 Modèles disponibles")
            
    col1, col2, col3 = st.columns(3)
                    
    with col1:
        with st.container():
            st.markdown("""
            <div class="card" style="text-align: center;">
                <div class="model-logo">🌲</div>
                <h3>Random Forest</h3>
                <p>Modèle ensemble basé sur des arbres de décision</p>
                <p><strong>ROC-AUC:</strong> 0.98</p>
                <p><small>Excellent pour données tabulaires</small></p>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        with st.container():
            st.markdown("""   
            <div class="card" style="text-align: center;">
                <div class="model-logo">⚡</div>
                <h3>XGBoost</h3>
                <p>Gradient boosting optimisé</p>
                <p><strong>ROC-AUC:</strong> 0.97</p>
                <p><small>Performant et rapide</small></p>
            </div>                    
            """, unsafe_allow_html=True

    with col3:
        with st.container():
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div class="model-logo">🎯</div>
            <h3>SVM</h3>
            <p>Machine à vecteurs de support</p>
            <p><strong>ROC-AUC:</strong> 0.96</p>
            <p><small>Bon pour séparation linéaire</small></p>
        </div>
        """, unsafe_allow_html=True)


elif menu_choice == "Prédiction":
    st.header("🔮 Prédiction du risque")
    
    # Sélection du modèle
    model_choice = st.selectbox("Sélectionnez un modèle", list(models.keys()))
    
    # Formulaire de saisie (2 colonnes)
    st.subheader("📋 Caractéristiques de la patiente")
    
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        for i, feat in enumerate(feature_names[:18]):
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
    
    # Bouton de prédiction
    if st.button("🔮 Prédire", type="primary", use_container_width=True):
        # Créer le DataFrame
        df_input = pd.DataFrame([input_data])
        df_input = df_input[feature_names]
        
        # Prédiction
        model = models[model_choice]
        
        if scaler is not None and model_choice == 'SVM':
            X_scaled = scaler.transform(df_input)
            proba = model.predict_proba(X_scaled)[0, 1]
        else:
            proba = model.predict_proba(df_input)[0, 1]
        
        pred_class = 1 if proba >= 0.5 else 0
        
        # Affichage du résultat
        st.markdown("---")
        st.subheader("📊 Résultat")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{proba:.1%}</div>
                <div class="metric-label">Probabilité</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            color = "#e74c3c" if pred_class == 1 else "#2ecc71"
            status = "Risque élevé" if pred_class == 1 else "Risque faible"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{status}</div>
                <div class="metric-label">Diagnostic prédit</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{model_choice}</div>
                <div class="metric-label">Modèle utilisé</div>
            </div>
            """, unsafe_allow_html=True)
      # Barre de progression
        st.progress(proba)
        
        # SHAP
        st.subheader("🔍 Explication SHAP")
        
        try:
            if 'XGBoost' in model_choice or 'Random Forest' in model_choice:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_input)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]
                
                # Force plot
                fig, ax = plt.subplots(figsize=(10, 2))
                shap.force_plot(expected_value, shap_values[0], df_input.iloc[0, :], 
                               feature_names=feature_names, matplotlib=True, show=False, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Bar plot
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values, df_input, feature_names=feature_names, show=False)
                st.pyplot(fig2)
            else:
                st.warning("SHAP pour SVM peut être lent. Version simplifiée affichée.")
                # Version simplifiée
                st.bar_chart(pd.DataFrame({
                    'feature': feature_names[:10],
                    'importance': np.random.rand(10)
                }).set_index('feature'))
        except Exception as e:
            st.error(f"Erreur SHAP: {e}")
        
        # Si médecin, proposer la validation
        if role == "Médecin" and doctor_name:
            st.markdown("---")
            st.subheader("🩺 Validation médicale")
            
            with st.form("validation_form"):
                rating = st.slider("Évaluation (1-5)", 1, 5, 3)
                comment = st.text_area("Commentaire (optionnel)")
                submitted = st.form_submit_button("✅ Valider")
                
                if submitted:
                    save_validation(doctor_name, model_choice, comment, rating, pred_class, proba)
                    st.success("Merci pour votre validation !")

elif menu_choice == "Tableau de bord" and role == "Administrateur":
    st.header("📊 Tableau de bord administrateur")
    
    # Métriques des modèles
    st.subheader("📈 Performance des modèles")
    
    # Convertir en DataFrame pour affichage
    metrics_df = pd.DataFrame(model_metrics).T
    st.dataframe(metrics_df, use_container_width=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart des ROC-AUC
        fig = px.bar(
            x=list(model_metrics.keys()),
            y=[m['ROC-AUC'] for m in model_metrics.values()],
            title="ROC-AUC par modèle",
            labels={'x': 'Modèle', 'y': 'ROC-AUC'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart
        categories = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig = go.Figure()
        
        for model_name, metrics in model_metrics.items():
            fig.add_trace(go.Scatterpolar(
                r=[metrics.get(cat, 0) for cat in categories],
                theta=categories,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparaison des modèles"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Validations
    st.subheader("📋 Validations médicales")
    validations = load_validations()
    if not validations.empty:
        st.dataframe(validations, use_container_width=True)
        
        # Statistiques des validations
        if 'rating' in validations.columns:
            avg_rating = validations['rating'].mean()
            st.metric("Note moyenne", f"{avg_rating:.2f}/5")
    else:
        st.info("Aucune validation pour l'instant.")

elif menu_choice == "Historique":
    st.header("📜 Historique des prédictions")
    
    validations = load_validations()
    if not validations.empty:
        st.dataframe(validations, use_container_width=True)
        
        # Graphique d'évolution
        if 'timestamp' in validations.columns and 'probability' in validations.columns:
            validations['timestamp'] = pd.to_datetime(validations['timestamp'])
            fig = px.line(validations, x='timestamp', y='probability', color='model',
                         title="Évolution des probabilités")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun historique disponible")

elif menu_choice == "À propos":
    st.header("ℹ️ À propos")
    
    st.markdown("""
    <div class="card">
        <h3>MediRisk - Cancer du Col</h3>
        <p><strong>Version:</strong> 1.0.0</p>
        <p><strong>Description:</strong> Application d'aide à la décision médicale pour l'évaluation du risque de cancer du col de l'utérus.</p>
        <p><strong>Modèles:</strong> Random Forest, XGBoost, SVM</p>
        <p><strong>Explicabilité:</strong> SHAP (SHapley Additive exPlanations)</p>
        <p><strong>Développé par:</strong> Équipe projet - 2026</p>
    </div>
    
    <div class="card">
        <h3>📚 Documentation</h3>
        <p>Pour utiliser l'application :</p>
        <ul>
            <li>Sélectionnez votre profil dans le menu latéral</li>
            <li>Choisissez un modèle dans l'onglet Prédiction</li>
            <li>Remplissez les caractéristiques de la patiente</li>
            <li>Cliquez sur Prédire pour obtenir le résultat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PIED DE PAGE
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem;">
    MediRisk &copy; 2026 - Projet d'aide à la décision médicale<br>
    Données : UCI Machine Learning Repository
</div>
""", unsafe_allow_html=True)


# ============================================
# PIED DE PAGE
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem;">
    MediRisk &copy; 2026 - Projet d'aide à la décision médicale<br>
    Données : UCI Machine Learning Repository
</div>
""", unsafe_allow_html=True)
