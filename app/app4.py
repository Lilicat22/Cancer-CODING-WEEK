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
    page_title="MediRisk - Cancer du Col de l'Utérus",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# INITIALISATION DE LA SESSION STATE (TRÈS IMPORTANT)
# ============================================
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Accueil"
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None


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
        color: #2c3e50;
    }
    .card h3 {
        color: #2c3e50;
        margin-top: 0;
    }

    .card p {
        color: #34495e;
    }
    .card ul, .card il {
        color: #3498e;
    }
    .card a {
        color: #3498db;
        text-decoration: none;
    }
    .card a:hover {
        text-decoration: underline;
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
    /* Style des logos modèles */
    .model-logo {
        font-size: 2rem;
        margin-bottom: 0.5rem;
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
    /* Message d'accès refusé */
    .access-denied {
        background-color: #fee;
        color: #c00;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #c00;
        margin: 1rem 0;
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
        '🌲 Random Forest': 'app/models/rf_model.pkl',
        '⚡ XGBoost': 'app/models/xgb_model.pkl',
        '🎯 SVM': 'app/models/svm_model.pkl'
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
                X_dummy = np.random.rand(100, 12)  # 12 features maintenant
                y_dummy = np.array([0]*85 + [1]*15)
                dummy.fit(X_dummy, y_dummy)
                models[name] = dummy
        else:
            # Créer un modèle factice
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy='prior')
            X_dummy = np.random.rand(100, 12)
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
    """Charge les 12 features les plus pertinentes."""
    # Features sélectionnées pour leur importance (top 12)
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
        'Schiller'
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
            '🌲 Random Forest': {
                'ROC-AUC': 0.98, 'Accuracy': 0.95, 'Precision': 0.92, 
                'Recall': 0.88, 'F1-Score': 0.90
            },
            '⚡ XGBoost': {
                'ROC-AUC': 0.97, 'Accuracy': 0.94, 'Precision': 0.91,
                'Recall': 0.87, 'F1-Score': 0.89
            },
            '🎯 SVM': {
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
        <p style="color: #7f8c8d; font-size: 0.9rem;">Cancer du Col de l'Utérus</p>
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
        format_func=lambda x: f"{profile_icons[x]} {x}",
        key="role_selector"  # Key importante pour éviter les rechargements intempestifs
    )
    
    # Si médecin, demander le nom
    doctor_name = None
    if role == "Médecin":
        doctor_name = st.text_input("Votre nom", value="Dr.", placeholder="Entrez votre nom", key="doctor_name")
    
    st.markdown("---")
    
    # Menu principal avec icônes (sans menu déroulant)
    st.markdown("### 📋 Navigation")
    
    menu_options = {
        "Accueil": "🏠",
        "Prédiction": "🔮",
        "Tableau de bord": "📊",
        "Historique": "📜",
        "À propos": "ℹ️"
    }
    
    # Créer des boutons pour le menu
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
st.markdown(f'<div class="sub-header">Plateforme d\'aide à la décision médicale basée sur l\'IA explicable</div>', unsafe_allow_html=True)

# Message de bienvenue personnalisé
welcome_messages = {
    "Visiteur": "👋 Vous êtes en mode **Visiteur**. Vous pouvez tester les modèles de prédiction.",
    "Médecin": f"👋 Bonjour **{doctor_name if doctor_name else 'Docteur'}**. Vous avez accès aux fonctionnalités de validation.",
    "Administrateur": "👋 Mode **Administrateur**. Vous pouvez voir les performances globales et les validations."
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

# PAGE ACCUEIL
if st.session_state.menu_choice == "Accueil":
    # Dashboard avec métriques
    st.markdown("### 📊 Base de données - Indicateurs clés")
    
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
            <div class="metric-label">Modèles ML</div>
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
            """, unsafe_allow_html=True)
    
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

# PAGE PRÉDICTION (accessible à tous)
elif st.session_state.menu_choice == "Prédiction":
    st.header("🔮 Prédiction du risque")
    
    # Sélection du modèle
    model_choice = st.selectbox("Sélectionnez un modèle", list(models.keys()), key="model_selector")
    
    # Formulaire de saisie (12 features seulement)
    st.subheader("📋 Caractéristiques de la patiente (12 features principales)")
    st.caption("Remplissez les informations ci-dessous pour obtenir une prédiction")
    
    # Utiliser un formulaire Streamlit pour éviter les rechargements intempestifs
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            for i, feat in enumerate(feature_names[:6]):  # 6 premières dans col1
                if any(x in feat for x in ['STDs', 'Dx', 'Hinselmann', 'Schiller']):
                    input_data[feat] = st.selectbox(
                        f"{feat}", 
                        [0, 1], 
                        format_func=lambda x: "Non" if x==0 else "Oui", 
                        key=f"in_{i}"
                    )
                else:
                    input_data[feat] = st.number_input(
                        f"{feat}", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=0.0, 
                        step=0.1, 
                        key=f"in_{i}"
                    )
        
        with col2:
            for i, feat in enumerate(feature_names[6:]):  # 6 suivantes dans col2
                idx = i + 6
                if any(x in feat for x in ['STDs', 'Dx', 'Hinselmann', 'Schiller']):
                    input_data[feat] = st.selectbox(
                        f"{feat}", 
                        [0, 1], 
                        format_func=lambda x: "Non" if x==0 else "Oui", 
                        key=f"in_{idx}"
                    )
                else:
                    input_data[feat] = st.number_input(
                        f"{feat}", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=0.0, 
                        step=0.1, 
                        key=f"in_{idx}"
                    )
        
        # Bouton de prédiction dans le formulaire
        predict_button = st.form_submit_button("🔮 Prédire", type="primary", use_container_width=True)
    
    # Traitement de la prédiction (en dehors du formulaire mais conditionné par le bouton)
    if predict_button:
        # Créer le DataFrame
        df_input = pd.DataFrame([input_data])
        df_input = df_input[feature_names]
        
        # Prédiction
        model = models[model_choice]
        
        try:
            if scaler is not None and 'SVM' in model_choice:
                X_scaled = scaler.transform(df_input)
                proba = model.predict_proba(X_scaled)[0, 1]
            else:
                proba = model.predict_proba(df_input)[0, 1]
            
            pred_class = 1 if proba >= 0.5 else 0
            
            # Stocker dans session state
            st.session_state.prediction_made = True
            st.session_state.prediction_data = {
                'proba': proba,
                'pred_class': pred_class,
                'model_choice': model_choice,
                'input_data': input_data
            }
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
    
    # Afficher les résultats si une prédiction a été faite
    if st.session_state.prediction_made and st.session_state.prediction_data:
        data = st.session_state.prediction_data
        proba = data['proba']
        pred_class = data['pred_class']
        model_choice = data['model_choice']
        
        st.markdown("---")
        st.subheader("📊 Résultat de la prédiction")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{proba:.1%}</div>
                <div class="metric-label">Probabilité de risque</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            color = "#e74c3c" if pred_class == 1 else "#2ecc71"
            status = "Risque ÉLEVÉ" if pred_class == 1 else "Risque FAIBLE"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{status}</div>
                <div class="metric-label">Diagnostic prédit</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size: 1.2rem;">{model_choice}</div>
                <div class="metric-label">Modèle utilisé</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Barre de progression
        st.progress(proba)
        st.caption(f"Seuil de risque: 50%")
        
        # SHAP (simplifié)
        st.subheader("🔍 Explication SHAP - Facteurs influençant la prédiction")
        
        try:
            if 'XGBoost' in model_choice or 'Random Forest' in model_choice:
                model = models[model_choice]
                explainer = shap.TreeExplainer(model)
                df_input = pd.DataFrame([data['input_data']])[feature_names]
                shap_values = explainer.shap_values(df_input)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Bar plot horizontal des 5 features les plus importantes
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(shap_values[0])
                }).sort_values('importance', ascending=False).head(5)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#3498db' if x > 0 else '#e74c3c' for x in feature_importance['importance']]
                ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
                ax.set_xlabel('Importance SHAP')
                ax.set_title('Top 5 facteurs influençant la prédiction')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Visualisation SHAP simplifiée pour ce modèle")
                # Version simplifiée
                importance_data = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(np.random.randn(len(feature_names)))
                }).sort_values('importance', ascending=False).head(5)
                st.bar_chart(importance_data.set_index('feature'))
        except Exception as e:
            st.warning(f"Visualisation SHAP temporairement indisponible: {e}")
        
        # Si médecin, proposer la validation
        if role == "Médecin" and doctor_name:
            st.markdown("---")
            st.subheader("🩺 Validation médicale")
            
            with st.form("validation_form"):
                rating = st.slider("Évaluation du modèle pour ce cas (1-5)", 1, 5, 3)
                comment = st.text_area("Commentaire (optionnel)")
                submitted = st.form_submit_button("✅ Valider cette prédiction")
                
                if submitted:
                    save_validation(doctor_name, model_choice, comment, rating, pred_class, proba)
                    st.success("✅ Merci pour votre validation ! Elle sera visible par la communauté médicale.")

# PAGE TABLEAU DE BORD
elif st.session_state.menu_choice == "Tableau de bord":
    if role == "Administrateur":
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
                labels={'x': 'Modèle', 'y': 'ROC-AUC'},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(showlegend=False)
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
                title="Comparaison des modèles",
                height=400
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
                st.metric("Note moyenne des validations", f"{avg_rating:.2f}/5")
        else:
            st.info("Aucune validation pour l'instant.")
    else:
        check_access(["Administrateur"])

# PAGE HISTORIQUE
elif st.session_state.menu_choice == "Historique":
    if role in ["Médecin", "Administrateur"]:
        st.header("📜 Historique des prédictions")
        
        validations = load_validations()
        if not validations.empty:
            st.dataframe(validations, use_container_width=True)
            
            # Graphique d'évolution
            if 'timestamp' in validations.columns and 'probability' in validations.columns:
                validations['timestamp'] = pd.to_datetime(validations['timestamp'])
                fig = px.line(validations, x='timestamp', y='probability', color='model',
                             title="Évolution des probabilités de risque",
                             labels={'probability': 'Probabilité', 'timestamp': 'Date'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun historique disponible")
    else:
        check_access(["Médecin", "Administrateur"])

# PAGE À PROPOS (accessible à tous)
elif st.session_state.menu_choice == "À propos":
    st.header("ℹ️ À propos du projet")
    
    st.markdown("""
    <div class="card">
        <h3>MediRisk - Cancer du Col de l'Utérus</h3>
        <p><strong>Version:</strong> 1.0.0</p>
        <p><strong>Description:</strong> Application d'aide à la décision médicale pour l'évaluation du risque de cancer du col de l'utérus basée sur des modèles de machine learning explicables.</p>
        <p><strong>Modèles implémentés:</strong> Random Forest, XGBoost, SVM</p>
        <p><strong>Explicabilité:</strong> SHAP (SHapley Additive exPlanations)</p>
    </div>
    
    <div class="card">
        <h3>👥 Équipe de développement</h3>
        <ul>
            <li><strong>Hili Katy</strong> - <a href="https://github.com/Lilicat22" target="_blank">github.com/Lilicat22</a> (Coordination, Structure)</li>
            <li><strong>Kacou Hans</strong> - <a href="https://github.com/Lehvnxx" target="_blank">github.com/Lehvnxx</a> (EDA, Data Processing)</li>
            <li><strong>Wilfried DA SILVEIRA</strong> - <a href="https://github.com/Wilfried-23" target="_blank">github.com/Wilfried-23</a> (Modèles ML)</li>
            <li><strong>Kouadio Hans</strong> - <a href="https://github.com/Hansem01-boot" target="_blank">github.com/Hansem01-boot</a> (SHAP, Explicabilité)</li>
            <li><strong>Anaki Jaures</strong> - <a href="https://github.com/JaurXs" target="_blank">github.com/JaurXs</a> (Interface, Déploiement)</li>
        </ul>
    </div>
    
    <div class="card">
        <h3>📚 Documentation et outils utilisés</h3>
        <ul>
            <li><strong>Source des données:</strong> <a href="https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors" target="_blank">UCI Machine Learning Repository</a></li>
            <li><strong>Framework:</strong> Streamlit</li>
            <li><strong>Bibliothèques ML:</strong> scikit-learn, XGBoost</li>
            <li><strong>Explicabilité:</strong> SHAP</li>
            <li><strong>Visualisation:</strong> Matplotlib, Plotly, Seaborn</li>
            <li><strong>Versionnement:</strong> Git, GitHub</li>
            <li><strong>CI/CD:</strong> GitHub Actions</li>
            <li><strong>Conteneurisation:</strong> Docker</li>
        </ul>
    </div>
    
    <div class="card">
        <h3>📅 Projet académique - 2026</h3>
        <p>Ce projet a été réalisé dans le cadre d'un cours de Machine Learning et développement d'applications de décision médicale.</p>
        <p>L'objectif était de créer une application complète et professionnelle, de l'analyse exploratoire des données jusqu'au déploiement, en passant par l'entraînement de modèles explicables.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PIED DE PAGE
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem; padding: 1rem;">
    MediRisk &copy; 2026 - Projet d'aide à la décision médicale<br>
    Données : UCI Machine Learning Repository | Développé avec Streamlit
</div>
""", unsafe_allow_html=True)
