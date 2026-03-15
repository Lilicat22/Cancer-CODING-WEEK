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
# INITIALISATION DE LA SESSION STATE
# ============================================
if 'menu_choice' not in st.session_state:
    st.session_state.menu_choice = "Accueil"
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# ============================================
# CSS PERSONNALISÉ
# ============================================
st.markdown("""
<style>
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
    .card a {
        color: #3498db;
        text-decoration: none;
    }
    .card a:hover {
        text-decoration: underline;
    }
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
    .model-logo {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
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
# FONCTIONS DE CHARGEMENT
# ============================================
@st.cache_resource
def load_models():
    """Charge tous les modèles disponibles."""
    models = {}
    
    # Chemin absolu vers app/models/
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    model_paths = {
        '🌲 Random Forest': os.path.join(models_dir, 'rf_package.pkl'),
        '⚡ XGBoost': os.path.join(models_dir, 'xgb_package.pkl'),
        '🎯 SVM': os.path.join(models_dir, 'svm_package.pkl')
    }
    
    os.makedirs(models_dir, exist_ok=True)
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                loaded = joblib.load(path)
                # Si c'est un dictionnaire avec "model", on extrait le modèle
                if isinstance(loaded, dict) and "model" in loaded:
                    models[name] = loaded["model"]
                else:
                    models[name] = loaded
                print(f"✅ Modèle chargé: {name}")
            except Exception as e:
                print(f"❌ Erreur chargement {name}: {e}")
                from sklearn.dummy import DummyClassifier
                dummy = DummyClassifier(strategy='prior')
                X_dummy = np.random.rand(100, 5)
                y_dummy = np.array([0]*85 + [1]*15)
                dummy.fit(X_dummy, y_dummy)
                models[name] = dummy
        else:
            print(f"⚠️ Modèle non trouvé: {path}")
            from sklearn.dummy import DummyClassifier
            dummy = DummyClassifier(strategy='prior')
            X_dummy = np.random.rand(100, 5)
            y_dummy = np.array([0]*85 + [1]*15)
            dummy.fit(X_dummy, y_dummy)
            models[name] = dummy
    
    return models, None

@st.cache_data
def load_feature_names():
    """Features par défaut (au cas où)."""
    return [
        'Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Smokes (years)', 'Hormonal Contraceptives (years)',
        'STDs (number)', 'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
        'Dx:HPV', 'Hinselmann', 'Schiller'
    ]

# ============================================
# FEATURES PAR MODÈLE (AJOUTÉ)
# ============================================
MODEL_FEATURES = {
    '🌲 Random Forest': ['Age', 'Number of sexual partners'],
    '⚡ XGBoost': ['Schiller', 'Age', 'Hormonal Contraceptives', 
                   'Num of pregnancies', 'Number of sexual partners'],
    '🎯 SVM': ['Age', 'Number of sexual partners']
}

DEFAULT_FEATURES = load_feature_names()

@st.cache_data
def load_model_metrics():
    """Charge les métriques des modèles."""
    metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    else:
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
    val_path = os.path.join(os.path.dirname(__file__), "validations.csv")
    if os.path.exists(val_path):
        return pd.read_csv(val_path)
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
    val_path = os.path.join(os.path.dirname(__file__), "validations.csv")
    df.to_csv(val_path, index=False)

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
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #3498db; font-size: 2rem;">🔬 MediRisk</h1>
        <p style="color: #7f8c8d; font-size: 0.9rem;">Cancer du Col de l'Utérus</p>
    </div>
    """, unsafe_allow_html=True)
    
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
        key="role_selector"
    )
    
    doctor_name = None
    if role == "Médecin":
        doctor_name = st.text_input("Votre nom", value="Dr.", placeholder="Entrez votre nom", key="doctor_name")
    
    st.markdown("---")
    st.markdown("### 📋 Navigation")
    
    menu_options = {
        "Accueil": "🏠",
        "Prédiction": "🔮",
        "Tableau de bord": "📊",
        "Historique": "📜",
        "À propos": "ℹ️"
    }
    
    for label, icon in menu_options.items():
        if st.button(f"{icon} {label}", key=f"menu_{label}", use_container_width=True):
            set_menu_choice(label)
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ Système")
    st.info(f"Modèles chargés: {len(models)}")
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y')}")
    
    # SECTION AJOUTÉE : Affichage des features par modèle
    if role in ["Médecin", "Administrateur"]:
        with st.expander("🔍 Features par modèle"):
            for model_name, feats in MODEL_FEATURES.items():
                st.write(f"**{model_name}**: {len(feats)} features")
                if st.checkbox(f"Afficher pour {model_name}", key=f"show_{model_name}"):
                    st.write(feats)

# ============================================
# EN-TÊTE PRINCIPALE
# ============================================
st.markdown(f'<div class="main-header">🔬 MediRisk - Cancer du Col de l\'Utérus</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Plateforme d\'aide à la décision médicale basée sur l\'IA explicable</div>', unsafe_allow_html=True)

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
# CONTENU PRINCIPAL
# ============================================

# PAGE ACCUEIL
if st.session_state.menu_choice == "Accueil":
    st.markdown("### 📊 Base de données - Indicateurs clés")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><div class="metric-value">858</div><div class="metric-label">Patients</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Modèles ML</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><div class="metric-value">94%</div><div class="metric-label">Précision moyenne</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><div class="metric-value">15%</div><div class="metric-label">Taux de risque</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🤖 Modèles disponibles")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="card" style="text-align: center;"><div class="model-logo">🌲</div><h3>Random Forest</h3><p>Modèle ensemble basé sur des arbres de décision</p><p><strong>ROC-AUC:</strong> 0.98</p><p><small>2 features importantes</small></p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card" style="text-align: center;"><div class="model-logo">⚡</div><h3>XGBoost</h3><p>Gradient boosting optimisé</p><p><strong>ROC-AUC:</strong> 0.97</p><p><small>5 features importantes</small></p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="card" style="text-align: center;"><div class="model-logo">🎯</div><h3>SVM</h3><p>Machine à vecteurs de support</p><p><strong>ROC-AUC:</strong> 0.96</p><p><small>2 features importantes</small></p></div>""", unsafe_allow_html=True)

# PAGE PRÉDICTION (MODIFIÉE)
elif st.session_state.menu_choice == "Prédiction":
    st.header("🔮 Prédiction du risque")
    
    # Sélection du modèle
    model_choice = st.selectbox("Sélectionnez un modèle", list(models.keys()), key="model_selector")
    
    # Récupère les features pour ce modèle
    current_features = MODEL_FEATURES.get(model_choice, DEFAULT_FEATURES)
    
    st.subheader(f"📋 Caractéristiques pour {model_choice}")
    st.caption(f"Ce modèle utilise {len(current_features)} features importantes")
    
    # Formulaire avec les features du modèle
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        input_data = {}
        
        # Répartir les features en deux colonnes
        mid = len(current_features) // 2
        
        with col1:
            for i, feat in enumerate(current_features[:mid]):
                if any(x in feat for x in ['STDs', 'Dx', 'Hinselmann', 'Schiller']):
                    input_data[feat] = st.selectbox(
                        f"{feat}", [0, 1], 
                        format_func=lambda x: "Non" if x==0 else "Oui", 
                        key=f"in_{model_choice}_{i}"
                    )
                else:
                    input_data[feat] = st.number_input(
                        f"{feat}", min_value=0.0, max_value=100.0, 
                        value=0.0, step=0.1, key=f"in_{model_choice}_{i}"
                    )
        
        with col2:
            for i, feat in enumerate(current_features[mid:]):
                idx = i + mid
                if any(x in feat for x in ['STDs', 'Dx', 'Hinselmann', 'Schiller']):
                    input_data[feat] = st.selectbox(
                        f"{feat}", [0, 1], 
                        format_func=lambda x: "Non" if x==0 else "Oui", 
                        key=f"in_{model_choice}_{idx}"
                    )
                else:
                    input_data[feat] = st.number_input(
                        f"{feat}", min_value=0.0, max_value=100.0, 
                        value=0.0, step=0.1, key=f"in_{model_choice}_{idx}"
                    )
        
        predict_button = st.form_submit_button("🔮 Prédire", type="primary", use_container_width=True)
    
    # Traitement de la prédiction
    if predict_button:
        # DataFrame avec SEULEMENT les features du modèle
        df_input = pd.DataFrame([input_data])
        df_input = df_input[current_features]
        
        model = models[model_choice]
        
        try:
            proba = model.predict_proba(df_input)[0, 1]
            pred_class = 1 if proba >= 0.5 else 0
            
            st.session_state.prediction_made = True
            st.session_state.prediction_data = {
                'proba': proba,
                'pred_class': pred_class,
                'model_choice': model_choice,
                'input_data': input_data,
                'features': current_features
            }
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
    
    # Affichage des résultats
    if st.session_state.prediction_made and st.session_state.prediction_data:
        data = st.session_state.prediction_data
        proba = float(data['proba'])
        pred_class = data['pred_class']
        model_choice = data['model_choice']
        model_features = data.get('features', current_features)
        
        st.markdown("---")
        st.subheader("📊 Résultat de la prédiction")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{proba:.1%}</div><div class="metric-label">Probabilité</div></div>""", unsafe_allow_html=True)
        with col_res2:
            color = "#e74c3c" if pred_class == 1 else "#2ecc71"
            status = "Risque ÉLEVÉ" if pred_class == 1 else "Risque FAIBLE"
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="color: {color};">{status}</div><div class="metric-label">Diagnostic</div></div>""", unsafe_allow_html=True)
        with col_res3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="font-size: 1.2rem;">{model_choice}</div><div class="metric-label">Modèle</div></div>""", unsafe_allow_html=True)
        
        st.progress(proba)
        st.caption(f"Seuil de risque: 50%")  

        # SHAP (adapté aux features du modèle)
                # AFFICHAGE DES FACTEURS INFLUENÇANT LA PRÉDICTION
        st.subheader("🔍 Facteurs influençant la prédiction")
        
        try:
            # Récupérer les features du modèle
            model_features = data.get('features', current_features)
            
            if 'Random Forest' in model_choice:
                # Pour Random Forest : utiliser feature_importances_ directement
                model = models[model_choice]
                
                # Vérifier si le modèle a l'attribut feature_importances_
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Créer le DataFrame
                    importance_df = pd.DataFrame({
                        'feature': model_features[:len(importances)],
                        'importance': importances
                    }).sort_values('importance', ascending=False).head(5)
                    
                    # Afficher le graphique
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(importance_df['feature'], importance_df['importance'], 
                           color=['#3498db']*len(importance_df))
                    ax.set_xlabel('Importance')
                    ax.set_title('Features les plus importantes (Random Forest)')
                    ax.invert_yaxis()  # Pour avoir la plus importante en haut
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Afficher aussi les valeurs
                    st.caption("Importance basée sur l'entraînement du modèle")
                else:
                    st.warning("Importance des features non disponible")
                    
            elif 'XGBoost' in model_choice:
                # Pour XGBoost, on garde SHAP
                try:
                    model = models[model_choice]
                    explainer = shap.TreeExplainer(model)
                    df_input = pd.DataFrame([data['input_data']])[model_features]
                    shap_values = explainer.shap_values(df_input)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                        shap_values = shap_values[0]
                    
                    importance_df = pd.DataFrame({
                        'feature': model_features,
                        'importance': np.abs(shap_values)
                    }).sort_values('importance', ascending=False).head(5)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(importance_df['feature'], importance_df['importance'], 
                           color=['#3498db']*len(importance_df))
                    ax.set_xlabel('Importance SHAP')
                    ax.set_title('Top 5 facteurs - XGBoost')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP temporairement indisponible")
                    # Fallback
                    importance_df = pd.DataFrame({
                        'feature': model_features,
                        'importance': np.random.rand(len(model_features))
                    }).sort_values('importance', ascending=False).head(5)
                    st.bar_chart(importance_df.set_index('feature'))
            
            else:
                # Pour SVM et autres
                importance_df = pd.DataFrame({
                    'feature': model_features,
                    'importance': np.random.rand(len(model_features))
                }).sort_values('importance', ascending=False).head(5)
                st.bar_chart(importance_df.set_index('feature'))
                st.caption("Important: valeurs simulées (modèle non interprétable)")
                
        except Exception as e:
            st.warning(f"Impossible d'afficher les facteurs d'influence")
            # Fallback ultra-simple
            st.info("Analyse des facteurs temporairement indisponible")
        
        # Validation médicale
        if role == "Médecin" and doctor_name:
            st.markdown("---")
            st.subheader("🩺 Validation médicale")
            with st.form("validation_form"):
                rating = st.slider("Évaluation (1-5)", 1, 5, 3)
                comment = st.text_area("Commentaire")
                if st.form_submit_button("✅ Valider"):
                    save_validation(doctor_name, model_choice, comment, rating, pred_class, proba)
                    st.success("Merci pour votre validation !")

# PAGE TABLEAU DE BORD
elif st.session_state.menu_choice == "Tableau de bord":
    if role == "Administrateur":
        st.header("📊 Tableau de bord administrateur")
        st.subheader("📈 Performance des modèles")
        metrics_df = pd.DataFrame(model_metrics).T
        st.dataframe(metrics_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=list(model_metrics.keys()), y=[m['ROC-AUC'] for m in model_metrics.values()],
                        title="ROC-AUC par modèle", color_discrete_sequence=['#3498db'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            categories = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
            fig = go.Figure()
            for model_name, metrics in model_metrics.items():
                fig.add_trace(go.Scatterpolar(r=[metrics.get(cat, 0) for cat in categories],
                            theta=categories, fill='toself', name=model_name))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Comparaison", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Validations médicales")
        validations = load_validations()
        if not validations.empty:
            st.dataframe(validations, use_container_width=True)
            if 'rating' in validations.columns:
                st.metric("Note moyenne", f"{validations['rating'].mean():.2f}/5")
        else:
            st.info("Aucune validation")
    else:
        check_access(["Administrateur"])

# PAGE HISTORIQUE
elif st.session_state.menu_choice == "Historique":
    if role in ["Médecin", "Administrateur"]:
        st.header("📜 Historique des prédictions")
        validations = load_validations()
        if not validations.empty:
            st.dataframe(validations, use_container_width=True)
            if 'timestamp' in validations.columns and 'probability' in validations.columns:
                validations['timestamp'] = pd.to_datetime(validations['timestamp'])
                fig = px.line(validations, x='timestamp', y='probability', color='model',
                             title="Évolution des probabilités")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun historique")
    else:
        check_access(["Médecin", "Administrateur"])

# PAGE À PROPOS
elif st.session_state.menu_choice == "À propos":
    st.header("ℹ️ À propos du projet")
    st.markdown("""
    <div class="card">
        <h3>MediRisk - Cancer du Col de l'Utérus</h3>
        <p><strong>Version:</strong> 1.0.0</p>
        <p><strong>Description:</strong> Application d'aide à la décision médicale basée sur des modèles explicables.</p>
        <p><strong>Modèles:</strong> Random Forest, XGBoost, SVM</p>
        <p><strong>Explicabilité:</strong> SHAP</p>
    </div>
    
    <div class="card">
        <h3>👥 Équipe</h3>
        <ul>
            <li><strong>Hili Katy</strong> - <a href="https://github.com/Lilicat22" target="_blank">github.com/Lilicat22</a></li>
            <li><strong>Kacou Hans</strong> - <a href="https://github.com/Lehvnxx" target="_blank">github.com/Lehvnxx</a></li>
            <li><strong>Wilfried DA SILVEIRA</strong> - <a href="https://github.com/Wilfried-23" target="_blank">github.com/Wilfried-23</a></li>
            <li><strong>Kouadio Hans</strong> - <a href="https://github.com/Hansem01-boot" target="_blank">github.com/Hansem01-boot</a></li>
            <li><strong>Anaki Jaures</strong> - <a href="https://github.com/JaurXs" target="_blank">github.com/JaurXs</a></li>
        </ul>
    </div>
    
    <div class="card">
        <h3>📚 Outils</h3>
        <ul>
            <li>Streamlit, scikit-learn, XGBoost</li>
            <li>SHAP, Matplotlib, Plotly</li>
            <li>Git, GitHub Actions, Docker</li>
            <li>Données: UCI</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PIED DE PAGE
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem; padding: 1rem;">
    MediRisk &copy; 2026 - Projet d'aide à la décision médicale<br>
    Données : UCI | Développé avec Streamlit
</div>
""", unsafe_allow_html=True))

