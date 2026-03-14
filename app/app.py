import streamlit as st

st.set_page_config(page_title="Cervical cancer Predaction",layout="wide")

# --------------- USER PROFILE -----------------#

if "role" not in st.session_state:
    st.title("Cervical Cancer Predact Platform")

    role = st.selectbox(
        "Select your profile",
        [
            "Visitor",
            "Doctor(Male)",
            "Administrator"
        ]
    )
    if st.button("Entrer platform"):
        st.session_state.role= role
        st.rerun()
    
    st.stop()

role= st.session_state.role

menu=["Home","Patient Data","Modele Testing"]

if "Doctor" in role:
    menu+= ["Medical Validation","Validated Models"]

if role == "Administrator":
    menu+= ["Admin Dashboard"]

page= st.sidebar.radio("Navigation", menu)


model_choice= st.selectbox(
    "Select a Machine Learning Model",
    [
        "XGBoost",
        "SVM",
        "Random Forest",
        "Losgistic Regression"
    ]
)

if st.button("Run Prediction"):

    if model_choice== "XGBoost":
        model= xgb_model

    elif model_choice== "SVM":
        model= svm_model

    elif model_choice== "Random Forest":
        model= rf_model

    prediction= model.predict("patient_data")

if "Doctor" in role:

    st.subheader("Model Validation")

    model_to_validate= st.selectbox(
        "Select model to validate",
        ["XGBoost","SVM","Random Forest"]
    )

    comment= st.text_area(
        "Medical comment about this model"
    )

    if st.button("Validate model"):

        validation={
            "doctor": role,
            "model": model_to_validate,
            "comment": comment
        }

        st.success("Model validated sucessfully")

