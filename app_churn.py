import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Image pour illustrer
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://webcdn.ringover.com/img/big/anti-churn-ba1ec.png", width=500)

# Charger le modèle et le scaler
model = joblib.load('model/churn_model.pkl')
scaler = joblib.load('model/churn_scaler.pkl')

# SHAP explainer
explainer = shap.Explainer(model)

st.sidebar.title("**Menu**")
menu = st.sidebar.radio(" 👇 Choisissez une option", ["Prédiction individuelle", "Prédiction par lot"])

if menu == "Prédiction individuelle":
    
    st.title("Prédiction de l'attriction d'un client 🏃‍♂️ 🏃‍♂️ 🏃‍♂️ 🏃‍♂️ 🏃‍♂️CHURN🏃 🏃‍♂️ 🏃‍♂️ 🏃‍♂️")
    st.header("Entrez les caractéristiques du client")

# Champs de saisie pour les données de l'utilisateur
    gender = st.selectbox("Gender", options=["Female","Male"])
    SeniorCitizen = st.selectbox("Senior citizen", options=[0,1])
    Partner = st.selectbox("Partner", options=["Yes", "No"])
    Dependents = st.selectbox("Dependents", options=['Yes', 'No'])
    tenure = st.number_input("Tenure", min_value=0, step=1)
    PhoneService = st.selectbox("Phone service", options=['Yes', 'No'])
    MultipleLines = st.selectbox("Multiple Lines", options=['Yes', 'No', 'No phone service'])
    InternetService = st.selectbox("Internet services", options=["Fiber optic", "DSL", "No"])
    OnlineSecurity = st.selectbox("Online security", options=["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online backup", options=["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device protection", options=["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech support", options=["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", options=["Month-to-month", "Two year", "One year"])
    PaperlessBilling = st.selectbox("Paperless belling", options=["Yes", "No"])
    PaymentMethod = st.selectbox("Payment method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly charges", min_value=0.00, step=0.01)
    TotalCharges = st.number_input("Total charges", min_value=0.00, step=0.01)

    # Convertir les entrées catégorielles en numériques

    gender_Male = 1 if gender == "Male" else 0
    Partner_Yes = 1 if Partner == "Yes" else 0
    Dependents_Yes = 1 if Dependents == "Yes" else 0
    PhoneService_Yes = 1 if PhoneService == "Yes" else 0
    MultipleLines_Nophoneservice = 1 if MultipleLines == "No phone service" else 0
    MultipleLines_Yes = 1 if MultipleLines == "Yes" else 0
    InternetService_Fiberoptic = 1 if InternetService == "Fiber optic" else 0
    InternetService_No = 1 if InternetService == "No" else 0
    OnlineSecurity_Nointernetservice = 1 if OnlineSecurity == "No internet service" else 0
    OnlineSecurity_Yes = 1 if OnlineSecurity == "Yes" else 0
    OnlineBackup_Nointernetservice = 1 if OnlineBackup == "No internet service" else 0
    OnlineBackup_Yes = 1 if OnlineBackup == "Yes" else 0
    DeviceProtection_Nointernetservice = 1 if DeviceProtection == "No internet service" else 0
    DeviceProtection_Yes = 1 if DeviceProtection == "Yes" else 0
    TechSupport_Nointernetservice = 1 if TechSupport == "No internet service" else 0
    TechSupport_Yes = 1 if TechSupport == "Yes" else 0
    StreamingTV_Nointernetservice = 1 if StreamingTV == "No internet service" else 0
    StreamingTV_Yes = 1 if StreamingTV == "Yes" else 0
    StreamingMovies_Nointernetservice = 1 if StreamingMovies == "No internet service" else 0
    StreamingMovies_Yes = 1 if StreamingMovies == "Yes" else 0
    Contract_Oneyear = 1 if Contract == "One year" else 0
    Contract_Twoyear = 1 if Contract == "Two year" else 0
    PaperlessBilling_Yes = 1 if PaperlessBilling == "Yes" else 0
    PaymentMethod_Creditcard = 1 if PaymentMethod == "Credit card (automatic)" else 0
    PaymentMethod_Electroniccheck = 1 if PaymentMethod == "Electronic check" else 0
    PaymentMethod_Mailedcheck = 1 if PaymentMethod == "Mailed check" else 0

    # Créer un DataFrame avec les données
    input_data = pd.DataFrame({
        'SeniorCitizen' : [SeniorCitizen], 
        'tenure' : [tenure], 
        'MonthlyCharges' : [MonthlyCharges], 
        'TotalCharges' : [TotalCharges],
        'gender_Male' : [gender_Male], 
        'Partner_Yes' : [Partner_Yes], 
        'Dependents_Yes' : [Dependents_Yes], 
        'PhoneService_Yes' : [PhoneService_Yes],
        'MultipleLines_No phone service' : [MultipleLines_Nophoneservice], 
        'MultipleLines_Yes' : [MultipleLines_Yes],
        'InternetService_Fiber optic' : [InternetService_Fiberoptic], 
        'InternetService_No' : [InternetService_No],
       'OnlineSecurity_No internet service' : [OnlineSecurity_Nointernetservice], 
       'OnlineSecurity_Yes' : [OnlineSecurity_Yes],
       'OnlineBackup_No internet service' : [OnlineBackup_Nointernetservice], 
       'OnlineBackup_Yes' : [OnlineBackup_Yes],
       'DeviceProtection_No internet service' : [DeviceProtection_Nointernetservice], 
       'DeviceProtection_Yes' : [DeviceProtection_Yes],
       'TechSupport_No internet service' : [TechSupport_Nointernetservice], 
       'TechSupport_Yes' : [TechSupport_Yes],
       'StreamingTV_No internet service' : [StreamingTV_Nointernetservice], 
       'StreamingTV_Yes' : [StreamingTV_Yes],
       'StreamingMovies_No internet service' : [StreamingMovies_Nointernetservice], 
       'StreamingMovies_Yes' : [StreamingMovies_Yes],
       'Contract_One year' : [Contract_Oneyear], 
       'Contract_Two year' : [Contract_Twoyear], 
       'PaperlessBilling_Yes' : [PaperlessBilling_Yes],
       'PaymentMethod_Credit card (automatic)' : [PaymentMethod_Creditcard],
       'PaymentMethod_Electronic check' : [PaymentMethod_Electroniccheck], 
       'PaymentMethod_Mailed check' : [PaymentMethod_Mailedcheck]
    })
    # Aligner les colonnes pour correspondre exactement à celles du modèle
    expected_columns = model.feature_names_in_  # Colonne d'entrée utilisée pour l'entraînement
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)
    
    # Appliquer le scaler
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges']])
    st.dataframe(input_data)

    # Prédiction

    if st.button("🔍 Prédire le churn"):
        prediction = model.predict(input_data)[0] 

        if prediction == 1:
            st.markdown("### 🛑 **Le client est à risque de churn.**")
            st.error("Prenez des mesures de fidélisation.")
        if prediction == 0 :
            st.markdown("### ✅ **Le client est probablement fidèle.**")
            st.success("Pas de risque immédiat détecté.")
        
        import streamlit.components.v1 as components
        import shap

        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)

        # Explication avec force_plot (JS)
        shap_values = explainer(input_data)
        force_plot = shap.force_plot(explainer.expected_value[0], shap_values.values[0], input_data)

        st.subheader("📊 Explication SHAP interactive")
        st_shap(force_plot, height=300)

elif menu == "Prédiction par lot": 
    st.title("Prédiction de l'attriction des clients 🏃‍♂️ 🏃‍♂️ 🏃‍♂️ 🏃‍♂️ 🏃‍♂️CHURN🏃 🏃‍♂️ 🏃‍♂️ 🏃‍♂️")
    st.markdown(" ")
    st.subheader("📁 Importer un fichier avec les données du client")
    uploaded_file = st.file_uploader(" ", type=["csv"])

    if uploaded_file is not None :
        df = pd.read_csv(uploaded_file, index_col='customerID')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

        st.success("✅ Données brutes importées :")
        st.dataframe(df)

        df_original = df.copy()

        # Encodage des variables catégorielles

        df['gender_Male'] = df['gender'].apply(lambda x : 1 if x=='Male' else 0)
        df['Partner_Yes'] = df['Partner'].apply(lambda x: 1 if x=='Yes' else 0)
        df['Dependents_Yes'] = df['Dependents'].apply(lambda x: 1 if x=='Yes' else 0)
        df['PhoneService_Yes'] = df['PhoneService'].apply(lambda x: 1 if x=="Yes" else 0)
        df['MultipleLines_No phone service'] = df['MultipleLines'].apply(lambda x: 1 if x=="No phone service" else 0)
        df['MultipleLines_Yes'] = df['MultipleLines'].apply(lambda x: 1 if x=="Yes" else 0)
        df['InternetService_Fiber optic'] = df['InternetService'].apply(lambda x: 1 if x=='Fiber optic' else 0)
        df['InternetService_No'] = df['InternetService'].apply(lambda x: 1 if x=='No' else 0)
        df['OnlineSecurity_No internet service'] = df['OnlineSecurity'].apply(lambda x: 1 if x=="No internet service" else 0)
        df["OnlineSecurity_Yes"] = df["OnlineSecurity"].apply(lambda x: 1 if x=="Yes" else 0)
        df["OnlineBackup_No internet service"] = df["OnlineBackup"].apply(lambda x: 1 if x=="No internet service" else 0)
        df["OnlineBackup_Yes"] = df['OnlineBackup'].apply(lambda x: 1 if x=="Yes" else 0)
        df['DeviceProtection_No internet service'] = df['DeviceProtection'].apply(lambda x: 1 if x == 'No internet service' else 0)
        df['DeviceProtection_Yes'] = df['DeviceProtection'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['TechSupport_No internet service'] = df['TechSupport'].apply(lambda x: 1 if x == 'No internet service' else 0)
        df['TechSupport_Yes'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['StreamingTV_No internet service'] = df['StreamingTV'].apply(lambda x: 1 if x == 'No internet service' else 0)
        df['StreamingTV_Yes'] = df['StreamingTV'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['StreamingMovies_No internet service'] = df['StreamingMovies'].apply(lambda x: 1 if x == 'No internet service' else 0)
        df['StreamingMovies_Yes'] = df['StreamingMovies'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['Contract_One year'] = df['Contract'].apply(lambda x: 1 if x == 'One year' else 0)
        df['Contract_Two year'] = df['Contract'].apply(lambda x: 1 if x == 'Two year' else 0)
        df['PaperlessBilling_Yes'] = df['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
        df['PaymentMethod_Credit card (automatic)'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Credit card (automatic)' else 0)
        df['PaymentMethod_Electronic check'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)
        df['PaymentMethod_Mailed check'] = df['PaymentMethod'].apply(lambda x: 1 if x == 'Mailed check' else 0)

        # Créer le DataFrame final avec les colonnes attendues

        input_data = df.copy()
        expected_columns = model.feature_names_in_
        input_data = input_data[expected_columns]

        # Appliquer le scaler

        input_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_data[['tenure', 'MonthlyCharges', 'TotalCharges']])

        # Prédiction

        prediction = model.predict(input_data)

        df['Churn'] = prediction
        df['Resultat'] = df['Churn'].apply(lambda x: "Churn" if x == 1 else "Fidele")

        # Réinitialiser l'index pour remettre customerID en colonne
        df_reset = df.reset_index()

        df_resultat = df_reset[["customerID", "Resultat"]]

        st.subheader("🔍 Résultats de la prédiction :")
        st.dataframe(df_resultat) 

        # Préparer le CSV pour téléchargement

        csv = df_resultat.to_csv(index=False).encode("utf-8")

        # Bouton pour télécharger les résultats

        st.download_button(label="📥 Télécharger les résultats", data=csv, file_name="resultats_prediction_churn.csv", mime="text/csv")
