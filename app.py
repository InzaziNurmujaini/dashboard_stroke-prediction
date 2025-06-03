import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np

# Load model dan preprocessing
rf_model = joblib.load("./models/rf_model.pkl")
logreg_model = joblib.load("./models/logreg_model.pkl")
selector = joblib.load("./models/chi_selector.pkl")
encoders = joblib.load("./models/label_encoders.pkl")

st.title("Dashboard Prediksi Risiko Stroke")
st.write("Masukkan informasi pasien untuk memprediksi risiko terkena stroke.")

# Pilih model
model_choice = st.selectbox("Pilih Model Prediksi", ["Random Forest", "Logistic Regression"])

st.subheader("Input Data Pasien")

# Input pengguna
gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
age = st.slider("Usia", 1, 100, 30)
hypertension = st.radio("Hipertensi?", ["Tidak", "Ya"])
heart_disease = st.radio("Penyakit Jantung?", ["Tidak", "Ya"])
ever_married = st.selectbox("Status Pernikahan", ["Yes", "No"])
work_type = st.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Kadar Glukosa Rata-rata", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
smoking_status = st.selectbox("Status Merokok", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Prediksi saat tombol ditekan
if st.button("Prediksi Stroke"):

    # Dataframe input
    input_data = {
        "gender": [gender],
        "age": [age],
        "hypertension": [1 if hypertension == "Ya" else 0],
        "heart_disease": [1 if heart_disease == "Ya" else 0],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [Residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status],
    }

    input_df = pd.DataFrame(input_data)

    # Encode kategorikal
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Seleksi fitur
    X_selected = selector.transform(input_df)

    # Prediksi
    if model_choice == "Random Forest":
        prediction = rf_model.predict(X_selected)[0]
        prob = rf_model.predict_proba(X_selected)[0][1]
    else:
        prediction = logreg_model.predict(X_selected)[0]
        prob = logreg_model.predict_proba(X_selected)[0][1]

        # Visualisasi feature importance (koefisien) untuk Logistic Regression
        st.subheader("Visualisasi Feature Importance")

        # Ambil nama fitur setelah seleksi
        try:
            feature_names = selector.get_feature_names_out()
        except:
            feature_names = input_df.columns[selector.get_support()]

        # Hitung pentingnya fitur berdasarkan nilai absolut dari koefisien
        importance = np.abs(logreg_model.coef_[0])

        coef_df = pd.DataFrame({
            "Fitur": feature_names,
            "Koefisien (Absolut)": importance
        }).sort_values(by="Koefisien (Absolut)", ascending=True)

        # Plot bar chart horizontal
        fig_logreg, ax = plt.subplots()
        ax.barh(coef_df["Fitur"], coef_df["Koefisien (Absolut)"], color="salmon")
        ax.set_xlabel("Koefisien (Absolut)")
        ax.set_title("Feature Importance")
        st.pyplot(fig_logreg)


    prob_percent = round(prob * 100, 2)

    st.subheader(f"Probabilitas Risiko Stroke: {prob_percent}%")

    if prob >= 0.75:
        st.error("⚠️ Risiko Tinggi terkena stroke")
    elif prob >= 0.5:
        st.warning("⚠️ Risiko Sedang terkena stroke")
    else:
        st.success("✅ Risiko Rendah terkena stroke")

    # --- Tambah Visualisasi fitur penting hanya untuk Random Forest ---
    if model_choice == "Random Forest":
        st.subheader("Visualisasi Feature Importance")

        # Nama fitur yang sudah dipilih selector
        # Selector mengubah fitur input_df, ambil nama fitur dari selector.get_feature_names_out()
        try:
            feature_names = selector.get_feature_names_out()
        except:
            # Jika selector tidak support get_feature_names_out, fallback pakai nama kolom input_df
            feature_names = input_df.columns.tolist()

        importances = rf_model.feature_importances_

        # Buat dataframe untuk plot
        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=True)

        # Bar chart
        fig, ax = plt.subplots()
        ax.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance dari Random Forest')
        st.pyplot(fig)

        # --- SHAP plot ---
        st.subheader("SHAP Summary Plot")

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_selected)

        # Plot SHAP summary bar
        fig_shap = plt.figure()
        shap.summary_plot(shap_values, X_selected, plot_type="bar", show=False)
        st.pyplot(fig_shap)
