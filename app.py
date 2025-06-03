import streamlit as st

st.title("Dashboard Prediksi Stroke")
st.write("Selamat datang! Ini adalah langkah awal dashboard prediksi risiko stroke.")

st.subheader("Masukkan Data Pasien")

# Form input
age = st.slider("Usia", 0, 100, 30)
hypertension = st.radio("Hipertensi?", ["Tidak", "Ya"])
heart_disease = st.radio("Penyakit Jantung?", ["Tidak", "Ya"])
avg_glucose = st.number_input("Kadar Glukosa Rata-rata", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoking_status = st.selectbox("Status Merokok", ["never smoked", "formerly smoked", "smokes", "unknown"])

# Tombol prediksi
if st.button("Prediksi Stroke"):
    # Sementara kita belum sambung ke model asli, tampilkan simulasi
    st.success("Prediksi (simulasi): Tidak berisiko stroke.")
