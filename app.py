import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ------------------ CONFIGURASI HALAMAN ------------------
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")

# ------------------ CUSTOM CSS TEMA PINK ------------------
st.markdown("""
    <style>
        .main {
            background-color: #ffe6f0;
        }
        div.block-container {
            background-color: #fff0f5;
            padding: 2rem 2rem;
            border-radius: 15px;
            color: #333333;
        }
        .stButton>button {
            background-color: #ff69b4;
            color: #333333;
            font-weight: bold;
            padding: 0.5em 1.5em;
            border: none;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #ff1493;
            color: white;
        }
        h1, h3, h4 {
            color: #a1005a;
        }
        p, label, .stSelectbox, .stSlider, .stNumberInput {
            color: #333333 !important;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ JUDUL HALAMAN ------------------
st.markdown("<h1 style='text-align: center;'>🌸 Prediksi Kategori Obesitas</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan informasi diri Anda di bawah ini untuk mengetahui prediksi tingkat obesitas menggunakan model terbaik kami.</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ LOAD MODEL & SCALER ------------------
model = pickle.load(open("dt_model.sav", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# ------------------ FORM INPUT ------------------
st.markdown("### 💖 Data Diri Anda")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("👩‍⚕️ Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("🎂 Umur", min_value=1, max_value=100, value=25)
    height = st.number_input("📏 Tinggi Badan (meter)", min_value=1.0, max_value=2.5, value=1.65)
    weight = st.number_input("⚖️ Berat Badan (kg)", min_value=20.0, max_value=150.0, value=60.0)
    family_history = st.selectbox("👨‍👩‍👧‍👦 Riwayat Keluarga Gemuk?", ["yes", "no"])
    FAVC = st.selectbox("🍟 Makanan Kalori Tinggi?", ["yes", "no"])
    FCVC = st.slider("🥗 Frekuensi Makan Sayur", 1.0, 3.0, 2.0)

with col2:
    NCP = st.slider("🍱 Makan per Hari", 1.0, 4.0, 3.0)
    CAEC = st.selectbox("🍩 Camilan Setelah Makan", ["no", "Sometimes", "Frequently", "Always"])
    SMOKE = st.selectbox("🚬 Merokok?", ["yes", "no"])
    CH2O = st.slider("💧 Air per Hari (L)", 0.0, 3.0, 2.0)
    SCC = st.selectbox("🔍 Sadar Kalori?", ["yes", "no"])
    FAF = st.slider("🏃 Aktivitas Fisik (jam/minggu)", 0.0, 3.0, 1.0)
    TUE = st.slider("📺 Waktu Layar (jam/hari)", 0.0, 2.0, 1.0)
    CALC = st.selectbox("🍷 Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    MTRANS = st.selectbox("🚗 Transportasi Utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# ------------------ PREDIKSI ------------------
st.markdown("---")
if st.button("💗 Prediksi Sekarang"):
    input_data = pd.DataFrame([[gender, age, height, weight, family_history, FAVC, FCVC,
                                NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS]],
                              columns=["Gender", "Age", "Height", "Weight", "family_history_with_overweight",
                                       "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC",
                                       "FAF", "TUE", "CALC", "MTRANS"])

    input_data = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🌟 Hasil Prediksi Anda: **{prediction}**")
    st.info("Model ini menggunakan algoritma Decision Tree yang telah dituning untuk memberikan prediksi yang akurat.")

    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 | Prediksi Obesitas by Aneira Vicentiya</p>", unsafe_allow_html=True)
