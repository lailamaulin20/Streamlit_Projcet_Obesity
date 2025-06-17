import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ====== CSS Tema Pink (Tulisan Lebih Gelap) ======
pink_theme = """
    <style>
    .stApp {
        background-color: #fff0f5;
        font-family: 'Arial', sans-serif;
        color: #4a235a;
    }
    h1, h2, h3, h4 {
        color: #d63384 !important;
        text-align: center;
    }
    .stButton>button {
        background-color: #ff66b2;
        color: black;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3399;
        color: white;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #6f1d1b !important;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #ffccf2;
        border-left: 5px solid #ff66b2;
        color: #4a235a !important;
    }
    </style>
"""
st.markdown(pink_theme, unsafe_allow_html=True)

# ====== Load Model dan Scaler ======
model = pickle.load(open('obesity_model.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ====== Judul Aplikasi ======
st.markdown("<h1>✨ Prediksi Kategori Obesitas ✨</h1>", unsafe_allow_html=True)
st.markdown("🌸 Masukkan data individu untuk mengetahui prediksi tingkat obesitas berdasarkan kebiasaan sehari-hari.")

# ====== Input Pengguna ======
gender = st.selectbox('💖 Jenis Kelamin', ['Male', 'Female'])
age = st.slider('🎂 Umur', 10, 100, 25)
height = st.number_input('📏 Tinggi Badan (meter)', value=1.60, step=0.01)
weight = st.number_input('⚖️ Berat Badan (kg)', value=60.0, step=0.1)
family_history = st.selectbox('👨‍👩‍👧‍👦 Riwayat keluarga obesitas?', ['yes', 'no'])
favc = st.selectbox('🍔 Sering makan makanan berkalori tinggi?', ['yes', 'no'])
fcvc = st.slider('🥗 Frekuensi makan sayur (0-3)', 0.0, 3.0, 2.0)
ncp = st.slider('🍱 Jumlah makanan utama per hari', 1, 4, 3)
caec = st.selectbox('🍪 Ngemil?', ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox('🚬 Merokok?', ['yes', 'no'])
ch2o = st.slider('💧 Konsumsi air putih (liter)', 0.0, 3.0, 2.0)
scc = st.selectbox('📋 Konsultasi kalori?', ['yes', 'no'])
faf = st.slider('🏃 Aktivitas fisik (jam per minggu)', 0.0, 3.0, 1.0)
tue = st.slider('📱 Waktu layar (jam per hari)', 0.0, 3.0, 2.0)
calc = st.selectbox('🍷 Konsumsi alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('🚗 Transportasi utama', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# ====== Buat DataFrame ======
input_dict = {
    'Gender': gender,
    'Age': age,
    'Height': height,
    'Weight': weight,
    'family_history_with_overweight': family_history,
    'FAVC': favc,
    'FCVC': fcvc,
    'NCP': ncp,
    'CAEC': caec,
    'SMOKE': smoke,
    'CH2O': ch2o,
    'SCC': scc,
    'FAF': faf,
    'TUE': tue,
    'CALC': calc,
    'MTRANS': mtrans
}
input_df = pd.DataFrame([input_dict])

# ====== Preprocessing ======
input_encoded = pd.get_dummies(input_df)
model_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)
input_scaled = scaler.transform(input_encoded)

# ====== Prediksi ======
if st.button("💡 Prediksi"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌟 Kategori Obesitas yang Diprediksi: **{prediction.replace('_', ' ').title()}**")
