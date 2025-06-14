# 🩺 Prediksi Obesitas Berdasarkan Gaya Hidup

Aplikasi web berbasis **Streamlit** yang dapat memprediksi tingkat obesitas seseorang berdasarkan berbagai faktor gaya hidup dan kebiasaan kesehatan.

## 🚀 Fitur Utama

- **Input Interaktif**: Masukkan data gaya hidup pengguna melalui antarmuka yang ramah pengguna.
- **Prediksi Obesitas**: Menggunakan berbagai model machine learning:
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- **Visualisasi Data (EDA)**:
  - Distribusi tingkat obesitas
  - Heatmap korelasi antar fitur
- **Evaluasi Model**: Akurasi dan Confusion Matrix
- **Unduh Hasil Prediksi**: Simpan hasil prediksi dalam format CSV.

## 🧪 Dataset

Dataset: `ObesityDataSet.csv`  
Sumber: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Obesity+Level+Estimation)

## ⚙️ Instalasi & Menjalankan Secara Lokal

1. **Clone repositori ini**:
```bash
git clone https://github.com/username/prediksi-obesitas.git
cd prediksi-obesitas
```

2. **Install dependensi**:
```bash
pip install -r requirements.txt
```

3. **Jalankan aplikasi**:
```bash
streamlit run app.py
```

## ☁️ Deploy ke Streamlit Cloud

1. Push ke GitHub
2. Masuk ke [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Klik **New App** dan pilih repository Anda
4. Isi:
   - File utama: `app.py`
   - Branch: `main` atau sesuai repo
5. Klik **Deploy**

## 📷 Screenshot

![demo](https://via.placeholder.com/800x400?text=Contoh+Tampilan+Aplikasi)

## 📦 Requirements

File `requirements.txt`:
```
streamlit
pandas
scikit-learn
matplotlib
seaborn
```

## 🧠 Model dan Proses

1. Data dikodekan menggunakan `LabelEncoder`
2. Data dibagi menjadi data latih dan uji
3. Model dilatih berdasarkan pilihan pengguna
4. Input pengguna diproses dan diprediksi
5. Label hasil dipetakan ke deskripsi yang mudah dimengerti

## 🧑‍💻 Kontributor

- Nama Anda – [GitHub Anda](https://github.com/username)

## 📄 Lisensi

Proyek ini menggunakan lisensi [MIT](LICENSE).