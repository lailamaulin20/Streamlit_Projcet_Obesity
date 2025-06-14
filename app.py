import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import io

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("ObesityDataSet.csv")

# Preprocessing function
def preprocess_data(df):
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    return df_encoded, label_encoders

# Train model
def train_model(X, y, model_type):
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# Mapping hasil prediksi ke label yang lebih ramah pengguna
prediksi_label_map = {
    "Insufficient_Weight": "Berat Badan Kurang",
    "Normal_Weight": "Normal Height",
    "Overweight_Level_I": "Sedikit Gemuk",
    "Overweight_Level_II": "Gemuk",
    "Obesity_Type_I": "Obesitas Ringan",
    "Obesity_Type_II": "Obesitas Sedang",
    "Obesity_Type_III": "Obesitas Parah"
}

# Aplikasi Streamlit
st.set_page_config(page_title="Prediksi Obesitas", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Prediksi Obesitas Berdasarkan Gaya Hidup")
st.write("Gunakan aplikasi ini untuk memprediksi kemungkinan obesitas berdasarkan data kesehatan dan gaya hidup Anda.")

# Load dan proses data
data = load_data()
df_encoded, label_encoders = preprocess_data(data)
X = df_encoded.drop("NObeyesdad", axis=1)
y = df_encoded["NObeyesdad"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar untuk kontrol
st.sidebar.header("⚙️ Konfigurasi Model")
model_choice = st.sidebar.selectbox("Pilih Model", ["Random Forest", "Logistic Regression", "KNN", "Decision Tree"])

# Latih model
model = train_model(X_train, y_train, model_choice)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Tampilan Prediksi
st.header("📥 Input Data Pengguna")
user_input = {}
with st.form(key='prediction_form'):
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            if data[col].dtype == 'object':
                user_input[col] = st.selectbox(col, options=data[col].unique(), key=col)
            else:
                user_input[col] = st.slider(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()), key=col)
    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    prediction = model.predict(input_df)[0]
    prediction_label = label_encoders['NObeyesdad'].inverse_transform([prediction])[0]
    hasil_akhir = prediksi_label_map.get(prediction_label, prediction_label)

    st.success(f"✅ Hasil Prediksi: **{hasil_akhir}**")

    # Unduh hasil
    csv = input_df.copy()
    csv['Prediksi'] = hasil_akhir
    csv_download = csv.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Unduh Hasil Prediksi", data=csv_download, file_name="prediksi_obesitas.csv", mime='text/csv')

# Statistik Ringkasan
with st.expander("📊 Statistik Ringkasan"):
    st.write(data.describe())
    st.write("Missing Values:", data.isnull().sum())

# Visualisasi
with st.expander("📈 Visualisasi Data (EDA)"):
    st.subheader("Distribusi Obesitas")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x="NObeyesdad", order=data['NObeyesdad'].value_counts().index, ax=ax, palette="pastel")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi")
    corr = df_encoded.corr()
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# Evaluasi Model
with st.expander("📉 Evaluasi Model"):
    st.metric("Akurasi", f"{acc:.2%}")
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm))
