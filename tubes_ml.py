import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("📊 Analisis Konsumsi Alkohol Mahasiswa (student-por.csv)")

# Load dataset secara otomatis
try:
    df = pd.read_csv("student-por.csv")
    st.success("Dataset berhasil dimuat: student-por.csv")
except FileNotFoundError:
    st.error("File student-por.csv tidak ditemukan di direktori.")
    st.stop()

# Preview dataset
st.subheader("📁 Dataset Preview")
st.dataframe(df)

# Filter interaktif berdasarkan beberapa kolom kategorik
st.sidebar.title("🔎 Filter Data")

# Filter school
school_options = df["school"].unique()
school_filter = st.sidebar.multiselect("School", school_options, default=school_options)

# Filter sex
sex_options = df["sex"].unique()
sex_filter = st.sidebar.multiselect("Sex", sex_options, default=sex_options)

# Filter address
address_options = df["address"].unique()
address_filter = st.sidebar.multiselect("Address", address_options, default=address_options)

# Filter Pstatus
pstatus_map = {"T": "Together", "A": "Apart"}
pstatus_options = df["Pstatus"].unique()
pstatus_filter = st.sidebar.multiselect(
    "Parent Status (Pstatus)",
    options=pstatus_options,
    format_func=lambda x: f"{pstatus_map.get(x, x)} ({x})",
    default=pstatus_options
)

# Terapkan filter ke dataset
df = df[
    df["school"].isin(school_filter) &
    df["sex"].isin(sex_filter) &
    df["address"].isin(address_filter) &
    df["Pstatus"].isin(pstatus_filter)
]

# Info dataset
st.subheader("📄 Info Dataset")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# Statistik deskriptif
st.subheader("📈 Descriptive Statistics")
st.write(df.describe())

# Visualisasi Walc
st.subheader("🍷 Distribusi Konsumsi Alkohol di Akhir Pekan (Walc)")
fig, ax = plt.subplots()
sns.countplot(x="Walc", data=df, ax=ax)
ax.set_title("Distribusi Konsumsi Alkohol di Akhir Pekan")
st.pyplot(fig)

# Encoding categorical columns
categorical_cols = df.select_dtypes(include='object').columns
df_encoded = df.copy()
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Target: high alcohol jika Walc > 3
df_encoded['high_alcohol'] = (df['Walc'] > 3).astype(int)

X = df_encoded.drop(columns=['Walc', 'high_alcohol'])
y = df_encoded['high_alcohol']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
st.subheader("🔧 Hyperparameter Tuning & Training")

with st.spinner("Training Logistic Regression..."):
    param_grid_lr = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear']
    }
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
    grid_lr.fit(X_train, y_train)

with st.spinner("Training Random Forest..."):
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
    grid_rf.fit(X_train, y_train)

with st.spinner("Training KNN..."):
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
    grid_knn.fit(X_train, y_train)

# Tampilkan hasil tuning
st.subheader("📌 Hasil Hyperparameter Tuning")
st.write("**Logistic Regression** Best Params:", grid_lr.best_params_)
st.write("Accuracy:", grid_lr.best_score_)

st.write("**Random Forest** Best Params:", grid_rf.best_params_)
st.write("Accuracy:", grid_rf.best_score_)

st.write("**KNN** Best Params:", grid_knn.best_params_)
st.write("Accuracy:", grid_knn.best_score_)

# Evaluasi sem
