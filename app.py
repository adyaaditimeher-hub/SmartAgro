import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation & Soil Health System",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Recommendation & Soil Health Monitoring System")
st.write("A Machine Learning based decision support system for farmers")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("Crop_recommendation.csv")

data = load_dataset()

# --------------------------------------------------
# SHOW DATASET COLUMNS (for verification)
# --------------------------------------------------
st.write("Dataset Columns:", data.columns.tolist())

# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------
# NOTE: Target column name is 'label'
X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X_train, y_train)

# --------------------------------------------------
# MODEL ACCURACY
# --------------------------------------------------
accuracy = accuracy_score(y_test, model.predict(X_test))

# --------------------------------------------------
# USER INPUT SECTION
# --------------------------------------------------
st.sidebar.header("🧪 Enter Soil & Climate Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0, 200, 50)
P = st.sidebar.number_input("Phosphorus (P)", 0, 200, 50)
K = st.sidebar.number_input("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 70.0)
ph = st.sidebar.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🌱 Predict Best Crop"):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    predicted_crop = model.predict(input_scaled)[0]

    # --------------------------------------------------
    # SOIL HEALTH ANALYSIS
    # --------------------------------------------------
    if N >= 80 and P >= 40 and K >= 40 and 6.0 <= ph <= 7.5:
        soil_health = "Healthy Soil"
    elif N >= 40 and P >= 20 and K >= 20:
        soil_health = "Moderate Soil"
    else:
        soil_health = "Poor Soil"

    # --------------------------------------------------
    # FERTILIZER RECOMMENDATION
    # --------------------------------------------------
    fertilizer = []

    if N < 40:
        fertilizer.append("Urea (Nitrogen)")
    if P < 20:
        fertilizer.append("DAP (Phosphorus)")
    if K < 20:
        fertilizer.append("Potash (Potassium)")

    if not fertilizer:
        fertilizer.append("No fertilizer required")

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------
    st.success(f"✅ Recommended Crop: **{predicted_crop.upper()}**")
    st.info(f"🌍 Soil Health Status: **{soil_health}**")

    st.subheader("🧪 Fertilizer Recommendation")
    for f in fertilizer:
        st.write("•", f)

    st.subheader("📊 Model Performance")
    st.write(f"Random Forest Accuracy: **{accuracy:.2f}**")

# --------------------------------------------------
# FUTURE SCOPE
# --------------------------------------------------
st.markdown("---")
st.subheader("🔮 Future Scope")
st.write(
    "Crop disease detection using Deep Learning (CNN) with leaf image datasets "
    "can be added as a future enhancement."
)
