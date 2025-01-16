import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulated dataset (replace this with your real dataset)
data = pd.DataFrame({
    "magnitude": np.random.uniform(4.0, 9.0, 100),
    "depth": np.random.uniform(1, 700, 100),
    "latitude": np.random.uniform(-90, 90, 100),
    "longitude": np.random.uniform(-180, 180, 100),
    "fault_distance": np.random.uniform(0, 100, 100),
    "high_risk": np.random.choice([0, 1], size=100)  # 0 = Low risk, 1 = High risk
})

# Split the dataset
X = data[["magnitude", "depth", "latitude", "longitude", "fault_distance"]]
y = data["high_risk"]

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Earthquake Risk Prediction")
st.write("This application predicts earthquake risk based on seismic and geological data.")

# User Input
st.sidebar.header("Input Parameters")
magnitude = st.sidebar.slider("Magnitude", 4.0, 9.0, step=0.1, value=5.5)
depth = st.sidebar.slider("Depth (km)", 1, 700, step=1, value=100)
latitude = st.sidebar.slider("Latitude", -90.0, 90.0, step=0.1, value=0.0)
longitude = st.sidebar.slider("Longitude", -180.0, 180.0, step=0.1, value=0.0)
fault_distance = st.sidebar.slider("Distance from Fault Line (km)", 0, 100, step=1, value=10)

# Prepare Input Data
input_data = np.array([[magnitude, depth, latitude, longitude, fault_distance]])
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
risk = "High Risk" if prediction[0] == 1 else "Low Risk"

# Output
st.subheader("Prediction")
st.write(f"The predicted earthquake risk is: **{risk}**")

# Optional: Model Performance
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
