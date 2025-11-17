import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/nb_pipeline.joblib")
BMW_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/4/44/BMW.svg"

st.set_page_config(page_title="Sales Classification", layout="centered")

css_code = """
<style>
:root {
  --bmw-blue: #2E9AD6;
  --bmw-blue-dark: #0C5F8C;
  --bmw-white: #FFFFFF;
  --bmw-white-soft: #F5F7FA;
}

html, body, [data-testid="stAppViewContainer"] {
  color: var(--bmw-white);
  background:
    radial-gradient(900px 500px at 20% 0%, rgba(46,154,214,0.15), rgba(10,10,10,0.85)),
    url('BMWLOGO') center center / 33% no-repeat,
    linear-gradient(180deg, #000000 0%, #111318 100%);
}

h1, h2, h3, h4, h5, h6 {
  color: var(--bmw-white-soft) !important;
  font-weight: 700;
}

label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stTextArea label {
    color: var(--bmw-white) !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

.stTextInput > div > div,
.stNumberInput > div > div,
div[data-baseweb="select"] > div {
  background: #FFFFFF !important;
  border: 1px solid #CCCCCC !important;
  border-radius: 10px !important;
  color: #000000 !important;
}

.stTextInput input, .stNumberInput input {
  color: #000000 !important;
  font-weight: 600 !important;
  opacity: 1 !important;
}

.stButton > button {
  width: 100px;
  height: 100px;
  border-radius: 9999px !important;
  background:
    url('BMWLOGO') center center / 55% no-repeat,
    radial-gradient(circle at 30% 30%, rgba(255,255,255,0.3), rgba(255,255,255,0.08) 60%),
    linear-gradient(180deg, var(--bmw-blue), var(--bmw-blue-dark));
  border: none;
  box-shadow: 0 10px 20px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.15);
  transition: all .15s ease;
  color: transparent;
}

.stButton > button:hover {
  transform: translateY(-3px);
  box-shadow: 0 14px 26px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.25);
}

.stButton > button:active { 
    transform: scale(0.97);
}
</style>
"""

css_code = css_code.replace("BMWLOGO", BMW_LOGO_URL)

st.markdown(css_code, unsafe_allow_html=True)

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

st.title("Sales Classification BMW ")

try:
    pipe = load_pipeline(MODEL_PATH)
    st.success(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

st.subheader("Input Features")

Mileage_KM = st.number_input("Mileage_KM", value=50000.0, min_value=0.0, step=1000.0)
Price_USD = st.number_input("Price_USD", value=15000.0, min_value=0.0, step=100.0)
Sales_Volume = st.number_input("Sales_Volume", value=100.0, min_value=0.0, step=1.0)
Fuel_Type = st.selectbox("Fuel_Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
Color = st.selectbox("Color", ["White", "Black", "Silver", "Blue", "Red", "Other"])
Region = st.text_input("Region", "MENA")
Model = st.text_input("Model", "Sedan")
Transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

if st.button("Predict"):
    df = pd.DataFrame([{ 
        "Mileage_KM": Mileage_KM,
        "Price_USD": Price_USD,
        "Sales_Volume": Sales_Volume,
        "Fuel_Type": Fuel_Type,
        "Color": Color,
        "Region": Region,
        "Model": Model,
        "Transmission": Transmission
    }])

    try:
        pred = pipe.predict(df)[0]
        st.success(f"Prediction: {pred}")

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(df)[0]
            try:
                classes = pipe.classes_
            except:
                classes = list(range(len(proba)))

            st.json({str(c): float(p) for c, p in zip(classes, proba)})
    except Exception as e:
        st.error(f"Prediction failed: {e}")
