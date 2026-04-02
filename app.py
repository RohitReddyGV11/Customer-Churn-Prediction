import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered",
)

# ── Custom CSS for premium look ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.65);
        font-size: 1.05rem;
        font-weight: 300;
        margin: 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c5c5c5;
        margin: 1.5rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(255,255,255,0.06);
        letter-spacing: 0.3px;
    }

    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .result-churn {
        background: linear-gradient(135deg, #1a0000, #3d0000);
        border-color: rgba(255, 75, 75, 0.3);
    }
    .result-no-churn {
        background: linear-gradient(135deg, #001a00, #003d00);
        border-color: rgba(75, 255, 75, 0.3);
    }
    .result-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-subtitle {
        font-size: 0.95rem;
        font-weight: 300;
        opacity: 0.7;
    }
    .result-prob {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.8rem 0 0.2rem 0;
        font-variant-numeric: tabular-nums;
    }
    .prob-churn { color: #ff4b4b; }
    .prob-no-churn { color: #00d26a; }
    .prob-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.5;
        font-weight: 500;
    }

    /* Gauge bar */
    .gauge-container {
        margin-top: 1.2rem;
        background: rgba(255,255,255,0.06);
        border-radius: 100px;
        height: 10px;
        overflow: hidden;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 100px;
        transition: width 0.8s ease;
    }
    .gauge-fill-churn {
        background: linear-gradient(90deg, #ff4b4b, #ff8080);
    }
    .gauge-fill-no-churn {
        background: linear-gradient(90deg, #00d26a, #80ffb4);
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35);
    }

    /* Input labels */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        opacity: 0.3;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Artifacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model("artifacts/model.h5")

    with open("artifacts/label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("artifacts/onehot_encoder_geography.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


# ── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔮 Customer Churn Predictor</h1>
    <p>Predict whether a bank customer is likely to leave using a trained neural network</p>
</div>
""", unsafe_allow_html=True)


# ── Input Form ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Customer Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=92, value=35)
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

with col2:
    balance = st.number_input("Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])


# ── Prediction ───────────────────────────────────────────────────────────────
st.markdown("")  # spacer

if st.button("🚀  Predict Churn"):

    # Prepare input data (following the same pipeline as prediction.ipynb)
    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
    }

    input_df = pd.DataFrame([input_data])

    # Label encode Gender
    input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])

    # OneHot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[input_data["Geography"]]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    )

    # Concat: drop Geography, add one-hot columns
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    prediction_prob = float(prediction[0][0])

    # ── Display Result ───────────────────────────────────────────────────
    is_churn = prediction_prob > 0.5
    pct = prediction_prob * 100

    if is_churn:
        st.markdown(f"""
        <div class="result-card result-churn">
            <div class="result-icon">⚠️</div>
            <div class="result-title" style="color:#ff4b4b;">High Churn Risk</div>
            <div class="result-subtitle">This customer is likely to leave the bank</div>
            <div class="result-prob prob-churn">{pct:.1f}%</div>
            <div class="prob-label">churn probability</div>
            <div class="gauge-container">
                <div class="gauge-fill gauge-fill-churn" style="width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-no-churn">
            <div class="result-icon">✅</div>
            <div class="result-title" style="color:#00d26a;">Low Churn Risk</div>
            <div class="result-subtitle">This customer is likely to stay with the bank</div>
            <div class="result-prob prob-no-churn">{pct:.1f}%</div>
            <div class="prob-label">churn probability</div>
            <div class="gauge-container">
                <div class="gauge-fill gauge-fill-no-churn" style="width:{pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Built with Streamlit • ANN Model powered by TensorFlow</div>', unsafe_allow_html=True)
