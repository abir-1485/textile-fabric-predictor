import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model & Preprocessor ---
model = joblib.load('fabric_property_predictor.pkl')
preprocessor = joblib.load('preprocessing_pipeline.pkl')

# --- Page Setup ---
st.set_page_config(page_title="Textile Property Predictor", page_icon="🧵", layout="centered")
st.title("🧵 Textile Fabric Property Prediction System")
st.write("Predict 9 fabric performance parameters using machine learning!")

st.divider()

# --- Input Fields ---
st.header("🔹 Input Fabric Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    warp_count = st.number_input("Warp Count (Ne)", min_value=5.0, max_value=100.0, value=30.0, step=0.5)
    weft_count = st.number_input("Weft Count (Ne)", min_value=5.0, max_value=100.0, value=20.0, step=0.5)
with col2:
    epi = st.number_input("EPI (Ends per Inch)", min_value=10.0, max_value=120.0, value=60.0, step=1.0)
    ppi = st.number_input("PPI (Picks per Inch)", min_value=10.0, max_value=120.0, value=55.0, step=1.0)
with col3:
    gsm = st.number_input("GSM (g/m²)", min_value=50.0, max_value=400.0, value=200.0, step=1.0)
    weave_type = st.selectbox("Weave Type", ["Plain", "Twill", "Satin", "Broken Twill", "Others"])

# --- Predict Button ---
if st.button("🔍 Predict Properties"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Warp Count': [warp_count],
        'Weft Count': [weft_count],
        'EPI': [epi],
        'PPI': [ppi],
        'GSM': [gsm],
        'Weave Type': [weave_type]
    })

    # Preprocess and predict
    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]

    # Output labels
    output_targets = [
        'Tensile Strength (Warp)', 'Tensile Strength (Weft)',
        'Tearing Strength (Warp)', 'Tearing Strength (Weft)',
        'Wash Fastness (Colour Change)',
        'Rubbing Fastness (Dry and Lengthwise)',
        'Dimensional Stability (Length)',
        'Dimensional Stability (Width)',
        'Spirality'
    ]

    st.success("✅ Prediction Successful!")
    st.divider()
    st.header("📊 Predicted Fabric Properties")

    for i, prop in enumerate(output_targets):
        st.write(f"**{prop}:** {prediction[i]:.3f}")

st.divider()
st.caption("Developed by Abir Ahmed | Powered by Machine Learning ⚙️")