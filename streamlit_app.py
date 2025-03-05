import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


st.set_page_config(page_title="🍏 ÉvalMaBouffe", page_icon="🍏", layout="wide")


@st.cache_resource
def load_model():
    try:
        model = joblib.load("model/final_model.joblib")
        one_hot_encoder = joblib.load("model/final_encoder.joblib")  
        return model, one_hot_encoder
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None, None


# Create two columns (for logo and title to be on the same line)
col1, col2 = st.columns([3, 1])  

with col1:
    st.title("🍏 ÉvalMaBouffe")

with col2:
    st.image("images/logo.png", width=150)


st.write("""
ÉvalMaBouffe vous permet d'entrer les informations nutritionnelles d'un produit et de prédire sa note nutritionnelle en fonction de son contenu.

""")

feature_names = [
    'Additives (n)', 'Energy (per 100g)', 'Fat (per 100g)', 'Saturated Fat (per 100g)', 
    'Carbohydrates (per 100g)', 'Sugars (per 100g)', 'Fiber (per 100g)'
]

# sidebar
st.sidebar.header("🔢 Enter Nutritional Information")
features = {}
for feature in feature_names:
    features[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.3f")

# The following fixes the problem of reloading the page whenever an input is entered :

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "predicted_label" not in st.session_state:
    st.session_state.predicted_label = None

st.markdown("---")  # Horizontal Line

# Load model only once (cached in session state)
if not st.session_state.model_loaded:
    st.session_state.rf_model, st.session_state.one_hot_encoder = load_model()
    if st.session_state.rf_model is not None:
        st.session_state.model_loaded = True
    else:
        st.error("⚠️ Model could not be loaded. Please check the files.")

# Prediction Button
if st.button("🚀 Predict Nutrition Grade"):
    if st.session_state.model_loaded:
        # Convert user input to numpy array
        input_data = np.array(list(features.values())).reshape(1, -1)

        # Make the prediction
        prediction = st.session_state.rf_model.predict(input_data)

        # Convert prediction to original label using OneHotEncoder
        predicted_label = st.session_state.one_hot_encoder.inverse_transform(prediction)[0][0]

        # Store prediction result in session state
        st.session_state.prediction_result = predicted_label
        st.session_state.predicted_label = predicted_label
        st.session_state.prediction_done = True  


if st.session_state.prediction_done:
    predicted_label = st.session_state.predicted_label

    # Display the prediction result
    st.success(f"🥗 **Predicted Nutrition Grade**: **{predicted_label}**")
    
    # Show corresponding image based on predicted grade
    image_path = f"images/{predicted_label}.png"
    
    try:
        img = Image.open(image_path)

        # Resize image
        st.image(img, caption=f"Nutrition Grade: {predicted_label}", width=250)
    except FileNotFoundError:
        st.error(f"Image for grade {predicted_label} not found!")

st.markdown("""
<h3 style="font-size: 20px;">📝 Made by:</h3>
<p style="font-size: 18px;"><b>IDRISSI Zineb</b>, <b>El-KILI Rim</b>, <b>ELBAZ Soukaina</b>, <b>NEKRO Doha</b></p>
<h3 style="font-size: 20px;">👨‍🏫 Supervised by:</h3>
<p style="font-size: 18px;"><b>EDDAROUICH Souad</b>, <b>ELOUAFI Abdelamine</b></p>
""", unsafe_allow_html=True)

