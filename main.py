import os
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'model.h5'
    
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"{model_path} not found. Current directory: {os.getcwd()}")

st.write("Current working directory:", os.getcwd())

try:
    model = load_model()
    st.write("Model loaded successfully")
except FileNotFoundError as e:
    st.error(f"Error: {e}")

st.write("""
# Weather Classification Group 4 (TORREVILLAS)
""")

uploaded_model_file = st.file_uploader("Upload the model file", type="h5")
if uploaded_model_file is not None:
    with open("model.h5", "wb") as f:
        f.write(uploaded_model_file.getbuffer())
    try:
        model = tf.keras.models.load_model("model.h5")
        st.write("Model loaded successfully from uploaded file")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")

file = st.file_uploader("Upload a weather image: Choose any picture of weather from your device gallery", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file (JPG or PNG)")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ["Cloudy", "Rain", "Shine", "Sunrise"]
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
