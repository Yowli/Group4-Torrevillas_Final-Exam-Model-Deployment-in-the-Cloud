import streamlit as st
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from PIL import Image, ImageOps
import numpy as np

# Define a custom loss function for loading the model
def custom_categorical_crossentropy(y_true, y_pred):
    return CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5', custom_objects={'categorical_crossentropy': custom_categorical_crossentropy})
    return model

model = load_model()

st.write("""
# Weather Classification Group 4 (TORREVILLAS, ROCHA)
""")

file = st.file_uploader("Upload a Weather Picture: Choose any picture of a weather from your device gallery", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)  
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ["Cloudy", "Rain", "Sunshine", "Sunrise"]
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
