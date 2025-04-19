import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.h5', compile=False)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

model = load_model()

st.write("""
# Weather Classification
""")

file = st.file_uploader("Upload a Weather Picture: Choose any picture of a weather from your device gallery", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)  # Correct the size to be a tuple
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    
    # Normalize the image data to [0, 1]
    img = img / 255.0

    # Reshape the image to match the model's input shape
    img_reshape = np.expand_dims(img, axis=0)
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
