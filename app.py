import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# prevent the reloading of the model
@st.cache_data()
def load():
    model_path = "model/best_model.h5"
    model = load_model(model_path, compile=False)

    return model


# Model loading
model = load()


def predict(uploaded):
    img = Image.open(uploaded)
    img = np.asarray(img)
    img_resize = cv2.resize(img, (224, 224))

    img_resize = np.expand_dims(img_resize, axis=0)

    pred = model.predict(img_resize)
    rec = pred[0][0]

    return rec


st.title("Smart Trash Project")

upload = st.file_uploader("Chargez l'image de votre objet", type=["png", "jpeg", "jpg"])

c1, c2 = st.columns(2)


if upload:
    proba_rec = predict(upload) * 100
    c1.image(Image.open(upload))
    if proba_rec > 50:
        c2.write(f"Je suis certain à {proba_rec:.2f} % que l'objet est recyclable")
    else:
        c2.write(
            f"Je suis certain à {100 - proba_rec:.2f} % que l'objet n'est pas recyclable"
        )
