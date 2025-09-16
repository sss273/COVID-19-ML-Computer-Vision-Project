# -*- coding: utf-8 -*-



import tensorflow as tf
model = tf.keras.models.load_model('my_model.hdf5')

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
         Covid-19 Prediction using Chest X-rays
         """
         )
st.write("This is a simple image classification web app to predict Covid-19 using Chest X-Ray")
file = st.file_uploader("Please upload an X-Ray image file", type=["jpg", "png", "jpeg"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)*100
        print(prediction)
        return prediction
if file is None:
    st.text("Please upload an X-ray image file")
else:
    image = Image.open(file)
    st.image(image, caption='Patient X-ray image', use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) >= 0.5:
        st.write("The patient is COVID-19")

    else:
        st.write("The patient has Healthy")
    
    st.text("Probability (0: Healthy, 1: COVID-19)")
    st.write(prediction)