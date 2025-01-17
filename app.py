#Importing the necessary libraries

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

#Loading the model

model = tf.keras.models.load_model('Cifar_Model.h5')

#Model Labels

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Sheep', 'Truck']

#Function to resize the user image
def resize_image(image):
    image = image.resize((32,32), resample = Image.Resampling.BILINEAR)
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis = 0)
    return image


#Function to predict
def predict(image):
    processed_image = resize_image(image)
    pred = model.predict(processed_image)
    return labels[np.argmax(pred)]


#Building the streamlit interface

st.title('Prediction using the Cifar_10 Model')
st.write('Upload an image to make some predictions with the model')

uploaded_file = st.file_uploader("Choose an image to upload", type = ['jpg', 'jpeg', 'bmp', 'png'])

#Compiling everything together

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_resized = image.resize((128, 128), resample=Image.Resampling.BILINEAR)
    st.image(img_resized, caption="Uploaded image", use_container_width=True)

    st.write("Predicting...")
    prediction = predict(image)
    st.write(f'The prediction is: {prediction}')