import streamlit as st
import cv2
import joblib
from PIL import Image
import numpy as np

st.toast("Khomaini")
st.image('dell_picture.jpg')
st.title("Testing")
st.text('Crack Detector using KNN')

#Load trained model
model = joblib.load('trained_model_KNN.pkl')

# Preprocessing and feature extraction
def preprocess_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges
 
def extract_features(img):
    resized = cv2.resize(img, (64, 64))
    return resized.flatten().reshape(1, -1)

uploaded_file = st.file_uploader("Upload an image",type=['png','jpg','jpeg'])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L") #convert to grayscale (crack regardless )
    img_array = np.array(image)
    
    #Load image
    st.image( image, caption="You have uploaded this image")
    
    #Process image
    processed_image = preprocess_image(img_array)
    features = extract_features(processed_image)
    
    #Prediction
    prediction = model.predict(features)[0]
    label = "Positive" if prediction == 1 else "Negative"
    
    #give result
    st.success(f"** Prediction: ** {label}")
    # st.balloons()
    # st.snow()
    
    
    
