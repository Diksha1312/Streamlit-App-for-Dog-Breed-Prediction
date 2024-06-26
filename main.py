import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load model
model = load_model("dog_breed.h5")

# Names of classes
CLASS_NAMES = ['scottish_deerhound', 'entlebucher', 'maltese_dog']

# Set title of app
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Upload dog image
dog_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
submit = st.button("Predict")

# After clicking submit
if submit:
    if dog_image is not None:
        # Convert to OpenCV image
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Display image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image")
        
        # Resize image to match model's expected sizing
        resized_image = cv2.resize(opencv_image, (224, 224))
        
        # Preprocess image for model prediction
        resized_image = resized_image.astype('float32') / 255.0
        resized_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension
        
        # Make prediction
        Y_pred = model.predict(resized_image)
        
        # Display predicted dog breed
        predicted_class = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(f"The predicted dog breed is {predicted_class}")
