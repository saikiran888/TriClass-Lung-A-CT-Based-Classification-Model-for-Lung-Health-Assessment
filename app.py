import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the trained model
model = load_model('cancer_classification_model.keras')

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to the input size of the model
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=-1)  # Add the channel dimension (since the model expects 1 channel)
    img = np.expand_dims(img, axis=0)  # Add the batch dimension
    return img

# Function to make predictions
def predict_image(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    class_names = ['Benign', 'Malignant', 'Normal']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, prediction

# Streamlit app
st.title("Cancer Classification Prediction🧑‍⚕️")

# Display a brief description of the app
st.write("""
    This app classifies images of cancerous tissue into three categories:
    - **Benign**: Non-cancerous tissue.
    - **Malignant**: Cancerous tissue.
    - **Normal**: Healthy tissue.

    Upload an image of tissue to get a prediction. The model accepts both **JPG** and **PNG** formats.
""")

# File uploader widget, allowing JPG and PNG images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    img = image.load_img(uploaded_file)
    img_array = np.array(img)

    # Show the uploaded image
    st.image(img_array, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predicted_class, prediction = predict_image(img_array)

    # Display the result
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"### Prediction Probabilities:")
    st.write(f"Benign: {prediction[0][0]*100:.2f}%")
    st.write(f"Malignant: {prediction[0][1]*100:.2f}%")
    st.write(f"Normal: {prediction[0][2]*100:.2f}%")
