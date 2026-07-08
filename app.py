import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Configure the page
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")

# 1. Load the model (Cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    # Make sure 'mnist_cnn.keras' is uploaded to your GitHub repo alongside this script
    return tf.keras.models.load_model('mnist_cnn.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Build the UI
st.title("Handwritten Digit Recognizer 🔢")
st.write("Upload an image of a single handwritten digit (0-9) to see the model's prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=200)
    st.write("Classifying...")

# 3. Preprocess the image to match MNIST format
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # ADD THIS LINE: Invert the image colors (makes white background black, and black text white)
    image = ImageOps.invert(image)
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize (0 to 1)
    img_array = np.array(image).astype('float32') / 255.0
    
    # Reshape to (1, 28, 28, 1) to match the CNN input shape
    img_array = img_array.reshape(1, 28, 28, 1)

    # 4. Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # 5. Display Results
    st.success(f"Predicted Digit: **{predicted_digit}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
