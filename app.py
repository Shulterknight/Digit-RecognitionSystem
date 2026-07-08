import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Configure the page
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")

# 1. Load the model (Cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Build the UI
st.title("Handwritten Digit Recognizer 🔢")
st.write("Upload an image of a single handwritten digit (0-9) in any color, and the app will dynamically process it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the original uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Upload', width=200)
    st.write("Classifying...")

    # 3. Dynamic Preprocessing Pipeline
    # Step A: Convert to grayscale
    image = ImageOps.grayscale(image)
    img_array = np.array(image)
    
    # Step B: Determine background intensity using the outer edges of the image
    edges = np.concatenate([
        img_array[0, :],   # Top edge
        img_array[-1, :],  # Bottom edge
        img_array[:, 0],   # Left edge
        img_array[:, -1]   # Right edge
    ])
    bg_intensity = np.median(edges)
    
    # Step C: If the background is lighter than the overall average, invert it!
    if bg_intensity > np.mean(img_array):
        image = ImageOps.invert(image)
        img_array = np.array(image) # Update array after inversion
        
    # Step D: Clean up background noise (crush dark grays to pure black)
    # Anything darker than the average pixel becomes 0 (black)
    threshold = np.mean(img_array)
    img_array = np.where(img_array < threshold, 0, img_array)
    
    # Step E: Convert back to image, resize to 28x28, and normalize for the model
    processed_image = Image.fromarray(img_array.astype('uint8'))
    processed_image = processed_image.resize((28, 28))
    
    # Optional: Display the processed image so the user can see what the model sees
    st.image(processed_image, caption='Processed for Model (28x28)', width=100)
    
    final_array = np.array(processed_image).astype('float32') / 255.0
    final_array = final_array.reshape(1, 28, 28, 1)

    # 4. Predict
    prediction = model.predict(final_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # 5. Display Results
    st.success(f"Predicted Digit: **{predicted_digit}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
