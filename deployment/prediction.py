import os 
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model(r'C:\Users\C O R E I 5\Documents\Hacktiv8\fase 2\best_model (1).h5')

def run():
    # Function to perform prediction
    def predict_image(img_path):
        # Load the image and preprocess it
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability
        
        return predicted_class, prediction

    # Class names for the road signs (adjust based on your model's classes)
    class_names = ['trafficlight', 'stop', 'speedlimit', 'crosswalk']

    # Streamlit app layout
    st.title("Road Sign Detection and Classification")

    # Upload images
    uploaded_files = st.file_uploader("Upload road sign images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # If files are uploaded, proceed with predictions
    if uploaded_files:
        # Number of uploaded images
        num_images = len(uploaded_files)
        cols = 4  # Number of columns in the grid
        rows = (num_images // cols) + (num_images % cols > 0)  # Calculate the number of rows needed

        # Create a grid for displaying images and predictions
        st.write("### Predictions for uploaded images")
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))

        # Loop through each uploaded file and predict
        for i, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file)
            img_path = f"temp_{i}.png"  # Temporary save for prediction
            img.save(img_path)

            # Perform inference
            predicted_class, prediction_probs = predict_image(img_path)

            # Display the image and prediction in a grid
            ax = axes[i // cols, i % cols]
            ax.imshow(img)
            ax.set_title(f"Predicted: {class_names[predicted_class[0]]}")
            ax.axis('off')

            # Delete the temp file after prediction
            os.remove(img_path)

        # Adjust layout and show the grid
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == '__main__':
    run()