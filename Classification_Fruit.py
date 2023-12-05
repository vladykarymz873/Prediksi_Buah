import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import requests

# Load the pre-trained model using TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="model5.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fungsi untuk melakukan prediksi
def predict_fruits(img):
    # Praproses gambar
    img = tf.image.resize(img, (32, 32))

    # Menambahkan dimensi batch (batch size = 1)
    img = tf.expand_dims(img, axis=0)

    # Set the input tensor to the input data
    input_tensor_index = input_details[0]['index']
    interpreter.set_tensor(input_tensor_index, img.numpy())  # Assuming img is a TensorFlow tensor

    # Run inference
    interpreter.invoke()

    # Get the output results
    output_tensor_index = output_details[0]['index']
    predictions = interpreter.get_tensor(output_tensor_index)

    # Assuming predictions is a 2D array, get the predicted class for the first image
    predicted_class = np.argmax(predictions[0])

    # Get the class label
    labels = {
        0: "Apple Braeburn",
        1: "Apple Granny Smith",
        2: "Apricot",
        3: "Avocado",
        4: "Banana",
        5: "Blueberry",
        6: "Cactus fruit",
        7: "Cantaloupe",
        8: "Cherry",
        9: "Clementine",
        10: "Corn",
        11: "Cucumber Ripe",
        12: "Grape Blue",
        13: "Kiwi",
        14: "Lemon",
        15: "Limes",
        16: "Mango",
        17: "Onion White",
        18: "Orange",
        19: "Papaya",
        20: "Passion Fruit",
        21: "Peach",
        22: "Pear",
        23: "Pepper Green",
        24: "Pepper Red",
        25: "Pineapple",
        26: "Plum",
        27: "Pomegranate",
        28: "Potato Red",
        29: "Raspberry",
        30: "Strawberry",
        31: "Tomato",
        32: "Watermelon"
    }
    predict_fruits_result = labels.get(predicted_class, 'Tidak Diketahui')
    probability = np.max(predictions[0])  # Menggunakan probabilitas tertinggi sebagai hasil

    return predict_fruits_result, probability


# Function to display image and prediction
def display_image_and_prediction(img, predict_fruits_result, probability):
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.success(f"Nama Buahnya adalah: {predict_fruits_result} dengan probabilitas: {probability:.2%}")

# Streamlit UI
st.title("Prediksi Buah-buahan Tropis dan Non-tropis")

# Option to upload image file
uploaded_file = st.file_uploader("Masukan Image Buah-buahan...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Make predictions when the user clicks the button
    if st.button("Predict"):
        predict_fruits_result, probability = predict_fruits(img)
        display_image_and_prediction(img, predict_fruits_result, probability)

# Option to upload image through URL
image_url = st.text_input("Masukan URL Image Buah-buahan:")
if image_url:
    try:
        # Fetch the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Open and display the image
        img = Image.open(BytesIO(response.content))
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        # Make predictions when the user clicks the button
        if st.button("Predict"):
            predict_fruits_result, probability = predict_fruits(img)
            display_image_and_prediction(img, predict_fruits_result, probability)
    except Exception as e:
        st.error(f"Error: {e}")
