import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk memuat model TensorFlow Lite
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Fungsi untuk memproses gambar dan mendapatkan prediksi
def process_image(image, input_size):
    # Mengubah gambar ke format yang sesuai dengan model
    image = image.resize(input_size)
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalisasi

    # Menambahkan dimensi batch dan mengonversi ke tipe data float32
    input_data = np.expand_dims(image_array, axis=0).astype(np.float32)
    return input_data

# Fungsi untuk mendapatkan kelas prediksi dan probabilitas dari output model
def get_prediction_and_probability(output_data):
    class_index = np.argmax(output_data)
    probability = tf.nn.softmax(output_data[0])[class_index]
    return class_index, probability.numpy()

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Aplikasi Prediksi Buah-buahan")
    uploaded_file = st.file_uploader("Unggah gambar buah-buahan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
        st.write("")
        st.write("Prediksi:")

        # Muat model TensorFlow Lite
        model_path = "model2.tflite"  # Ganti dengan path model Anda
        interpreter = load_model(model_path)

        # Praproses gambar
        input_size = (32, 32)  # Sesuaikan dengan ukuran input model
        image = Image.open(uploaded_file)
        input_data = process_image(image, input_size)

        # Mengambil input dan output tensors dari model
        input_tensor_index = interpreter.get_input_details()[0]['index']
        output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

        # Menjalankan model
        interpreter.set_tensor(input_tensor_index, input_data)
        interpreter.invoke()

        # Mendapatkan prediksi dan probabilitas
        class_index, probability = get_prediction_and_probability(output()[0])

        # Menampilkan hasil prediksi
        class_name = class_names[class_index]
        st.write(f"Kelas Prediksi: {class_name}")
        st.write(f"Probabilitas: {probability:.2%}")

if __name__ == "__main__":
    main()
