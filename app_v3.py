import streamlit as st
import librosa
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import load_model
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Load the saved model
model = load_model('audio_classification_model.h5')

# Define the target shape for input spectrograms
target_shape = (128, 128)

# Define your class labels
classes = ['females', 'males']

# Function to preprocess and classify an audio file
def preprocess_and_predict(audio_file):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    # Make predictions
    predictions = model.predict(mel_spectrogram)

    # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return classes[predicted_class_index], class_probabilities[predicted_class_index], class_probabilities.tolist()

# Streamlit UI
st.title('Audio Classification')

# Add file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Classifying...")

    # Call the prediction function
    predicted_class, accuracy, class_probabilities = preprocess_and_predict(uploaded_file)

    # Display results
    st.write("Predicted Class:", predicted_class)
    st.write("Accuracy:", accuracy)
    st.write("Class Probabilities:")
    for i, class_label in enumerate(classes):
        st.write(f"Class: {class_label}, Probability: {class_probabilities[i]}")
