# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
# import pickle

# # Load Tokenizer
# try:
#     with open("tokenizer.pkl", "rb") as file:
#         tokenizer = pickle.load(file)
# except FileNotFoundError:
#     st.error("Tokenizer file not found. Please check the file path.")
#     st.stop()

# # Load Emotion Label Encoder
# try:
#     with open("emotion_encoder.pkl", "rb") as file:
#         emotion_encoder = pickle.load(file)
# except FileNotFoundError:
#     st.error("Emotion label encoder file not found.")
#     st.stop()

# # Load Sentiment Label Encoder
# try:
#     with open("sentiment_encoder.pkl", "rb") as file:
#         sentiment_encoder = pickle.load(file)
# except FileNotFoundError:
#     st.error("Sentiment label encoder file not found.")
#     st.stop()

# # Streamlit UI
# st.title("Sentiment & Emotion Analysis")
# st.write("Enter a text below to predict **Sentiment** and **Emotion**.")

# # Text Input Field
# text_input = st.text_area("Enter text for analysis:")

# if st.button("Analyze"):
#     if not text_input.strip():
#         st.warning("⚠️ Input cannot be empty or just spaces.")
#         st.stop()

#     with st.spinner("Analyzing... Please wait."):
#         try:
#             # Load models lazily (only when needed)
#             model_sentiment = tf.keras.models.load_model("sentiment_model.h5")
#             model_emotion = tf.keras.models.load_model("emotion_model.h5")
#         except FileNotFoundError:
#             st.error("Model files not found. Ensure the correct paths.")
#             st.stop()

#         # Tokenize & Pad Input
#         max_length = 100  # Should match training setup
#         sequence = tokenizer.texts_to_sequences([text_input])
#         padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

#         # Predict Sentiment
#         sentiment_pred = model_sentiment.predict(padded_sequence)
#         sentiment_label = sentiment_encoder.inverse_transform([np.argmax(sentiment_pred)])[0]

#         # Predict Emotion
#         emotion_pred = model_emotion.predict(padded_sequence)
#         emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]

#     # Display Results
#     st.subheader("Result")
#     st.success(f"**Predicted Sentiment:** {sentiment_label}")
#     st.info(f"**Predicted Emotion:** {emotion_label}")

# another code------------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image
import cv2
import pytesseract

# Load Tokenizer
@st.cache_resource()
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Tokenizer file not found. Please check the path.")
        st.stop()

tokenizer = load_tokenizer()

# Load Encoders
@st.cache_resource()
def load_encoders():
    try:
        with open("emotion_encoder.pkl", "rb") as file:
            emotion_encoder = pickle.load(file)
        with open("sentiment_encoder.pkl", "rb") as file:
            sentiment_encoder = pickle.load(file)
        return emotion_encoder, sentiment_encoder
    except FileNotFoundError:
        st.error("Encoder files not found. Please check the paths.")
        st.stop()

emotion_encoder, sentiment_encoder = load_encoders()

# Load Models
@st.cache_resource()
def load_models():
    try:
        return (
            tf.keras.models.load_model("sentiment_model.h5"),
            tf.keras.models.load_model("emotion_model.h5")
        )
    except FileNotFoundError:
        st.error("Model files not found. Ensure the correct paths.")
        st.stop()

model_sentiment, model_emotion = load_models()

# Function to Extract Text from Image (OCR)
def extract_text_from_image(image):
    try:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

# Sidebar Navigation

st.sidebar.image("https://marketplace.canva.com/EAFLXxbZKC0/3/0/900w/canva-beige-aesthetic-reviews-instagram-story-POrgTNUD14Q.jpg")  # Ensure you have 'logo.png' in your working directory
page = st.sidebar.radio("Go to:", ["Text Input", "Image Upload","About"])

# Sample Reviews
sample_reviews = [
    "This movie was amazing! I loved every second of it.",
    "The service was terrible, I will never come back.",
    "I feel so happy today, everything is going great!",
    "This book was quite boring, I couldn't finish it.",
]

if page == "Text Input":
    st.title("Social Media Sentiment Analysis using LSTM")
    st.markdown(
    "<img src='https://www.thryv.com/media/2025/02/hero-social-media-management-01.png' width='300' height='175'>",
    unsafe_allow_html=True
)
    st.write("Enter a text below to predict **Sentiment** and **Emotion**.")
    
    # Sample Review Selection
    selected_sample = st.selectbox("Try a sample review:", ["Select", *sample_reviews])
    
    # Text Input Field
    text_input = st.text_area("Enter text for analysis:", selected_sample if selected_sample != "Select" else "")
    
    if st.button("Analyze"):
        if text_input.strip():
            with st.spinner("Analyzing... Please wait."):
                sequence = tokenizer.texts_to_sequences([text_input])
                padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
                sentiment_pred = model_sentiment.predict(padded_sequence)
                sentiment_label = sentiment_encoder.inverse_transform([np.argmax(sentiment_pred)])[0]
                emotion_pred = model_emotion.predict(padded_sequence)
                emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
            st.subheader("Result")
            st.success(f"**Predicted Sentiment:** {sentiment_label}")
            st.info(f"**Predicted Emotion:** {emotion_label}")
        else:
            st.warning("⚠️ Input cannot be empty.")

elif page == "Image Upload":
    st.title("Social Media Sentiment Analysis using LSTM")
    st.write("Upload an image containing a review or text to analyze.")
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg", "webp"])
    
    extracted_text = ""
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        extracted_text = extract_text_from_image(image)
        if extracted_text:
            st.subheader("Extracted Text")
            st.write(extracted_text)
    
    text_input = st.text_area("Enter or edit extracted text for analysis:", extracted_text)
    
    if st.button("Analyze"):
        if text_input.strip():
            with st.spinner("Analyzing... Please wait."):
                sequence = tokenizer.texts_to_sequences([text_input])
                padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
                sentiment_pred = model_sentiment.predict(padded_sequence)
                sentiment_label = sentiment_encoder.inverse_transform([np.argmax(sentiment_pred)])[0]
                emotion_pred = model_emotion.predict(padded_sequence)
                emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
            st.subheader("Result")
            st.success(f"**Predicted Sentiment:** {sentiment_label}")
            st.info(f"**Predicted Emotion:** {emotion_label}")
        else:
            st.warning("⚠️ Input cannot be empty.")

elif page == "About":
    st.title("About This Project")
    st.write(
        "The project Social Media Sentiment Analysis using LSTM that analyzes text from user product or service reviews, "
        "predicting both the sentiment (Positive, Neutral, or Negative) and emotion (Happy, Angry, Sad, etc.). "
        "It leverages deep learning models trained on large datasets for high accuracy predictions. "
        "This tool can be used for analyzing customer reviews, social media posts, and more!"
    )
    st.subheader("Features:")
    st.markdown("Text-based Sentiment & Emotion Analysis")
    st.markdown("Text Extraction from Images")
    st.markdown("Sample Reviews for Quick Testing")
    st.write("Developed using Streamlit, TensorFlow")