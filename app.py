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

# Load Tokenizer
try:
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
except FileNotFoundError:
    st.error("Tokenizer file not found. Please check the file path.")
    st.stop()

# Load Emotion Label Encoder
try:
    with open("emotion_encoder.pkl", "rb") as file:
        emotion_encoder = pickle.load(file)
except FileNotFoundError:
    st.error("Emotion label encoder file not found.")
    st.stop()

# Load Sentiment Label Encoder
try:
    with open("sentiment_encoder.pkl", "rb") as file:
        sentiment_encoder = pickle.load(file)
except FileNotFoundError:
    st.error("Sentiment label encoder file not found.")
    st.stop()

# Streamlit UI
st.title("Social Media Sentiment Analysis using LSTM")
st.write("Enter a text below to predict **Sentiment** and **Emotion**.")

# Sample Reviews
sample_reviews = [
    "I couldn't stop smiling while watching this movie, absolutely loved it!",
    "The service at this place was so bad, I felt completely ignored.",
    "Woke up feeling super positive today, everything just seems perfect!",
    "Tried reading this book, but it was so dull that I gave up halfway."
]

# Select Sample Review
test_sample = st.selectbox("Try a sample review:", ["Select", *sample_reviews])

# Text Input Field
text_input = st.text_area("Enter text for analysis:", test_sample if test_sample != "Select" else "")

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("⚠️ Input cannot be empty or just spaces.")
        st.stop()

    with st.spinner("Analyzing... Please wait."):
        try:
            # Load models lazily (only when needed)
            model_sentiment = tf.keras.models.load_model("sentiment_model.h5")
            model_emotion = tf.keras.models.load_model("emotion_model.h5")
        except FileNotFoundError:
            st.error("Model files not found. Ensure the correct paths.")
            st.stop()

        # Tokenize & Pad Input
        max_length = 100  # Should match training setup
        sequence = tokenizer.texts_to_sequences([text_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

        # Predict Sentiment
        sentiment_pred = model_sentiment.predict(padded_sequence)
        sentiment_label = sentiment_encoder.inverse_transform([np.argmax(sentiment_pred)])[0]

        # Predict Emotion
        emotion_pred = model_emotion.predict(padded_sequence)
        emotion_label = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]

    # Display Results
    st.subheader("Result")
    st.success(f"**Predicted Sentiment:** {sentiment_label}")
    st.info(f"**Predicted Emotion:** {emotion_label}")

