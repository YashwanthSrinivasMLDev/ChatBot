import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import joblib # For saving/loading the model and encoder
from custom_dataset import custom_training_data
# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # A good, small, fast sentence embedding model
MODEL_PATH = './models'
CLASSIFIER_MODEL_FILE = os.path.join(MODEL_PATH, 'intent_classifier.joblib')
LABEL_ENCODER_FILE = os.path.join(MODEL_PATH, 'label_encoder.joblib')
EMBEDDING_MODEL_DIR = os.path.join(MODEL_PATH, 'sentence_transformer_model')

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# --- 1. Data Preparation ---
# This is our small, in-memory dataset of intents and example phrases.
# In a real-world scenario, this would come from a CSV, database, or a more complex annotation process.



df = pd.DataFrame(custom_training_data)

# --- 2. Load Sentence Transformer Model ---
@st.cache_resource # Cache the model loading for efficiency in Streamlit
def load_embedding_model():
    """Loads the pre-trained Sentence Transformer model."""
    try:
        # Try loading from local path first
        model = SentenceTransformer(EMBEDDING_MODEL_DIR)
        st.success(f"Loaded embedding model from local cache: {MODEL_NAME}")
    except Exception:
        # If not found locally, download and save
        st.warning(f"Embedding model not found locally. Downloading {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(EMBEDDING_MODEL_DIR)
        st.success(f"Downloaded and cached embedding model: {MODEL_NAME}")
    return model

embedding_model = load_embedding_model()

# --- 3. Train Intent Classifier (Logistic Regression) ---
# This function will be run only once, then cached by Streamlit
@st.cache_resource
def train_intent_classifier(dataframe, _embedding_model):
    """
    Trains a Logistic Regression classifier for intent recognition.
    Caches the trained model and label encoder.
    """
    st.info("Training intent classifier...")

    # Encode labels (convert string intents to numerical IDs)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    dataframe['intent_encoded'] = label_encoder.fit_transform(dataframe['intent'])

    # Generate embeddings for training phrases
    # This can be time-consuming for large datasets
    embeddings = embedding_model.encode(dataframe['phrase'].tolist(), show_progress_bar=True)

    X = embeddings
    y = dataframe['intent_encoded']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train Logistic Regression model
    classifier = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    st.success(f"Intent Classifier Training Complete! Accuracy: {accuracy:.4f}")
    st.text("Classification Report:\n" + report)

    # Save the trained model and label encoder
    joblib.dump(classifier, CLASSIFIER_MODEL_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    st.success("Model and Label Encoder saved.")

    return classifier, label_encoder

# Check if models are already trained and saved, otherwise train
if os.path.exists(CLASSIFIER_MODEL_FILE) and os.path.exists(LABEL_ENCODER_FILE):
    st.info("Loading pre-trained intent classifier and label encoder...")
    classifier = joblib.load(CLASSIFIER_MODEL_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    st.success("Pre-trained models loaded.")
else:
    classifier, label_encoder = train_intent_classifier(df, embedding_model)


# --- 4. Chatbot Response Logic ---
# A dictionary mapping intents to predefined responses
responses = {
    "greet": "Hello! How can I assist you today?",
    "identify": "I am an AI assistant, designed to help you with your queries.",
    "offer_help": "I can help you with a variety of tasks, from answering questions to setting reminders. What do you need?",
    "farewell": "Goodbye! Have a great day!",
    "thank_you": "You're welcome!",
    "weather_query": "I can't check the live weather right now, but I can tell you it's always sunny in the cloud!",
    "tell_joke": "Why don't scientists trust atoms? Because they make up everything!",
    "set_reminder": "I can set reminders for you. What would you like to be reminded about?",
    "play_music": "I can play music for you. What genre or song would you like?",
    "purpose_query": "My purpose is to assist users like you by understanding your requests and providing helpful information.",
    "customer_support": "I can connect you to our customer support team. What is your issue regarding?",
    "password_reset": "To reset your password, please visit our website's 'Forgot Password' section or contact support.",
    "general_knowledge": "That's an interesting question! I can provide general knowledge. What specific fact are you looking for?",
    "unknown": "I'm sorry, I don't understand that request. Could you please rephrase it or ask something different?"
}

def get_chatbot_response(user_input, classifier, label_encoder, embedding_model, confidence_threshold=0.7):
    """
    Predicts the intent of the user input and returns a corresponding response.
    Includes a confidence threshold for 'unknown' intent.
    """
    if not user_input.strip():
        return "Please type something so I can respond!"

    # Get embedding for the user input
    user_embedding = embedding_model.encode([user_input])

    # Predict probabilities for each intent
    probabilities = classifier.predict_proba(user_embedding)[0]
    predicted_intent_id = np.argmax(probabilities)
    confidence = probabilities[predicted_intent_id]

    # Decode the predicted intent ID back to its string label
    predicted_intent = label_encoder.inverse_transform([predicted_intent_id])[0]

    st.write(f"Detected Intent: **{predicted_intent}** (Confidence: {confidence:.2f})")

    # Check confidence threshold
    if confidence < confidence_threshold:
        return responses["unknown"]
    else:
        return responses.get(predicted_intent, responses["unknown"]) # Fallback to unknown if intent not in responses


# --- 5. Streamlit UI ---
st.set_page_config(page_title="Intent-Based AI Chatbot", layout="centered")

st.markdown("""
<style>
.main-title {
    font-size: 2.8em;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 0.1em;
}
.subtitle {
    font-size: 1.2em;
    color: #666666;
    text-align: center;
    margin-top: 0;
    margin-bottom: 1.5em;
}
.chat-message {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 1.05em;
}
.user-message {
    background-color: #e0f7fa; /* Light blue */
    text-align: right;
    margin-left: 20%;
    border-top-right-radius: 0;
}
.bot-message {
    background-color: #f0f0f0; /* Light gray */
    text-align: left;
    margin-right: 20%;
    border-top-left-radius: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🤖 Intent-Based AI Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Understand user intent and provide smart responses.</p>', unsafe_allow_html=True)

st.write("---")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message"><b>Bot:</b> {message["content"]}</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="chat-message user-message"><b>You:</b> {user_input}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        # Get bot response
        bot_response = get_chatbot_response(user_input, classifier, label_encoder, embedding_model)

    # Add bot response to chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    st.markdown(f'<div class="chat-message bot-message"><b>Bot:</b> {bot_response}</div>', unsafe_allow_html=True)

