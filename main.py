import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import joblib # For saving/loading the label encoder
import torch # Import PyTorch
import torch.nn as nn # Neural network modules
import torch.optim as optim # Optimizers
from datetime import datetime # For dynamic date/time responses

# --- Import your data from data_loader.py ---
from custom_dataset import custom_training_data
# -------------------------------------------

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # A good, small, fast sentence embedding model
MODEL_PATH = './models'
CLASSIFIER_MODEL_FILE = os.path.join(MODEL_PATH, 'intent_classifier_nn.pt') # Changed file extension for PyTorch model
LABEL_ENCODER_FILE = os.path.join(MODEL_PATH, 'label_encoder.joblib')
EMBEDDING_MODEL_DIR = os.path.join(MODEL_PATH, 'sentence_transformer_model')

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Determine device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# st.info(f"Using device: {device}")

# --- 1. Data Preparation (Now using imported df_data) ---
df = pd.DataFrame(custom_training_data)
# --- 2. Load Sentence Transformer Model ---
@st.cache_resource # Cache the model loading for efficiency in Streamlit
def load_embedding_model():
    """Loads the pre-trained Sentence Transformer model."""
    try:
        # Try loading from local path first
        model = SentenceTransformer(EMBEDDING_MODEL_DIR)
        # st.success(f"Loaded embedding model from local cache: {MODEL_NAME}")
    except Exception:
        # If not found locally, download and save
        st.warning(f"Embedding model not found locally. Downloading {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(EMBEDDING_MODEL_DIR)
        # st.success(f"Downloaded and cached embedding model: {MODEL_NAME}")
    return model

embedding_model = load_embedding_model()

# --- Define the PyTorch Neural Network ---
class IntentClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntentClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU() # ReLU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # No softmax here, CrossEntropyLoss handles it internally
        return x

# --- 3. Train Intent Classifier (Neural Network) ---
@st.cache_resource(hash_funcs={IntentClassifierNN: lambda _: None}) # Add hash_funcs for PyTorch Model
def train_intent_classifier(dataframe, _embedding_model, _device):
    """
    Trains a Neural Network classifier for intent recognition using PyTorch.
    Caches the trained model and label encoder.
    """
    # st.info("Training intent classifier (Neural Network with PyTorch)...")

    # Encode labels (convert string intents to numerical IDs)
    label_encoder = LabelEncoder()
    dataframe['intent_encoded'] = label_encoder.fit_transform(dataframe['intent'])
    num_classes = len(label_encoder.classes_) # Get the number of unique intents

    # Generate embeddings for training phrases
    embeddings = _embedding_model.encode(dataframe['phrase'].tolist(), show_progress_bar=True)
    embedding_dim = embeddings.shape[1] # Get the dimension of the embeddings (e.g., 384 for MiniLM)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(dataframe['intent_encoded'].values, dtype=torch.long) # Labels should be long type for CrossEntropyLoss

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the Neural Network model and move to device
    classifier = IntentClassifierNN(embedding_dim, num_classes).to(_device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss() # Combines LogSoftmax and NLLLoss
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Train the model
    epochs = 20 # Use a small number of epochs for quick demonstration
    batch_size = 32

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        classifier.train() # Set model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(_device), labels.to(_device)

            optimizer.zero_grad() # Zero the gradients
            outputs = classifier(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights

    # Evaluate the model
    classifier.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation for evaluation
        X_test_device = X_test.to(_device)
        y_test_device = y_test.to(_device)

        outputs = classifier(X_test_device)
        _, y_pred_ids = torch.max(outputs, 1) # Get predicted class IDs

        accuracy = accuracy_score(y_test_device.cpu().numpy(), y_pred_ids.cpu().numpy())
        report = classification_report(y_test_device.cpu().numpy(), y_pred_ids.cpu().numpy(),
                                       target_names=label_encoder.classes_, zero_division=0)

    st.success(f"Intent Classifier Training Complete! Accuracy: {accuracy:.4f}")
    st.text("Classification Report:\n" + report)

    # Save the trained PyTorch model's state dictionary and label encoder
    torch.save(classifier.state_dict(), CLASSIFIER_MODEL_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    st.success("Model and Label Encoder saved.")

    return classifier, label_encoder

# Check if models are already trained and saved, otherwise train
if os.path.exists(CLASSIFIER_MODEL_FILE) and os.path.exists(LABEL_ENCODER_FILE):
    # st.info("Loading pre-trained intent classifier (Neural Network with PyTorch) and label encoder...")
    # Load LabelEncoder first to get num_classes and embedding_dim for model initialization
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    num_classes = len(label_encoder.classes_)
    # Dummy encode a phrase to get embedding_dim if model not yet loaded
    # Or, preferably, save embedding_dim during training and load it.
    # For simplicity, we'll re-encode a dummy phrase to get embedding_dim
    dummy_embedding = embedding_model.encode(["dummy phrase"])
    embedding_dim = dummy_embedding.shape[1]

    classifier = IntentClassifierNN(embedding_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_FILE, map_location=device))
    classifier.eval() # Set to evaluation mode after loading
    # st.success("Pre-trained models loaded.")
else:
    classifier, label_encoder = train_intent_classifier(df, embedding_model, device)


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
    "time_query": "The current time is [TIME_PLACEHOLDER].", # Placeholder for dynamic time
    "date_query": "Today's date is [DATE_PLACEHOLDER].", # Placeholder for dynamic date
    "affirmation": "Great! How else can I assist you?",
    "negation": "Understood. Is there anything else I can clarify or help with?",
    "chitchat": "I'm doing well, thank you for asking! How about you?",
    "about_product_service": "I can provide information about our products and services. Which one are you interested in?",
    "feedback": "Thank you for your feedback! It's valuable for my continuous improvement.",
    "unknown": "I'm sorry, I don't understand that request. Could you please rephrase it or ask something different?"
}

def get_chatbot_response(user_input, classifier, label_encoder, _embedding_model, confidence_threshold=0.7):
    """
    Predicts the intent of the user input and returns a corresponding response.
    Includes a confidence threshold for 'unknown' intent.
    """
    if not user_input.strip():
        return "Please type something so I can respond!"

    # Get embedding for the user input and move to device
    user_embedding = torch.tensor(_embedding_model.encode([user_input]), dtype=torch.float32).to(device)

    classifier.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        # Predict logits
        logits = classifier(user_embedding)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_intent_id = torch.argmax(probabilities).item() # .item() to get Python scalar
    confidence = probabilities[predicted_intent_id].item()

    # Decode the predicted intent ID back to its string label
    predicted_intent = label_encoder.inverse_transform([predicted_intent_id])[0]

    st.write(f"Detected Intent: **{predicted_intent}** (Confidence: {confidence:.2f})")

    # Check confidence threshold
    if confidence < confidence_threshold:
        return responses["unknown"]
    else:
        # Dynamic responses for time and date
        if predicted_intent == "date_query":
            return responses["date_query"].replace("[DATE_PLACEHOLDER]", datetime.now().strftime("%B %d, %Y"))
        elif predicted_intent == "time_query":
            return responses["time_query"].replace("[TIME_PLACEHOLDER]", datetime.now().strftime("%I:%M %p"))

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
    color: #000000; /* Explicitly set text color to black */
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