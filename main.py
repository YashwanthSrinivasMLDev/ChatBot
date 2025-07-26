import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import os
import joblib # For saving/loading the label encoder
import torch # Import PyTorch
import torch.nn as nn # Neural network modules
import torch.optim as optim # Optimizers
from datetime import datetime # For dynamic date/time responses

# --- Import your data loading function from data_loader.py ---
from custom_dataset import get_training_dataframe
# -----------------------------------------------------------

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # A good, small, fast sentence embedding model
MODEL_PATH = './models'
CLASSIFIER_MODEL_FILE = os.path.join(MODEL_PATH, 'intent_classifier_nn.pt') # PyTorch model
LABEL_ENCODER_FILE = os.path.join(MODEL_PATH, 'label_encoder.joblib')
EMBEDDING_MODEL_DIR = os.path.join(MODEL_PATH, 'sentence_transformer_model')

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# Determine device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.info(f"Using device: {device}")

# --- 1. Data Preparation (Load CLINC150) ---
# Use Streamlit's cache_data to cache the DataFrame loading
@st.cache_data
def load_data():
    # Receive both df_data and intent_names
    # get_training_dataframe() now returns (DataFrame, list_of_intent_names)
    df_data, intent_names = get_training_dataframe()
    return df_data, intent_names

df, original_intent_names = load_data() # Unpack the returned values

if df.empty:
    st.error("Failed to load training data. Please check the `data_loader.py` file and your internet connection.")
    st.stop() # Stop the app if data loading fails

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

# --- Define the PyTorch Neural Network ---
class IntentClassifierNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntentClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) # Increased neurons
        self.dropout1 = nn.Dropout(0.4) # Increased dropout
        self.fc2 = nn.Linear(256, 128) # Increased neurons
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU() # ReLU activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# --- 3. Train Intent Classifier (Neural Network) ---
@st.cache_resource(hash_funcs={IntentClassifierNN: lambda _: None})
def train_intent_classifier(dataframe, _embedding_model, _device, _original_intent_names): # Added _original_intent_names
    """
    Trains a Neural Network classifier for intent recognition using PyTorch.
    Caches the trained model and label encoder.
    """
    st.info("Training intent classifier (Neural Network with PyTorch)... This may take a while for CLINC150.")

    # --- FIX: Adjust LabelEncoder usage for CLINC150's numerical intents ---
    label_encoder = LabelEncoder()
    # Fit the encoder on the *original string intent names* to ensure classes_ is correctly populated
    # This assumes _original_intent_names is a list of strings ordered by their numerical IDs (0, 1, ..., 149)
    label_encoder.fit(_original_intent_names)

    # The 'intent' column from CLINC150 is already numerical (0-149).
    # We just ensure it's used as the target for training and is of integer type.
    dataframe['intent_encoded'] = dataframe['intent'].astype(int)

    num_classes = len(label_encoder.classes_) # Now num_classes will be correct based on string names

    # Generate embeddings for training phrases
    st.info("Generating embeddings for CLINC150 dataset...")
    embeddings = _embedding_model.encode(dataframe['phrase'].tolist(), show_progress_bar=True)
    embedding_dim = embeddings.shape[1] # Get the dimension of the embeddings (e.g., 384 for MiniLM)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(dataframe['intent_encoded'].values, dtype=torch.long)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the Neural Network model and move to device
    classifier = IntentClassifierNN(embedding_dim, num_classes).to(_device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Train the model
    epochs = 100
    batch_size = 64

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(_device), labels.to(_device)

            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.progress((epoch + 1) / epochs, text=f"Training Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")


    # Evaluate the model
    classifier.eval()
    with torch.no_grad():
        X_test_device = X_test.to(_device)
        y_test_device = y_test.to(_device)

        outputs = classifier(X_test_device)
        _, y_pred_ids = torch.max(outputs, 1)

        accuracy = accuracy_score(y_test_device.cpu().numpy(), y_pred_ids.cpu().numpy())
        # --- FIX: Pass the actual string intent names to target_names ---
        # Use the _original_intent_names list directly
        report = classification_report(y_test_device.cpu().numpy(), y_pred_ids.cpu().numpy(),
                                       target_names=_original_intent_names, zero_division=0)

    st.success(f"Intent Classifier Training Complete! Accuracy: {accuracy:.4f}")
    st.text("Classification Report:\n" + report)

    # Save the trained PyTorch model's state dictionary and label encoder
    torch.save(classifier.state_dict(), CLASSIFIER_MODEL_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    st.success("Model and Label Encoder saved.")

    return classifier, label_encoder

# Check if models are already trained and saved, otherwise train
if os.path.exists(CLASSIFIER_MODEL_FILE) and os.path.exists(LABEL_ENCODER_FILE):
    st.info("Loading pre-trained intent classifier (Neural Network with PyTorch) and label encoder...")
    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    num_classes = len(label_encoder.classes_)
    dummy_embedding = embedding_model.encode(["dummy phrase"])
    embedding_dim = dummy_embedding.shape[1]

    classifier = IntentClassifierNN(embedding_dim, num_classes).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_FILE, map_location=device))
    classifier.eval()
    st.success("Pre-trained models loaded.")
else:
    # Pass the original_intent_names to the training function
    classifier, label_encoder = train_intent_classifier(df, embedding_model, device, original_intent_names)


# --- 4. Chatbot Response Logic ---
# A dictionary mapping intents to predefined responses
# NOTE: You will need to expand these responses significantly for 150 intents!
# For now, I've kept a generic "unknown" fallback for intents not explicitly listed.
responses = {
    # CLINC150 has 150 intents. You need to map these numerical IDs to string names
    # and then to responses. For now, we'll use a generic mapping for the first few
    # and a fallback for the rest.
    # The 'intent' column in df is already the numerical ID.
    # We need to map the numerical ID back to the string name for the responses dictionary.
    # Let's create a reverse mapping from numerical ID to string name using the label_encoder.
}

# Dynamically populate responses based on original_intent_names
# This is a placeholder; you'd want more specific responses for each of the 150 intents.
for i, intent_name in enumerate(original_intent_names):
    if intent_name == "oos_other": # Out of scope intent
        responses[intent_name] = "I'm sorry, I don't understand that request. Could you please rephrase it or ask something different?"
    elif intent_name == "greeting":
        responses[intent_name] = "Hello! How can I assist you today?"
    elif intent_name == "goodbye":
        responses[intent_name] = "Goodbye! Have a great day!"
    elif intent_name == "thank_you":
        responses[intent_name] = "You're welcome!"
    elif intent_name == "time":
        responses[intent_name] = "The current time is [TIME_PLACEHOLDER]."
    elif intent_name == "date":
        responses[intent_name] = "Today's date is [DATE_PLACEHOLDER]."
    elif intent_name == "tell_joke":
        responses[intent_name] = "Why don't scientists trust atoms? Because they make up everything!"
    elif intent_name == "weather":
        responses[intent_name] = "I can't check the live weather right now, but I can tell you it's always sunny in the cloud!"
    else:
        # Generic fallback for all other 150 intents if not explicitly defined
        responses[intent_name] = f"I detected the intent '{intent_name}'. How can I help with that?"

# Ensure a general unknown fallback if an intent somehow isn't mapped
responses["unknown"] = "I'm sorry, I don't understand that request. Could you please rephrase it or ask something different?"


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

    # Decode the predicted intent ID back to its string label using the label_encoder
    # label_encoder.inverse_transform expects numerical IDs and returns string labels
    predicted_intent_name = label_encoder.inverse_transform([predicted_intent_id])[0]

    st.write(f"Detected Intent: **{predicted_intent_name}** (Confidence: {confidence:.2f})")

    # Check confidence threshold
    if confidence < confidence_threshold:
        return responses["unknown"]
    else:
        # Dynamic responses for time and date
        if predicted_intent_name == "date": # Use the string name from CLINC150
            return responses["date"].replace("[DATE_PLACEHOLDER]", datetime.now().strftime("%B %d, %Y"))
        elif predicted_intent_name == "time": # Use the string name from CLINC150
            return responses["time"].replace("[TIME_PLACEHOLDER]", datetime.now().strftime("%I:%M %p"))

        # Use the string intent name to get the response
        return responses.get(predicted_intent_name, responses["unknown"])


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
