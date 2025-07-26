import pandas as pd
from datasets import load_dataset
import os

# Define a path to cache the dataset locally
DATASET_CACHE_DIR = "./data_cache"
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)


def get_training_dataframe():
    """
    Loads the CLINC150 dataset from Hugging Face and returns it as a Pandas DataFrame.
    Caches the dataset locally after the first download.
    """
    try:
        # Load the 'full' split of the CLINC150 dataset
        # This will download it to DATASET_CACHE_DIR if not already present
        dataset = load_dataset("clinc_oos", "plus", split="train", cache_dir=DATASET_CACHE_DIR)

        # Convert to Pandas DataFrame
        # The dataset has 'text' (user utterance) and 'intent' columns
        df_data = pd.DataFrame(dataset)

        print(f"Successfully loaded CLINC150 dataset. Total phrases: {len(df_data)}")
        print(f"Unique intents: {df_data['intent'].nunique()}")
        print("\nPhrases per intent (top 5):\n", df_data['intent'].value_counts().head())

        return df_data

    except Exception as e:
        print(f"Error loading CLINC150 dataset: {e}")
        print("Please ensure you have an internet connection for the first download.")
        return pd.DataFrame({'phrase': [], 'intent': []})


# The df_data variable is no longer directly defined here,
# instead, it's returned by the function.
# You will call get_training_dataframe() in your main app.


custom_training_data =   {
    'phrase': [
        # Greet (Now with ~50 examples)
        "Hi there", "Hello", "Hey", "Good morning", "Good evening", "Howdy", "Greetings", "What's up?", "Yo", "Hello there",
        "How are you?", "How's it going?", "What's new?", "How do you do?", "Nice to meet you",
        "Hi", "Hey there", "Good day", "Morning", "Evening", "Afternoon", "Sup", "Aloha", "Hola", "Bonjour",
        "How's life?", "Long time no see", "Pleased to meet you", "Glad to see you", "What's happening?",
        "How are things?", "How's everything?", "Good to see you", "Hello bot", "Hi bot",
        "Good afternoon", "Good night", "Greetings to you", "How do you fare?", "It's a pleasure to speak with you",
        "Hello, chatbot", "Hi, assistant", "Hey, AI", "How's your day?", "What's cooking?",
        "Top of the morning to ya", "Salutations", "Warm greetings", "Nice to connect", "Glad you're here",

        # Identify (Now with ~20 examples)
        "What's your name?", "Who are you?", "Tell me about yourself", "Your identity?", "What should I call you?",
        "Are you a bot?", "Are you human?", "Are you an AI?", "What kind of AI are you?", "Who created you?",
        "What's your designation?", "Are you a program?", "What is your purpose?", "Can you introduce yourself?", "Your function?",
        "What's your origin?", "Are you a virtual assistant?", "Do you have a name?", "Tell me your story", "What are you called?",

        # Offer Help (Now with ~20 examples)
        "How can I help you?", "What do you do?", "What services do you offer?", "Help me", "Can you assist me?",
        "What are your capabilities?", "What can you do for me?", "Guide me", "I need assistance", "Show me your features",
        "What kind of support do you provide?", "Can you give me information?", "What tasks can you perform?", "How do you work?", "Explain your functionalities",
        "What are your main functions?", "What's your primary role?", "How can you be useful?", "Tell me your uses", "What services are available?",

        # Farewell (Now with ~20 examples)
        "Bye", "Goodbye", "See you later", "Farewell", "Catch you later",
        "So long", "Adios", "Cheerio", "I'm leaving", "Talk to you soon",
        "Until next time", "Have a good one", "Take care", "I'm off", "Gotta go",
        "See ya", "Later", "Peace out", "Bye bye", "Fare thee well",

        # Thank You (Now with ~20 examples)
        "Thank you", "Thanks a lot", "Appreciate it", "Many thanks", "Cheers",
        "Thank you so much", "I'm grateful", "Much obliged", "Thanks for your help", "You're a lifesaver",
        "I really appreciate your help", "Thanks a million", "That was very helpful, thank you", "Couldn't have done it without you", "Gracias",
        "Much gratitude", "Thanks for everything", "You've been a great help", "I'm thankful", "Many thanks for your assistance",

        # Weather Query (Now with ~20 examples)
        "What is the weather like today?", "Is it sunny outside?", "Will it rain tomorrow?", "What's the forecast?", "Tell me about the weather",
        "Current temperature?", "Weather in London?", "How's the weather in New York?", "Is it going to snow?", "Weather for next week?",
        "What's the climate like?", "Any storms coming?", "What's the humidity?", "Is it hot or cold?", "Weather report please",
        "Tell me the weather conditions", "What's the outlook?", "Is it cloudy?", "What's the wind speed?", "Weather update",

        # Tell Joke (Now with ~20 examples)
        "Tell me a joke", "Make me laugh", "Do you know any jokes?", "Hit me with a joke", "Joke please",
        "Something funny", "Can you tell me something humorous?", "I need a laugh", "Entertain me with a joke", "Tell me something amusing",
        "Crack a joke", "Got any good jokes?", "Tell me a pun", "I'm feeling down, cheer me up with a joke", "A funny story",
        "Surprise me with humor", "Give me some comedy", "Tell me a riddle", "Make me smile", "I want to hear a joke",

        # Set Reminder (Now with ~20 examples)
        "Set an alarm for 7 AM", "Remind me to call mom at 3 PM", "Create a reminder", "Set a reminder for me", "Add a reminder",
        "Remind me about a meeting", "Alarm at 8 o'clock", "Can you set a notification?", "Remind me in 10 minutes", "Schedule a reminder",
        "Set a reminder for tomorrow", "Remind me to buy groceries", "Can you put a note on my calendar?", "Schedule an alert", "Remind me every hour",
        "Create a new reminder", "I need a reminder for later", "Can you notify me?", "Set a daily alarm", "Remind me about an event",

        # Play Music (Now with ~20 examples)
        "Play some music", "Start a song", "Put on a playlist", "Play a tune", "Music on",
        "Play rock music", "Play classical music", "Play my favorite song", "Start playing the album", "Queue up some jazz",
        "Turn on the radio", "Play something relaxing", "I want to hear pop music", "Next song", "Previous track",
        "Shuffle my playlist", "Play a specific artist", "Can you play a podcast?", "Volume up", "Volume down",

        # Purpose Query (Now with ~20 examples)
        "What is your purpose?", "Why were you created?", "What's your function?", "What's your goal?", "What are you for?",
        "What do you aim to do?", "Your objective?", "What's your mission?", "What's your raison d'être?", "What's your role?",
        "What drives you?", "What's your main objective?", "What's your reason for being?", "What are you designed to do?", "What's your ultimate aim?",
        "What's your primary purpose?", "What's your core functionality?", "Why do you exist?", "What's your job?", "Tell me about your existence",

        # Customer Support (Now with ~20 examples)
        "I need assistance with my account", "Help with my order", "I have a billing question", "My product is broken", "I want to file a complaint",
        "Connect me to support", "I need to talk to someone about my subscription", "Technical issue", "My payment failed", "I need help with a return",
        "I have a problem with my service", "My internet is down", "I can't access my features", "I need a refund", "My delivery is late",
        "I have a question about my bill", "Support please", "Can you help me with a technical problem?", "My device isn't working", "I need customer service",

        # Password Reset (Now with ~20 examples)
        "How do I reset my password?", "Forgot my login details", "Password recovery", "I can't log in", "Reset my account password",
        "Lost my password", "Need help with my login", "Can't access my account", "Password assistance", "Unlock my account",
        "My password isn't working", "I need to change my password", "Help with forgotten password", "Account lockout", "How to regain access?",
        "Login issues", "Password help", "My account is locked", "Recover my account", "I forgot my credentials",

        # General Knowledge (Now with ~20 examples)
        "What is the capital of France?", "Who invented the light bulb?", "Tell me a fact", "What is the highest mountain?", "Who wrote Romeo and Juliet?",
        "When was the internet invented?", "What is photosynthesis?", "Explain quantum physics simply", "Tell me about space", "Who is Albert Einstein?",
        "What is the speed of light?", "Who discovered gravity?", "What is the largest ocean?", "Tell me about history", "Explain a scientific concept",
        "What is the boiling point of water?", "Who is the current president of the USA?", "What is the population of China?", "Tell me about dinosaurs", "What is the theory of relativity?",

        # Time Query (Now with ~20 examples)
        "What time is it?", "Current time?", "Time in New York?", "What's the time now?", "Local time?",
        "What's the time in Tokyo?", "Can you tell me the time?", "Time check", "What's the UTC time?", "Time in my city?",
        "What time is it in London?", "What's the exact time?", "Give me the time", "Time please", "Current hour?",
        "What's the time zone?", "Is it morning or evening?", "What time is it in Paris?", "Time right now?", "Can you tell me the current time?",

        # Date Query (Now with ~20 examples)
        "What is today's date?", "What's the date tomorrow?", "Date today?", "Current date?", "What day is it?",
        "Date of next Monday?", "Can you tell me the date?", "What's the full date?", "Date in two weeks?", "What's the day and date?",
        "What's the current day?", "What's the calendar date?", "Today's day and month?", "What's the date in a week?", "Tell me the date",
        "Date of next Tuesday?", "What's the day of the week?", "What's the month and year?", "What's the specific date?", "Can you tell me today's date?",

        # Affirmation (Now with ~20 examples)
        "Yes", "Confirm", "That's right", "Correct", "Okay",
        "Affirmative", "Indeed", "You got it", "Exactly", "Right",
        "Sure", "Absolutely", "Definitely", "That's correct", "Yep",
        "Uh-huh", "Alright", "True", "I agree", "Sounds good",

        # Negation (Now with ~20 examples)
        "No", "Incorrect", "That's wrong", "Not really", "Nope",
        "Negative", "False", "Not true", "Absolutely not", "By no means",
        "Nah", "Not at all", "I don't think so", "Wrong", "That's incorrect",
        "Never", "Unlikely", "Not quite", "No way", "Definitely not",

        # Chitchat (Now with ~20 examples)
        "How are you doing?", "What's new with you?", "Are you busy?", "Tell me something interesting", "How was your day?",
        "What's up?", "How's life?", "Anything exciting happening?", "What's on your mind?", "Feeling good?",
        "How do you feel?", "What's cracking?", "How's your mood?", "Tell me about your day", "Are you having fun?",
        "What's going on?", "How's it hanging?", "Any news?", "What's the latest?", "How's your existence?",

        # About Product/Service (Now with ~20 examples)
        "Tell me about your product X", "What are the features of service Y?", "How much does product Z cost?", "Is product A available?", "Where can I buy service B?",
        "Details about your subscription plan", "What's included in the premium package?", "How does your service work?", "Pricing for product C?", "Can I get a demo of D?",
        "Explain your offerings", "What products do you sell?", "Tell me about your services", "How much is the monthly fee?", "Is there a free trial?",
        "What are the benefits of your product?", "Can I see a catalog?", "Do you have different versions?", "What's the price range?", "How do I sign up for your service?",

        # Feedback (Now with ~20 examples)
        "I have some feedback", "I want to give feedback", "How can I provide feedback?", "Suggestions for improvement", "I have a suggestion",
        "This is great feedback", "I want to report a bug", "My experience was good", "My experience was bad", "I'd like to rate your service",
        "I have a complaint", "I want to leave a review", "How can I submit a suggestion?", "Tell me how to give feedback", "I want to share my thoughts",
        "Can I provide input?", "Where can I leave a comment?", "I have an idea", "Is there a feedback form?", "I want to make a suggestion"
    ],
    'intent': [
        # Greet (EXPANDED)
        "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet",
        "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet",
        "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet",
        "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet",
        "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet",

        "identify", "identify", "identify", "identify", "identify",
        "identify", "identify", "identify", "identify", "identify",
        "identify", "identify", "identify", "identify", "identify",
        "identify", "identify", "identify", "identify", "identify",

        "offer_help", "offer_help", "offer_help", "offer_help", "offer_help",
        "offer_help", "offer_help", "offer_help", "offer_help", "offer_help",
        "offer_help", "offer_help", "offer_help", "offer_help", "offer_help",
        "offer_help", "offer_help", "offer_help", "offer_help", "offer_help",

        "farewell", "farewell", "farewell", "farewell", "farewell",
        "farewell", "farewell", "farewell", "farewell", "farewell",
        "farewell", "farewell", "farewell", "farewell", "farewell",
        "farewell", "farewell", "farewell", "farewell", "farewell",

        "thank_you", "thank_you", "thank_you", "thank_you", "thank_you",
        "thank_you", "thank_you", "thank_you", "thank_you", "thank_you",
        "thank_you", "thank_you", "thank_you", "thank_you", "thank_you",
        "thank_you", "thank_you", "thank_you", "thank_you", "thank_you",

        "weather_query", "weather_query", "weather_query", "weather_query", "weather_query",
        "weather_query", "weather_query", "weather_query", "weather_query", "weather_query",
        "weather_query", "weather_query", "weather_query", "weather_query", "weather_query",
        "weather_query", "weather_query", "weather_query", "weather_query", "weather_query",

        "tell_joke", "tell_joke", "tell_joke", "tell_joke", "tell_joke",
        "tell_joke", "tell_joke", "tell_joke", "tell_joke", "tell_joke",
        "tell_joke", "tell_joke", "tell_joke", "tell_joke", "tell_joke",
        "tell_joke", "tell_joke", "tell_joke", "tell_joke", "tell_joke",

        "set_reminder", "set_reminder", "set_reminder", "set_reminder", "set_reminder",
        "set_reminder", "set_reminder", "set_reminder", "set_reminder", "set_reminder",
        "set_reminder", "set_reminder", "set_reminder", "set_reminder", "set_reminder",
        "set_reminder", "set_reminder", "set_reminder", "set_reminder", "set_reminder",

        "play_music", "play_music", "play_music", "play_music", "play_music",
        "play_music", "play_music", "play_music", "play_music", "play_music",
        "play_music", "play_music", "play_music", "play_music", "play_music",
        "play_music", "play_music", "play_music", "play_music", "play_music",

        "purpose_query", "purpose_query", "purpose_query", "purpose_query", "purpose_query",
        "purpose_query", "purpose_query", "purpose_query", "purpose_query", "purpose_query",
        "purpose_query", "purpose_query", "purpose_query", "purpose_query", "purpose_query",
        "purpose_query", "purpose_query", "purpose_query", "purpose_query", "purpose_query",

        "customer_support", "customer_support", "customer_support", "customer_support", "customer_support",
        "customer_support", "customer_support", "customer_support", "customer_support", "customer_support",
        "customer_support", "customer_support", "customer_support", "customer_support", "customer_support",
        "customer_support", "customer_support", "customer_support", "customer_support", "customer_support",

        "password_reset", "password_reset", "password_reset", "password_reset", "password_reset",
        "password_reset", "password_reset", "password_reset", "password_reset", "password_reset",
        "password_reset", "password_reset", "password_reset", "password_reset", "password_reset",
        "password_reset", "password_reset", "password_reset", "password_reset", "password_reset",

        "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge",
        "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge",
        "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge",
        "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge",

        "time_query", "time_query", "time_query", "time_query", "time_query",
        "time_query", "time_query", "time_query", "time_query", "time_query",
        "time_query", "time_query", "time_query", "time_query", "time_query",
        "time_query", "time_query", "time_query", "time_query", "time_query",

        "date_query", "date_query", "date_query", "date_query", "date_query",
        "date_query", "date_query", "date_query", "date_query", "date_query",
        "date_query", "date_query", "date_query", "date_query", "date_query",
        "date_query", "date_query", "date_query", "date_query", "date_query",

        "affirmation", "affirmation", "affirmation", "affirmation", "affirmation",
        "affirmation", "affirmation", "affirmation", "affirmation", "affirmation",
        "affirmation", "affirmation", "affirmation", "affirmation", "affirmation",
        "affirmation", "affirmation", "affirmation", "affirmation", "affirmation",

        "negation", "negation", "negation", "negation", "negation",
        "negation", "negation", "negation", "negation", "negation",
        "negation", "negation", "negation", "negation", "negation",
        "negation", "negation", "negation", "negation", "negation",

        "chitchat", "chitchat", "chitchat", "chitchat", "chitchat",
        "chitchat", "chitchat", "chitchat", "chitchat", "chitchat",
        "chitchat", "chitchat", "chitchat", "chitchat", "chitchat",
        "chitchat", "chitchat", "chitchat", "chitchat", "chitchat",

        "about_product_service", "about_product_service", "about_product_service", "about_product_service", "about_product_service",
        "about_product_service", "about_product_service", "about_product_service", "about_product_service", "about_product_service",
        "about_product_service", "about_product_service", "about_product_service", "about_product_service", "about_product_service",
        "about_product_service", "about_product_service", "about_product_service", "about_product_service", "about_product_service",

        "feedback", "feedback", "feedback", "feedback", "feedback",
        "feedback", "feedback", "feedback", "feedback", "feedback",
        "feedback", "feedback", "feedback", "feedback", "feedback",
        "feedback", "feedback", "feedback", "feedback", "feedback"
    ]
}