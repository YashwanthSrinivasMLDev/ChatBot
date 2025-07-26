custom_training_data = {
    'phrase': [
        # Greet
        "Hi there", "Hello", "Hey", "Good morning", "Good evening", "Howdy", "Greetings", "What's up?", "Yo", "Hello there",
        "How are you?", "How's it going?", "What's new?", "How do you do?", "Nice to meet you",

        # Identify
        "What's your name?", "Who are you?", "Tell me about yourself", "Your identity?", "What should I call you?",
        "Are you a bot?", "Are you human?", "Are you an AI?", "What kind of AI are you?", "Who created you?",

        # Offer Help
        "How can I help you?", "What do you do?", "What services do you offer?", "Help me", "Can you assist me?",
        "What are your capabilities?", "What can you do for me?", "Guide me", "I need assistance", "Show me your features",

        # Farewell
        "Bye", "Goodbye", "See you later", "Farewell", "Catch you later",
        "So long", "Adios", "Cheerio", "I'm leaving", "Talk to you soon",

        # Thank You
        "Thank you", "Thanks a lot", "Appreciate it", "Many thanks", "Cheers",
        "Thank you so much", "I'm grateful", "Much obliged", "Thanks for your help", "You're a lifesaver",

        # Weather Query
        "What is the weather like today?", "Is it sunny outside?", "Will it rain tomorrow?", "What's the forecast?", "Tell me about the weather",
        "Current temperature?", "Weather in London?", "How's the weather in New York?", "Is it going to snow?", "Weather for next week?",

        # Tell Joke
        "Tell me a joke", "Make me laugh", "Do you know any jokes?", "Hit me with a joke", "Joke please",
        "Something funny", "Can you tell me something humorous?", "I need a laugh", "Entertain me with a joke", "Tell me something amusing",

        # Set Reminder
        "Set an alarm for 7 AM", "Remind me to call mom at 3 PM", "Create a reminder", "Set a reminder for me", "Add a reminder",
        "Remind me about a meeting", "Alarm at 8 o'clock", "Can you set a notification?", "Remind me in 10 minutes", "Schedule a reminder",

        # Play Music
        "Play some music", "Start a song", "Put on a playlist", "Play a tune", "Music on",
        "Play rock music", "Play classical music", "Play my favorite song", "Start playing the album", "Queue up some jazz",

        # Purpose Query
        "What is your purpose?", "Why were you created?", "What's your function?", "What's your goal?", "What are you for?",
        "What do you aim to do?", "Your objective?", "What's your mission?", "What's your raison d'être?", "What's your role?",

        # Customer Support
        "I need assistance with my account", "Help with my order", "I have a billing question", "My product is broken", "I want to file a complaint",
        "Connect me to support", "I need to talk to someone about my subscription", "Technical issue", "My payment failed", "I need help with a return",

        # Password Reset
        "How do I reset my password?", "Forgot my login details", "Password recovery", "I can't log in", "Reset my account password",
        "Lost my password", "Need help with my login", "Can't access my account", "Password assistance", "Unlock my account",

        # General Knowledge
        "What is the capital of France?", "Who invented the light bulb?", "Tell me a fact", "What is the highest mountain?", "Who wrote Romeo and Juliet?",
        "When was the internet invented?", "What is photosynthesis?", "Explain quantum physics simply", "Tell me about space", "Who is Albert Einstein?",

        # Time Query
        "What time is it?", "Current time?", "Time in New York?", "What's the time now?", "Local time?",
        "What's the time in Tokyo?", "Can you tell me the time?", "Time check", "What's the UTC time?", "Time in my city?",

        # Date Query
        "What is today's date?", "What's the date tomorrow?", "Date today?", "Current date?", "What day is it?",
        "Date of next Monday?", "Can you tell me the date?", "What's the full date?", "Date in two weeks?", "What's the day and date?",

        # Affirmation
        "Yes", "Confirm", "That's right", "Correct", "Okay",
        "Affirmative", "Indeed", "You got it", "Exactly", "Right",

        # Negation
        "No", "Incorrect", "That's wrong", "Not really", "Nope",
        "Negative", "False", "Not true", "Absolutely not", "By no means",

        # Chitchat
        "How are you doing?", "What's new with you?", "Are you busy?", "Tell me something interesting", "How was your day?",
        "What's up?", "How's life?", "Anything exciting happening?", "What's on your mind?", "Feeling good?",

        # About Product/Service (example for a specific product)
        "Tell me about your product X", "What are the features of service Y?", "How much does product Z cost?", "Is product A available?", "Where can I buy service B?",
        "Details about your subscription plan", "What's included in the premium package?", "How does your service work?", "Pricing for product C?", "Can I get a demo of D?",

        # Feedback
        "I have some feedback", "I want to give feedback", "How can I provide feedback?", "Suggestions for improvement", "I have a suggestion",
        "This is great feedback", "I want to report a bug", "My experience was good", "My experience was bad", "I'd like to rate your service"
    ],
    'intent': [
        "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet", "greet",
        "greet", "greet", "greet", "greet", "greet",

        "identify", "identify", "identify", "identify", "identify",
        "identify", "identify", "identify", "identify", "identify",

        "offer_help", "offer_help", "offer_help", "offer_help", "offer_help",
        "offer_help", "offer_help", "offer_help", "offer_help", "offer_help",

        "farewell", "farewell", "farewell", "farewell", "farewell",
        "farewell", "farewell", "farewell", "farewell", "farewell",

        "thank_you", "thank_you", "thank_you", "thank_you", "thank_you",
        "thank_you", "thank_you", "thank_you", "thank_you", "thank_you",

        "weather_query", "weather_query", "weather_query", "weather_query", "weather_query",
        "weather_query", "weather_query", "weather_query", "weather_query", "weather_query",

        "tell_joke", "tell_joke", "tell_joke", "tell_joke", "tell_joke",
        "tell_joke", "tell_joke", "tell_joke", "tell_joke", "tell_joke",

        "set_reminder", "set_reminder", "set_reminder", "set_reminder", "set_reminder",
        "set_reminder", "set_reminder", "set_reminder", "set_reminder", "set_reminder",

        "play_music", "play_music", "play_music", "play_music", "play_music",
        "play_music", "play_music", "play_music", "play_music", "play_music",

        "purpose_query", "purpose_query", "purpose_query", "purpose_query", "purpose_query",
        "purpose_query", "purpose_query", "purpose_query", "purpose_query", "purpose_query",

        "customer_support", "customer_support", "customer_support", "customer_support", "customer_support",
        "customer_support", "customer_support", "customer_support", "customer_support", "customer_support",

        "password_reset", "password_reset", "password_reset", "password_reset", "password_reset",
        "password_reset", "password_reset", "password_reset", "password_reset", "password_reset",

        "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge",
        "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge", "general_knowledge",

        "time_query", "time_query", "time_query", "time_query", "time_query",
        "time_query", "time_query", "time_query", "time_query", "time_query",

        "date_query", "date_query", "date_query", "date_query", "date_query",
        "date_query", "date_query", "date_query", "date_query", "date_query",

        "affirmation", "affirmation", "affirmation", "affirmation", "affirmation",
        "affirmation", "affirmation", "affirmation", "affirmation", "affirmation",

        "negation", "negation", "negation", "negation", "negation",
        "negation", "negation", "negation", "negation", "negation",

        "chitchat", "chitchat", "chitchat", "chitchat", "chitchat",
        "chitchat", "chitchat", "chitchat", "chitchat", "chitchat",

        "about_product_service", "about_product_service", "about_product_service", "about_product_service", "about_product_service",
        "about_product_service", "about_product_service", "about_product_service", "about_product_service", "about_product_service",

        "feedback", "feedback", "feedback", "feedback", "feedback",
        "feedback", "feedback", "feedback", "feedback", "feedback"
    ]
}
