# import requests

# API_URL = "http://localhost:8000/chat"  # Change if deployed elsewhere

# print("🧠 Internet Search Chatbot (type 'exit' to quit)\n")

# while True:
#     user_input = input("You: ")

#     if user_input.strip().lower() in {"exit", "quit"}:
#         print("Goodbye!")
#         break

#     try:
#         response = requests.post(
#             API_URL,
#             json={"query": user_input},
#             timeout=10
#         )
#         response.raise_for_status()
#         bot_reply = response.json().get("response", "[No response]")
#     except Exception as e:
#         bot_reply = f"Error: {e}"

#     print(f"Bot: {bot_reply}\n")



import requests
import os
from dotenv import load_dotenv

# Load API key from .env or environment variable
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")  # You can also hardcode this if preferred

API_URL = "http://localhost:8000/chat"  # Update if hosted elsewhere

print("🧠 Internet Search Chatbot (type 'exit' to quit)\n")

# Choose chatbot and action type
chatbot_name = "OpenAI"  # Currently only "OpenAI" is supported
action_type = "internet search"  # Or "summary"

while True:
    user_input = input("You: ")

    if user_input.strip().lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    try:
        payload = {
            "chatbot_name": chatbot_name,
            "action_type": action_type,
            "query": user_input,
            "api_key": API_KEY
        }

        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()

        bot_reply = response.json().get("response", "[No response]")
    except Exception as e:
        bot_reply = f"Error: {e}"

    print(f"Bot: {bot_reply}\n")
