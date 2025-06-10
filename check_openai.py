import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Loads .env from the current directory or parent
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("🔴 OpenAI API Key not found in .env file.")
else:
    try:
        client = OpenAI(api_key=API_KEY)
        client.models.list()  # This is the actual API call
        print("✅ OpenAI API Key is working and successfully authenticated!")
    except Exception as e:
        print(f"🔴 OpenAI API Key test failed: {e}")
