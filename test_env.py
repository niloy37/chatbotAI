from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Print all environment variables (without showing their values)
print("Environment variables found:")
for key in ["GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_ENV"]:
    value = os.getenv(key)
    if value:
        print(f"{key}: {'*' * len(value)}")
    else:
        print(f"{key}: Not found") 