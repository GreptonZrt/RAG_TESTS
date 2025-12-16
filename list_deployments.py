"""Quick script to list available Azure OpenAI deployments."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("API_VERSION", "2024-12-01")

if not endpoint or not api_key:
    print("ERROR: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set.")
    exit(1)

# Azure OpenAI deployments list endpoint
url = f"{endpoint}/openai/deployments?api-version={api_version}"
headers = {"api-key": api_key}

print(f"Fetching deployments from: {endpoint}")
print(f"API version: {api_version}\n")

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if "data" in data and isinstance(data["data"], list):
        deployments = data["data"]
        if not deployments:
            print("No deployments found. Create deployments in Azure Portal first.")
        else:
            print(f"Found {len(deployments)} deployment(s):\n")
            for dep in deployments:
                name = dep.get("id", "unknown")
                model = dep.get("model", "unknown")
                status = dep.get("status", "unknown")
                print(f"  - Deployment: {name}")
                print(f"    Model: {model}")
                print(f"    Status: {status}\n")
            
            print("\nUpdate your .env file with these deployment names:")
            print("EMBEDDING_DEPLOYMENT=<choose-an-embedding-deployment>")
            print("CHAT_DEPLOYMENT=<choose-a-chat-deployment>")
    else:
        print("Unexpected response format:")
        print(data)
        
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
    print(f"Response: {e.response.text if e.response else 'No response'}")
except Exception as e:
    print(f"Error: {e}")
