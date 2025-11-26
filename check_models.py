import os
import google.genai

def check_setup():
    # 1. Check if API Key is visible
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: 'GOOGLE_API_KEY' environment variable not found.")
        print("   Did you run 'source ~/.bashrc'?")
        return
    else:
        print(f"✅ API Key found: {api_key[:5]}...*******")

    # 2. List available models
    print("\n🔍 Fetching available models from Google AI Studio...")
    try:
        client = google.genai.Client(api_key=api_key)
        # We list models to see exactly what strings are valid
        # (The syntax depends slightly on the SDK version, this is standard)
        for model in client.models.list(config={"query_base": True}):
            # Filter for Gemini models to keep output clean
            if "gemini" in model.name:
                print(f"   - {model.name.replace('models/', '')}")
                
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    check_setup()