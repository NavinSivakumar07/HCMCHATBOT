from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Try to specify dimension if the model supports it (text-embedding-004 supports it)
try:
    result = client.models.embed_content(
        model="text-embedding-004",
        contents="test",
        config={
            "output_dimensionality": 1024
        }
    )
    print(f"Success! Dimension: {len(result.embeddings[0].values)}")
except Exception as e:
    print(f"Failed: {e}")
