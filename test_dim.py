from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="test"
)
print(f"Dimension: {len(result.embeddings[0].values)}")
