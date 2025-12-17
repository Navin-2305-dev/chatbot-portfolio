import os
from datetime import datetime
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --------------------------------------------------
# Setup
# --------------------------------------------------
load_dotenv()

REQUIRED_ENV_VARS = [
    "GEMINI_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "FLASK_SECRET_KEY"
]

missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# --------------------------------------------------
# Gemini Client (NEW SDK)
# --------------------------------------------------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------------------------------
# Vector Store Setup
# --------------------------------------------------
COLLECTION_NAME = "ChatBot-Portfolio"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=20
    )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

# --------------------------------------------------
# LLM Generator
# --------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_response(query, context):
    prompt = f"""
You are Navin Assistant.
Answer ONLY from the given context.
Speak in first person.

Context:
{context}

Question:
{query}

Answer:
"""
    response = gemini_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text.strip()

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.post("/api/chatbot")
def chatbot():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return {"error": "Query cannot be empty"}, 400

    session.setdefault("chat_history", [])

    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(query, k=2)
        context = "\n\n".join(d.page_content for d in docs)

        response = generate_response(query, context)

        session["chat_history"].append({
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })

        return {"response": response}

    except Exception as e:
        app.logger.error(e)
        return {"error": "Internal server error"}, 500

@app.get("/api/chatbot/ping")
def ping():
    return {"status": "alive"}

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )
