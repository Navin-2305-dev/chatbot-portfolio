import os
from datetime import datetime

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain_community.embeddings import CohereEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------
load_dotenv()

REQUIRED_ENV_VARS = [
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "FLASK_SECRET_KEY"
]

missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# --------------------------------------------------
# Flask App Setup
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=True
)

# --------------------------------------------------
# Gemini Configuration
# --------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="navin-portfolio-chatbot"
)

COLLECTION_NAME = "ChatBot-Portfolio"
VECTOR_SIZE = 384


# --------------------------------------------------
# Qdrant Initialization
# --------------------------------------------------
def initialize_vector_store() -> Qdrant | None:
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=20
        )

        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if COLLECTION_NAME not in collection_names:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )

        return Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )

    except Exception as e:
        app.logger.error(f"Qdrant init failed: {e}")
        return None


# --------------------------------------------------
# LLM Response Generator
# --------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_response(query: str, context: str) -> str:
    prompt = f"""
You are **Navin Assistant** – a professional AI representing Navin B.

------------------------------
Context (Resume + GitHub):
{context}
------------------------------

Rules:
- Speak in first person ("I", "my")
- Use ONLY the provided context
- No assumptions or fabrication
- Be professional and concise
- Use bullet points where helpful

User Question:
{query}

Answer:
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text.strip()


# --------------------------------------------------
# API Routes
# --------------------------------------------------
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    session.setdefault("chat_history", [])

    vector_store = initialize_vector_store()
    if not vector_store:
        return jsonify({"error": "Knowledge base unavailable"}), 503

    try:
        docs = vector_store.similarity_search(query, k=2)
        context = "\n\n".join(doc.page_content for doc in docs)

        response = generate_response(query, context)

        session["chat_history"].append({
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })
        session.modified = True

        return jsonify({"response": response})

    except Exception as e:
        app.logger.error(f"Chatbot error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.get("/api/chatbot/ping")
def ping():
    return jsonify({"status": "alive"})


@app.get("/api/chat_history")
def chat_history():
    return jsonify(session.get("chat_history", []))


# --------------------------------------------------
# App Runner
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
