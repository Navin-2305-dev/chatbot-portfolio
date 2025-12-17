import os
from datetime import datetime
from flask import Flask, request, session
from flask_cors import CORS
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from google.genai import Client

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Qdrant

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
CORS(app, supports_credentials=True)

# --------------------------------------------------
# Gemini Client
# --------------------------------------------------
gemini_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# --------------------------------------------------
# Gemini Embeddings (CORRECT)
# --------------------------------------------------
class GeminiEmbeddings(Embeddings):
    def __init__(self, model="models/text-embedding-004"):
        self.client = gemini_client
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            res = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            vectors.append(res.embeddings[0].values)
        return vectors

    def embed_query(self, text):
        res = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return res.embeddings[0].values

embeddings = GeminiEmbeddings()

# --------------------------------------------------
# Vector Store (CORRECT INITIALIZATION)
# --------------------------------------------------
COLLECTION_NAME = "ChatBot-Portfolio"

def get_vector_store():
    return Qdrant.from_existing_collection(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
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

        if not docs:
            return {"response": "I don’t have information about that yet."}

        context = "\n\n".join(d.page_content for d in docs)
        response = generate_response(query, context)

        session["chat_history"].append({
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        })

        return {"response": response}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": "Internal server error",
            "details": str(e)
        }, 500

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
