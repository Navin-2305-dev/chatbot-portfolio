import os
from datetime import datetime
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
CORS(app, resources={r"/api/*": {"origins": "*"}})

required_keys = {
    "GEMINI_API_KEY": "Gemini API key",
    "QDRANT_URL": "Qdrant URL",
    "QDRANT_API_KEY": "Qdrant API key"
}

missing_keys = [name for name in required_keys if not os.getenv(name)]
if missing_keys:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_query"  # Optimized for queries
)

def get_vector_store():
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=20
        )
        collection_name = "navin_portfolio"

        # Connect to existing collection (assumes it was created with dimension 768)
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_response(query, context):
    try:
        prompt = f"""
        You are **Navin Assistant** – a professional, conversational AI representing Navin B. Answer questions confidently and concisely based *only* on the provided context from Navin's resume and GitHub.

        **Context:**
        {context}

        **Rules:**
        - Speak in first person as Navin ("I", "my", etc.).
        - Be professional, friendly, and engaging.
        - Use bullet points for lists (skills, projects, etc.).
        - Never invent or assume information not in the context.
        - Keep responses balanced: not too short, not overly long.
        - Answer adequately based on the context.

        **User Question:** {query}

        **Your Answer (as Navin):**
        """
        model = genai.GenerativeModel("gemini-1.5-flash")  # Updated to a valid current model; change if needed
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Generation error: {e}")
        return "I'm having trouble processing your request right now."

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
    except Exception:
        return jsonify({"error": "Invalid request data"}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    vector_store = get_vector_store()
    if not vector_store:
        return jsonify({"error": "Knowledge base unavailable"}), 503

    try:
        docs = vector_store.similarity_search(query, k=4)  # Increased to 4 for better context
        context = "\n\n".join(doc.page_content for doc in docs)
        response = generate_response(query, context)

        session["chat_history"].append({
            "query": query,
            "response": response,
            "timestamp": str(datetime.now())
        })
        session.modified = True

        return jsonify({"response": response})
    except Exception as e:
        print(f"Retrieval/generation error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/chat_history", methods=["GET"])
def get_chat_history():
    return jsonify(session.get("chat_history", []))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)