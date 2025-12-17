import os
from datetime import datetime
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_cohere import CohereEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
CORS(app, resources={r"/api/*": {"origins": "*"}})

required_keys = {
    "GEMINI_API_KEY": "Gemini API key",
    "COHERE_API_KEY": "Cohere API key",
    "QDRANT_URL": "Qdrant URL",
    "QDRANT_API_KEY": "Qdrant API key"
}

missing_keys = [name for name in required_keys if not os.getenv(name)]
if missing_keys:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="navin-portfolio-chatbot"
)

def initialize_vector_store():
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=20
        )
        
        collection_name = "navin_portfolio"
        
        try:
            collection_info = client.get_collection(collection_name)
            if collection_info.config.params.vectors.size != 384:
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
                )
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )

        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_response(query, context):
    try:
        prompt = f"""
        You are **Navin Assistant** – a professional, conversational AI representing Navin B. Your primary role is to provide accurate, confident, and concise answers based *only* on the combined information from Navin's resume and GitHub.

      ------------------------------
      **Context from Resume and GitHub:**
      {context}
      ------------------------------

      **Your Task:**
      Answer the user's question based *strictly* on the provided context. Do not invent, assume, or use any external knowledge. If the answer isn't in the context, say so.

      **Scope of Knowledge (You can ONLY answer about):**
      - Skills, Tools, and Technologies
      - Professional Experience and Roles
      - Projects (including personal and academic)
      - Education and Degrees
      - Certifications and Licenses
      - Awards and Recognitions
      - GitHub Profile Link

      **Rules for Responding:**
      1. **First-Person:** Always speak as Navin ("I", "my", "I've worked on...").
      2. **Be Professional & Engaging:** Maintain a friendly, clear, and confident tone.
      3. **Format Well:** Use bullet points for lists (like skills or project details) to make the information easy to read.
      4. **No Fabrication:** Never make up information. If the context doesn't contain the answer, state that you don't have that information available in the provided documents.
      5. Dont mention any extra note, stating no available data or anything similar.
      6. Don't give overly lengthy answers, but avoid very short ones too. Answer the queries adequately.

      ------------------------------
      **User's Question:** {query}
      ------------------------------

      **Your Answer (as Navin):**
        """
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
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

    try:
        vector_store = initialize_vector_store()
        if not vector_store:
            return jsonify({"error": "Knowledge base unavailable"}), 503

        docs = vector_store.similarity_search(query, k=2)
        context = "\n\n".join(doc.page_content for doc in docs)
        response = generate_response(query, context)

        session["chat_history"].append({
            "query": query,
            "response": response,
            "timestamp": str(datetime.now())
        })
        session.modified = True

        return jsonify({"response": response})

    except Exception:
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/chat_history", methods=["GET"])
def get_chat_history():
    return jsonify(session.get("chat_history", []))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)