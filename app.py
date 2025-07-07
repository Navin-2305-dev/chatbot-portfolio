import os
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import psutil
from tenacity import retry, stop_after_attempt, wait_fixed
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# API Configuration
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.critical("GEMINI_API_KEY not found")
    raise ValueError("Gemini API key is required")
genai.configure(api_key=gemini_api_key)

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

def get_vector_db():
    try:
        return Qdrant(
            client=QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=20
            ),
            collection_name="navin_portfolio",
            embeddings=embeddings
        )
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
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

        ------------------------------
        **User's Question:** {query}
        ------------------------------

        **Your Answer (as Navin):**
        """
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "I'm having trouble processing your request."

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    # Request validation
    if request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 415
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
    except Exception as e:
        return jsonify({"error": "Invalid JSON"}), 400

    # Initialize session
    if "chat_history" not in session:
        session["chat_history"] = []

    # Process query
    try:
        vector_db = get_vector_db()
        if not vector_db:
            return jsonify({"error": "Knowledge base unavailable"}), 503

        results = vector_db.similarity_search(query, k=2)
        context = "\n\n".join(doc.page_content for doc in results)
        response = generate_response(query, context)

        # Update session
        session["chat_history"].append({"query": query, "response": response})
        session.modified = True

        return jsonify({
            "response": response,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        })

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "vector_store_initialized": get_vector_db() is not None,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)