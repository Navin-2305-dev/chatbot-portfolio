import os
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from dotenv import load_dotenv
import psutil
from tenacity import retry, stop_after_attempt, wait_fixed
import chromadb

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a-secure-default-secret-key")
CORS(app, resources={r"/api/*": {"origins": "*"}})
logger.info("Flask app initialized with CORS enabled")

# API and Configuration
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.critical("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY is required.")
genai.configure(api_key=gemini_api_key)
logger.info("Gemini API configured successfully")

# Constants
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
CHROMA_PERSIST_DIR = "./chroma_db"

# Global Variables
vector_db = None
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu", "trust_remote_code": False},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector Store Initialization
def initialize_vector_store():
    global vector_db
    try:
        if not os.path.exists(CHROMA_PERSIST_DIR):
            logger.error(f"Chroma database directory {CHROMA_PERSIST_DIR} does not exist.")
            return False

        logger.info(f"Loading pre-built Chroma vector store from {CHROMA_PERSIST_DIR}...")
        vector_db = Chroma(
            collection_name="navin_portfolio",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        logger.info("Pre-built vector store loaded successfully.")
        
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage after loading vector store: {memory_usage:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"Error loading pre-built vector store: {e}", exc_info=True)
        return False

# Generative AI Response Function
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
        2. **Stay in Scope:** If the question is about personal opinions, future plans not in the resume, or anything outside the defined scope, politely decline. Use this response:
            "I can only provide information based on my resume and GitHub profile. My expertise covers my skills, projects, and professional experience. Do you have a question about one of those areas?"
        3. **Be Professional & Engaging:** Maintain a friendly, clear, and confident tone.
        4. **Format Well:** Use bullet points for lists (like skills or project details) to make the information easy to read.
        5. **No Fabrication:** Never make up information. If the context doesn't contain the answer, state that you don't have that information available in the provided documents.
        6. Dont mention any extra note, stating no available data or anything similar.

        ------------------------------
        **User's Question:** {query}
        ------------------------------

        **Your Answer (as Navin):**
        """
        logger.debug(f"Generating response for query: '{query}'")
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        if not response.text:
            logger.warning("Received an empty response from the Gemini API.")
            return "I'm sorry, I encountered an issue while processing your request. Could you please rephrase or try again?"
        
        logger.info("Successfully generated response from Gemini.")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating response from Gemini: {e}", exc_info=True)
        return "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."

# Flask Routes
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    logger.debug("Received new request to /api/chatbot")
    
    if request.content_type != "application/json":
        logger.warning(f"Invalid content type received: {request.content_type}")
        return jsonify({"response": "Invalid request: Content-Type must be application/json."}), 415

    try:
        data = request.get_json()
        query = data.get("query", "").strip()
    except Exception as e:
        logger.error(f"Failed to parse JSON request body: {e}")
        return jsonify({"response": "Invalid request: Malformed JSON."}), 400

    if not query:
        logger.warning("Request received with an empty query.")
        return jsonify({"response": "Invalid request: Query cannot be empty."}), 400

    logger.info(f"Processing query: '{query}'")

    if "chat_history" not in session:
        session["chat_history"] = []
        logger.debug("Initialized new session chat history.")

    if vector_db is None:
        logger.critical("Vector store is not initialized. Cannot process query.")
        return jsonify({"response": "I'm sorry, my knowledge base is currently unavailable. Please try again later."}), 503

    # Retrieve context from vector store
    logger.debug("Performing similarity search in vector store.")
    try:
        results = vector_db.similarity_search(query, k=2)  # Reduced k for lower memory usage
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        logger.debug(f"Retrieved {len(results)} documents for context.")
    except Exception as e:
        logger.error(f"Error during similarity search: {e}", exc_info=True)
        return jsonify({"response": "Error retrieving information from the knowledge base."}), 500

    # Generate the final response
    response_text = generate_response(query, context)

    # Update chat history
    session["chat_history"].append({"query": query, "response": response_text})
    session.modified = True
    logger.debug(f"Chat history updated. Total entries: {len(session['chat_history'])}")

    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage after processing query: {memory_usage:.2f} MB")

    return jsonify({"response": response_text})

@app.route("/api/health", methods=["GET"])
def health_check():
    status = {
        "status": "ok",
        "vector_store_initialized": vector_db is not None,
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
    }
    return jsonify(status)

# Initialize vector store
if not initialize_vector_store():
    logger.critical("Failed to initialize the vector store on startup.")
    exit(1)

# Main Execution
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)