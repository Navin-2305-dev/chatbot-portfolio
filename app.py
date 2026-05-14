import os
import logging
from datetime import datetime
from typing import List, Optional

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv

from google import genai
from google.genai import types

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
    before_sleep_log,
)

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ============================================================
# Setup
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
)


# ============================================================
# Configuration
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "navin_portfolio")

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))

GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1"))

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


# ============================================================
# Environment validation
# ============================================================

def validate_environment() -> None:
    required_env = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }

    missing = [
        key
        for key, value in required_env.items()
        if not value
    ]

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


validate_environment()


# ============================================================
# Gemini client
# ============================================================

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# Retry helper
# ============================================================

def is_retryable_genai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    error_text = str(exc).upper()

    retryable_keywords = [
        "408",
        "429",
        "500",
        "502",
        "503",
        "504",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "SERVICE UNAVAILABLE",
    ]

    return any(keyword in error_text for keyword in retryable_keywords)


# ============================================================
# Gemini Embedding Wrapper
# ============================================================

class GeminiGenAIEmbeddings(Embeddings):
    """
    LangChain-compatible embedding wrapper using google-genai.

    This replaces:
    - langchain_google_genai
    - GoogleGenerativeAIEmbeddings
    - models/text-embedding-004

    It must match the embedding logic used in build_db.py.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        output_dimensionality: int = 768,
        batch_size: int = 1,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.batch_size = batch_size

    def _prepare_document(
        self,
        text: str,
        title: Optional[str] = None,
    ) -> str:
        clean_text = text.replace("\n", " ").strip()
        clean_title = title or "portfolio_document"

        return f"title: {clean_title} | text: {clean_text}"

    def _prepare_query(self, query: str) -> str:
        clean_query = query.replace("\n", " ").strip()

        return f"task: search result | query: {clean_query}"

    @retry(
        retry=retry_if_exception(is_retryable_genai_error),
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_single_batch(self, texts: List[str]) -> List[List[float]]:
        contents = [
            types.Content(
                parts=[
                    types.Part.from_text(text=text)
                ]
            )
            for text in texts
        ]

        result = self.client.models.embed_content(
            model=self.model,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality
            ),
        )

        return [embedding.values for embedding in result.embeddings]

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        total_batches = (
            len(texts) + self.batch_size - 1
        ) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_number = (i // self.batch_size) + 1
            batch = texts[i:i + self.batch_size]

            logger.info(
                f"Embedding batch {batch_number}/{total_batches} "
                f"with {len(batch)} text(s)"
            )

            batch_embeddings = self._embed_single_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prepared_texts = [
            self._prepare_document(text)
            for text in texts
        ]

        return self._embed_texts(prepared_texts)

    def embed_query(self, text: str) -> List[float]:
        prepared_query = self._prepare_query(text)

        return self._embed_texts([prepared_query])[0]


# ============================================================
# Embeddings instance
# ============================================================

embeddings = GeminiGenAIEmbeddings(
    api_key=GEMINI_API_KEY,
    model=EMBEDDING_MODEL,
    output_dimensionality=VECTOR_DIMENSION,
    batch_size=EMBEDDING_BATCH_SIZE,
)


# ============================================================
# Qdrant vector store
# ============================================================

def get_vector_store():
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )

        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )

    except Exception as e:
        logger.exception(f"Error connecting to Qdrant: {e}")
        return None


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(query: str, context: str) -> str:
    return f"""
You are Navin Assistant — a professional, conversational AI representing Navin B.

You must answer based only on the provided context from Navin's resume, GitHub, and portfolio knowledge base.

Context:
{context}

Rules:
- Strictly dont include bold words or sentences in your response.
- Speak in first person as Navin.
- Use "I", "my", and "I've" naturally.
- Be professional, friendly, and confident.
- Never invent information that is not present in the context.
- If the context does not contain the answer, say that I have not provided that detail yet.
- Keep answers concise but useful.
- Use bullet points when listing skills, projects, tools, or achievements.
- Do not mention "context", "documents", "vector database", or "retrieval" to the user.

User question:
{query}

Answer as Navin:
""".strip()


# ============================================================
# Gemini response generation
# ============================================================

@retry(
    retry=retry_if_exception(is_retryable_genai_error),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_response(query: str, context: str) -> str:
    prompt = build_prompt(query, context)

    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    answer = getattr(response, "text", None)

    if not answer:
        return "I couldn't generate a response right now."

    return answer.strip()


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "ok",
            "service": "Navin Portfolio Chatbot API",
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "collection": COLLECTION_NAME,
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL,
            "collection": COLLECTION_NAME,
        }
    )


@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    try:
        data = request.get_json()
        query = data.get("query", "").strip()

        logger.info(f"Received query: {query}")

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

    except Exception:
        return jsonify({"error": "Invalid request data"}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    vector_store = get_vector_store()

    if vector_store is None:
        return jsonify({"error": "Knowledge base unavailable"}), 503

    try:
        docs = vector_store.similarity_search(query, k=RETRIEVAL_K)

        if not docs:
            response_text = (
                "I don't have enough information about that yet. "
                "Please ask me about my skills, projects, experience, "
                "resume, or GitHub work."
            )
        else:
            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            response_text = generate_response(query, context)

        session["chat_history"].append(
            {
                "query": query,
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
            }
        )

        session.modified = True

        return jsonify(
            {
                "response": response_text,
                "sources_found": len(docs),
            }
        )

    except Exception as e:
        logger.exception(f"Retrieval or generation error: {e}")

        return jsonify(
            {
                "error": "Internal server error",
                "details": str(e),
            }
        ), 500


@app.route("/api/chat_history", methods=["GET"])
def get_chat_history():
    return jsonify(session.get("chat_history", []))


@app.route("/api/clear_chat_history", methods=["POST"])
def clear_chat_history():
    session["chat_history"] = []
    session.modified = True

    return jsonify({"message": "Chat history cleared"})


# ============================================================
# Run app
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )