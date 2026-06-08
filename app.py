import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from embeddings import GeminiGenAIEmbeddings, is_retryable_genai_error

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    [o.strip() for o in _raw_origins.split(",")]
    if _raw_origins != "*"
    else "*"
)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})


# ── Configuration ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "navin_portfolio")

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1"))

GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# How many past turns the LLM sees for context (each turn = 1 user + 1 assistant)
HISTORY_WINDOW = int(os.getenv("CONVERSATION_HISTORY_WINDOW", "6"))

# Max sessions to hold in RAM (oldest evicted first to prevent memory leak)
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))

# Cookie name for the session ID
SESSION_COOKIE = "portfolio_session_id"


# ── Environment validation ────────────────────────────────────────────────────

def _validate_environment() -> None:
    missing = [
        name for name in ("GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


_validate_environment()


# ── Gemini client ─────────────────────────────────────────────────────────────

genai_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Embedding instance (shared with build_db at module level) ─────────────────

embeddings = GeminiGenAIEmbeddings(
    api_key=GEMINI_API_KEY,
    model=EMBEDDING_MODEL,
    output_dimensionality=VECTOR_DIMENSION,
    batch_size=EMBEDDING_BATCH_SIZE,
)


# ── Qdrant vector store factory ───────────────────────────────────────────────

def _get_vector_store() -> Optional[QdrantVectorStore]:
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        return QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
    except Exception as e:
        logger.exception(f"Qdrant connection error: {e}")
        return None


# ── Server-side session store ─────────────────────────────────────────────────
# Simple in-memory dict: {session_id: {"history": [...], "created_at": ..., "last_active": ...}}
# For multi-worker deployments, swap this for Redis (same interface).

_sessions: Dict[str, dict] = {}


def _evict_old_sessions() -> None:
    """Remove oldest sessions when cap is reached to prevent unbounded growth."""
    if len(_sessions) >= MAX_SESSIONS:
        oldest = sorted(_sessions.items(), key=lambda kv: kv[1]["last_active"])
        to_remove = oldest[: len(_sessions) - MAX_SESSIONS + 1]
        for sid, _ in to_remove:
            del _sessions[sid]
            logger.info(f"Session evicted (capacity): {sid}")


def _get_or_create_session(session_id: Optional[str]) -> tuple[str, dict]:
    """Return (session_id, session_dict) — creating a new session if needed."""
    if session_id and session_id in _sessions:
        _sessions[session_id]["last_active"] = datetime.utcnow().isoformat()
        return session_id, _sessions[session_id]

    # New session
    _evict_old_sessions()
    new_id = str(uuid.uuid4())
    _sessions[new_id] = {
        "history": [],
        "created_at": datetime.utcnow().isoformat(),
        "last_active": datetime.utcnow().isoformat(),
    }
    logger.info(f"New session created: {new_id}")
    return new_id, _sessions[new_id]


def _read_session_id_from_request() -> Optional[str]:
    """Read session ID from cookie or X-Session-ID header."""
    return (
        request.cookies.get(SESSION_COOKIE)
        or request.headers.get("X-Session-ID")
    )


# ── Prompt builder with conversation history ──────────────────────────────────

def _build_prompt(query: str, context: str, history: List[dict]) -> str:
    """
    Builds the full prompt fed to Gemini.

    Structure:
      1. System persona & rules
      2. Retrieved knowledge base context
      3. Recent conversation history (last HISTORY_WINDOW turns)
      4. Current user question
    """
    system_block = """You are Navin Assistant — a professional, friendly AI representing Navin B on his portfolio website.

Rules:
- Answer ONLY from the provided context. Never invent facts.
- Speak in first person as Navin: use "I", "my", "I've".
- Do not use bold, markdown headers, or heavy formatting in your replies.
- Use bullet points only when listing multiple skills, tools, or projects.
- If the context does not contain the answer, say: "I haven't shared that detail yet — feel free to reach out to me directly."
- Never mention "context", "vector database", "retrieval", "documents", or any internal system details.
- Be warm, professional, and concise.
- Remember what the user said earlier in this conversation and refer back to it naturally when relevant."""

    context_block = f"Knowledge Base Context:\n{context}"

    # Build history block (last HISTORY_WINDOW turns)
    history_block = ""
    if history:
        recent = history[-HISTORY_WINDOW:]
        lines = ["Conversation so far:"]
        for turn in recent:
            lines.append(f"User: {turn['query']}")
            lines.append(f"Navin: {turn['response']}")
        history_block = "\n".join(lines)

    question_block = f"User's current question:\n{query}\n\nAnswer as Navin:"

    parts = [system_block, context_block]
    if history_block:
        parts.append(history_block)
    parts.append(question_block)

    return "\n\n".join(parts)


# ── Gemini generation with retry ──────────────────────────────────────────────

@retry(
    retry=retry_if_exception(is_retryable_genai_error),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _generate_response(query: str, context: str, history: List[dict]) -> str:
    prompt = _build_prompt(query, context, history)
    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    answer = getattr(response, "text", None)
    return answer.strip() if answer else "I couldn't generate a response right now. Please try again."


# ── Helper: set session cookie on response ────────────────────────────────────

def _attach_session_cookie(response, session_id: str):
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="Lax",
        max_age=60 * 60 * 24 * 7,  # 1 week
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "service": "Navin Portfolio Chatbot API",
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "collection": COLLECTION_NAME,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "active_sessions": len(_sessions),
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GENERATION_MODEL,
        "collection": COLLECTION_NAME,
    })


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request body (JSON):
        { "query": "Tell me about your projects" }

    Session is tracked via cookie (portfolio_session_id) or
    X-Session-ID header. A new session is created automatically
    on first contact and the ID is returned in the response cookie
    and response body.

    Response (JSON):
        {
            "response": "...",
            "session_id": "...",
            "sources_found": 4,
            "turn": 3
        }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    try:
        data = request.get_json()
        query = (data.get("query") or "").strip()
    except Exception:
        return jsonify({"error": "Invalid request body"}), 400

    if not query:
        return jsonify({"error": "query cannot be empty"}), 400

    # Resolve session
    raw_sid = _read_session_id_from_request()
    session_id, session = _get_or_create_session(raw_sid)

    # Retrieve context from vector DB
    vector_store = _get_vector_store()
    if vector_store is None:
        return jsonify({"error": "Knowledge base unavailable. Please try again later."}), 503

    try:
        docs = vector_store.similarity_search(query, k=RETRIEVAL_K)
    except Exception as e:
        logger.exception(f"Retrieval error: {e}")
        return jsonify({"error": "Retrieval failed", "details": str(e)}), 500

    # Build context string with source attribution for quality
    if docs:
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section", "")
            label = f"[{source}" + (f"/{section}" if section else "") + "]"
            context_parts.append(f"{label}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No specific information found in the knowledge base for this query."

    # Generate response with history context
    try:
        response_text = _generate_response(query, context, session["history"])
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        return jsonify({"error": "Generation failed", "details": str(e)}), 500

    # Persist turn to session history
    turn = {
        "query": query,
        "response": response_text,
        "timestamp": datetime.utcnow().isoformat(),
        "sources": len(docs),
    }
    session["history"].append(turn)
    session["last_active"] = datetime.utcnow().isoformat()

    resp = jsonify({
        "response": response_text,
        "session_id": session_id,
        "sources_found": len(docs),
        "turn": len(session["history"]),
    })
    return _attach_session_cookie(resp, session_id)


# ── Session management ────────────────────────────────────────────────────────

@app.route("/api/session/new", methods=["POST"])
def new_session():
    """Force-create a new session (useful for 'Start over' button in UI)."""
    _evict_old_sessions()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "history": [],
        "created_at": datetime.utcnow().isoformat(),
        "last_active": datetime.utcnow().isoformat(),
    }
    resp = jsonify({"session_id": session_id, "message": "New session started"})
    return _attach_session_cookie(resp, session_id)


@app.route("/api/session/history", methods=["GET"])
def session_history():
    """Return the conversation history for the current session."""
    raw_sid = _read_session_id_from_request()
    if not raw_sid or raw_sid not in _sessions:
        return jsonify({"history": [], "session_id": None})

    session = _sessions[raw_sid]
    return jsonify({
        "session_id": raw_sid,
        "history": session["history"],
        "turn_count": len(session["history"]),
        "created_at": session["created_at"],
        "last_active": session["last_active"],
    })


@app.route("/api/session/delete", methods=["POST"])
def delete_session():
    """Delete session data and clear the cookie."""
    raw_sid = _read_session_id_from_request()
    deleted = False
    if raw_sid and raw_sid in _sessions:
        del _sessions[raw_sid]
        deleted = True
        logger.info(f"Session deleted: {raw_sid}")

    resp = jsonify({"deleted": deleted, "message": "Session cleared"})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.route("/api/session/summary", methods=["GET"])
def session_summary():
    """Admin-only: total active sessions count (no PII)."""
    return jsonify({"active_sessions": len(_sessions)})


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)