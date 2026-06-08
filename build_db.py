"""
build_db.py
───────────
Builds (or rebuilds) the Qdrant vector database for the portfolio chatbot.

Data sources:
  1. Resume PDF     — via PyPDFLoader
  2. LinkedIn       — via linkedin_data.py (structured JSON)
  3. GitHub         — via github_data.py   (repos + READMEs + topics)

Run:
    python build_db.py

Environment variables (all in .env):
    GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
    QDRANT_COLLECTION_NAME, RESUME_PATH
    GITHUB_USERNAME, GITHUB_TOKEN  (optional, raises rate limit)
    LINKEDIN_PROFILE_PATH
    GEMINI_EMBEDDING_MODEL, VECTOR_DIMENSION
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_BATCH_SIZE
    RECREATE_COLLECTION_ON_BUILD   (default: true)
"""

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Local shared modules
from embeddings import GeminiGenAIEmbeddings
from github_data import fetch_github_documents
from linkedin_data import linkedin_profile_to_documents

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "navin_portfolio")
RESUME_PATH = os.getenv("RESUME_PATH", "Uploads/Navin - Software_resume.pdf")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1"))
RECREATE = os.getenv("RECREATE_COLLECTION_ON_BUILD", "true").lower() == "true"


# ── Environment validation ────────────────────────────────────────────────────

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def validate_environment() -> None:
    require_env("GEMINI_API_KEY")
    require_env("QDRANT_URL")
    require_env("QDRANT_API_KEY")


# ── Document loading ──────────────────────────────────────────────────────────

def load_resume() -> List[Document]:
    """Load and tag resume PDF pages."""
    if not os.path.exists(RESUME_PATH):
        logger.warning(f"Resume PDF not found at: {RESUME_PATH}")
        return []

    pages = PyPDFLoader(RESUME_PATH).load()
    for doc in pages:
        doc.metadata["source"] = "resume"
        doc.metadata["file_path"] = RESUME_PATH

    logger.info(f"Resume: {len(pages)} pages loaded from '{RESUME_PATH}'")
    return pages


def load_all_documents() -> List[Document]:
    """Aggregate documents from all sources."""
    documents: List[Document] = []

    # 1. Resume
    documents.extend(load_resume())

    # 2. LinkedIn
    try:
        linkedin_docs = linkedin_profile_to_documents()
        documents.extend(linkedin_docs)
        logger.info(f"LinkedIn: {len(linkedin_docs)} documents added")
    except Exception as e:
        logger.error(f"LinkedIn loading failed: {e}")

    # 3. GitHub
    try:
        github_docs = fetch_github_documents()
        documents.extend(github_docs)
        logger.info(f"GitHub: {len(github_docs)} documents added")
    except Exception as e:
        logger.error(f"GitHub loading failed: {e}")

    if not documents:
        raise ValueError("No documents found to embed. Check your data sources.")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split long documents into chunks.

    Short documents (LinkedIn sections, GitHub overviews) that are already
    well under CHUNK_SIZE are returned as-is — no unnecessary splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced 0 chunks.")

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _get_existing_dimension(client: QdrantClient) -> Optional[int]:
    try:
        info = client.get_collection(COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "size"):
            return vectors_config.size
        if isinstance(vectors_config, dict):
            return next(iter(vectors_config.values())).size
    except Exception as e:
        logger.warning(f"Could not read collection config: {e}")
    return None


def ensure_collection(client: QdrantClient) -> None:
    """Create or recreate the Qdrant collection."""
    exists = client.collection_exists(COLLECTION_NAME)

    if exists and RECREATE:
        logger.info(f"Recreating '{COLLECTION_NAME}' (RECREATE_COLLECTION_ON_BUILD=true)…")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if exists:
        current_dim = _get_existing_dimension(client)
        if current_dim != VECTOR_DIMENSION:
            logger.info(
                f"Dimension mismatch (existing={current_dim}, required={VECTOR_DIMENSION}). "
                "Deleting collection…"
            )
            client.delete_collection(COLLECTION_NAME)
            exists = False
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' exists with correct dimension {current_dim}")

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{COLLECTION_NAME}' ({VECTOR_DIMENSION}d, cosine)")


# ── Entry point ───────────────────────────────────────────────────────────────

def build_vector_db() -> None:
    validate_environment()

    embeddings = GeminiGenAIEmbeddings(
        api_key=require_env("GEMINI_API_KEY"),
        model=EMBEDDING_MODEL,
        output_dimensionality=VECTOR_DIMENSION,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    documents = load_all_documents()
    chunks = split_documents(documents)

    qdrant_client = QdrantClient(
        url=require_env("QDRANT_URL"),
        api_key=require_env("QDRANT_API_KEY"),
        timeout=30,
    )
    ensure_collection(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    logger.info(f"Inserting {len(chunks)} chunks into Qdrant…")
    vector_store.add_documents(chunks)

    logger.info("=" * 60)
    logger.info("✅  Vector database built successfully!")
    logger.info(f"   Collection : {COLLECTION_NAME}")
    logger.info(f"   Model      : {EMBEDDING_MODEL}")
    logger.info(f"   Dimensions : {VECTOR_DIMENSION}")
    logger.info(f"   Chunks     : {len(chunks)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_vector_db()