import os
import logging
import requests
from typing import List, Optional

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
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


# ============================================================
# Setup
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "navin_portfolio")

RESUME_PATH = os.getenv(
    "RESUME_PATH",
    "Uploads/Navin - Software_resume.pdf"
)

GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "Navin-2305-dev")

# Stable long-term Gemini embedding model
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

# 768 is recommended and cost-efficient for your portfolio chatbot
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Keep batch size small to avoid Gemini temporary 503/capacity issues
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1"))

# For your build script, recreate collection every time to avoid duplicate chunks
RECREATE_COLLECTION_ON_BUILD = (
    os.getenv("RECREATE_COLLECTION_ON_BUILD", "true").lower() == "true"
)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


# ============================================================
# Environment validation
# ============================================================

def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}"
        )

    return value


def validate_environment() -> None:
    require_env("GEMINI_API_KEY")
    require_env("QDRANT_URL")
    require_env("QDRANT_API_KEY")


# ============================================================
# Retry handling for Gemini temporary failures
# ============================================================

def is_retryable_genai_error(exc: Exception) -> bool:
    """
    Retries temporary Gemini/API failures.

    Examples:
    - 503 UNAVAILABLE
    - 429 rate limit
    - 500/502/504 server errors
    """

    status_code = getattr(exc, "status_code", None)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    error_text = str(exc).upper()

    retryable_keywords = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "502",
        "504",
        "DEADLINE_EXCEEDED",
        "SERVICE UNAVAILABLE",
    ]

    return any(keyword in error_text for keyword in retryable_keywords)


# ============================================================
# Gemini Embedding Wrapper
# ============================================================

class GeminiGenAIEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper using the new google-genai SDK.

    This avoids:
    - langchain_google_genai
    - deprecated google.generativeai
    - models/text-embedding-004

    Uses:
    - gemini-embedding-2
    - output_dimensionality=768 by default
    - retry handling for 503 / 429 / 5xx errors
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
        """
        Document formatting for retrieval.
        Gemini embedding 2 uses instruction-style text formatting.
        """

        clean_text = text.replace("\n", " ").strip()
        clean_title = title or "portfolio_document"

        return f"title: {clean_title} | text: {clean_text}"

    def _prepare_query(self, query: str) -> str:
        """
        Query formatting for retrieval/search.
        """

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

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

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
# Data loading
# ============================================================

def fetch_github_projects() -> str:
    """
    Fetch latest public GitHub repositories for portfolio chatbot context.
    """

    try:
        response = requests.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}/repos",
            params={
                "sort": "updated",
                "per_page": 15,
            },
            timeout=10,
        )

        response.raise_for_status()

        repos = response.json()
        
        print(repos)

        if not repos:
            return (
                f"No public repositories found for GitHub user: "
                f"{GITHUB_USERNAME}"
            )

        project_blocks = []

        for repo in repos:
            name = repo.get("name", "Unnamed Repository")
            description = repo.get("description") or "No description provided"
            html_url = repo.get("html_url", "")
            language = repo.get("language") or "Not specified"
            stars = repo.get("stargazers_count", 0)
            updated_at = repo.get("updated_at", "Unknown")

            project_blocks.append(
                f"Project Name: {name}\n"
                f"Description: {description}\n"
                f"Primary Language: {language}\n"
                f"Stars: {stars}\n"
                f"Last Updated: {updated_at}\n"
                f"URL: {html_url}"
            )
        print(project_blocks)

        return "\n\n".join(project_blocks)

    except Exception as e:
        logger.warning(f"GitHub fetch error: {e}")

        return f"See my GitHub profile: https://github.com/{GITHUB_USERNAME}"


def load_documents() -> List[Document]:
    documents: List[Document] = []

    if os.path.exists(RESUME_PATH):
        resume_docs = PyPDFLoader(RESUME_PATH).load()

        for doc in resume_docs:
            doc.metadata["source"] = "resume"
            doc.metadata["file_path"] = RESUME_PATH

        documents.extend(resume_docs)

        logger.info("Resume PDF loaded")

    else:
        logger.warning(f"Resume PDF not found at: {RESUME_PATH}")

    github_projects = fetch_github_projects()

    documents.append(
        Document(
            page_content=f"GitHub Projects:\n\n{github_projects}",
            metadata={
                "source": "github",
                "github_username": GITHUB_USERNAME,
            },
        )
    )

    if not documents:
        raise ValueError("No documents found to embed.")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced 0 chunks.")

    logger.info(f"Split into {len(chunks)} chunks")

    return chunks


# ============================================================
# Qdrant helpers
# ============================================================

def get_existing_vector_dimension(
    client: QdrantClient,
    collection_name: str,
) -> Optional[int]:
    """
    Reads existing vector dimension from Qdrant collection.
    Supports normal single-vector and named-vector configs.
    """

    try:
        info = client.get_collection(collection_name)
        vectors_config = info.config.params.vectors

        # Normal single-vector collection
        if hasattr(vectors_config, "size"):
            return vectors_config.size

        # Named-vector collection
        if isinstance(vectors_config, dict):
            first_vector = next(iter(vectors_config.values()))
            return first_vector.size

        return None

    except Exception as e:
        logger.warning(f"Could not read existing collection config: {e}")
        return None


def ensure_qdrant_collection(client: QdrantClient) -> None:
    """
    Creates or recreates the Qdrant collection.

    For this portfolio build script, recreating is best because:
    - It prevents duplicate chunks on rerun.
    - It keeps resume/GitHub content fresh.
    - It avoids incompatible vectors if model/dimension changes.
    """

    exists = client.collection_exists(COLLECTION_NAME)

    if exists and RECREATE_COLLECTION_ON_BUILD:
        logger.info(
            f"Recreating collection '{COLLECTION_NAME}' "
            f"to avoid duplicate chunks..."
        )

        client.delete_collection(COLLECTION_NAME)
        exists = False

    if exists:
        current_dim = get_existing_vector_dimension(client, COLLECTION_NAME)

        if current_dim != VECTOR_DIMENSION:
            logger.info(
                f"Dimension mismatch detected: existing={current_dim}, "
                f"required={VECTOR_DIMENSION}. Deleting old collection..."
            )

            client.delete_collection(COLLECTION_NAME)
            exists = False

        else:
            logger.info(
                f"Collection '{COLLECTION_NAME}' already exists "
                f"with correct dimension: {current_dim}"
            )

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            f"Created collection '{COLLECTION_NAME}' "
            f"with {VECTOR_DIMENSION} dimensions"
        )


# ============================================================
# Main build function
# ============================================================

def build_vector_db() -> None:
    validate_environment()

    embeddings = GeminiGenAIEmbeddings(
        api_key=require_env("GEMINI_API_KEY"),
        model=EMBEDDING_MODEL,
        output_dimensionality=VECTOR_DIMENSION,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    documents = load_documents()
    chunks = split_documents(documents)

    qdrant_client = QdrantClient(
        url=require_env("QDRANT_URL"),
        api_key=require_env("QDRANT_API_KEY"),
        timeout=30,
    )

    ensure_qdrant_collection(qdrant_client)

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    vector_store.add_documents(chunks)

    logger.info("✅ Vector database rebuilt and populated successfully!")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Vector dimension: {VECTOR_DIMENSION}")
    logger.info(f"Total chunks inserted: {len(chunks)}")


if __name__ == "__main__":
    build_vector_db()