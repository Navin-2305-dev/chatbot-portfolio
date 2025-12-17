import os
import logging
import requests
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "navin_portfolio"
VECTOR_DIMENSION = 768  # Default for models/text-embedding-004

# Explicit API key to avoid any auth issues
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_document",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def fetch_github_projects():
    try:
        response = requests.get(
            "https://api.github.com/users/Navin-2305-dev/repos",
            params={"sort": "updated", "per_page": 10},
            timeout=10
        )
        response.raise_for_status()
        repos = response.json()
        return "\n".join(
            f"{r['name']}: {r.get('description', 'No description')} ({r['html_url']})"
            for r in repos
        )
    except Exception as e:
        logger.warning(f"GitHub fetch error: {e}")
        return "See my GitHub: https://github.com/Navin-2305-dev"

def build_vector_db():
    documents = []

    resume_path = "Uploads/Navin - Software_resume.pdf"
    if os.path.exists(resume_path):
        documents.extend(PyPDFLoader(resume_path).load())
        logger.info("Resume PDF loaded")
    else:
        logger.error("Resume PDF not found at Uploads/Navin - Software_resume.pdf")

    documents.append(
        Document(
            page_content=f"GitHub Projects:\n{fetch_github_projects()}",
            metadata={"source": "github"}
        )
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=20
    )

    # Handle dimension mismatch automatically
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        current_dim = info.config.params.vectors.size
        if current_dim != VECTOR_DIMENSION:
            logger.info(f"Dimension mismatch detected ({current_dim} → {VECTOR_DIMENSION}). Deleting old collection...")
            client.delete_collection(COLLECTION_NAME)
        else:
            logger.info("Collection exists with correct dimension.")

    # Create collection if needed
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
        )
        logger.info("Created new collection with 768 dimensions")

    # Add documents
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_documents(chunks)
    logger.info("✅ Vector database rebuilt and populated successfully!")

if __name__ == "__main__":
    build_vector_db()