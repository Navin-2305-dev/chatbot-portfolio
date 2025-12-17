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

COLLECTION_NAME = "navin_portfolio"  # Consistent with app.py

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_document"  # Optimized for documents
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
        logger.info("Resume loaded")
    else:
        logger.error("Resume PDF not found!")

    documents.append(
        Document(
            page_content=f"GitHub Projects:\n{fetch_github_projects()}",
            metadata={"source": "github"}
        )
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=20
    )

    # Create collection if it doesn't exist (768 dim for text-embedding-004)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        logger.info("Collection created")
    else:
        logger.info("Collection already exists")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)
    logger.info("✅ Vector database built successfully")

if __name__ == "__main__":
    build_vector_db()