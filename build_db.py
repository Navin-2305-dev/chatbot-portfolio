import os
import logging
<<<<<<< HEAD
import requests
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain.docstore.document import Document

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
=======
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import requests
from langchain.docstore.document import Document

# Setup logging
>>>>>>> 70093230aa7dc5fa37876110637ee5bffe9d7dec
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

<<<<<<< HEAD
# -------------------------------------------------------------------
# GitHub Fetch (SAFE + TOKEN OPTIONAL)
# -------------------------------------------------------------------
def fetch_github_projects() -> str:
    try:
        headers = {"Accept": "application/vnd.github.v3+json"}

        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        response = requests.get(
            "https://api.github.com/users/Navin-2305-dev/repos",
            params={"sort": "updated", "per_page": 10},
=======
def fetch_github_projects():
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"
        }
        response = requests.get(
            "https://api.github.com/users/Navin-2305-dev/repos?sort=updated&per_page=10",
>>>>>>> 70093230aa7dc5fa37876110637ee5bffe9d7dec
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
<<<<<<< HEAD

        repos = response.json()
        if not repos:
            return "No repositories found."

        return "\n".join(
            f"{repo['name']}: {repo.get('description', 'No description')} "
            f"({repo['html_url']})"
            for repo in repos
        )

=======
        repos = response.json()
        return "\n".join(
            f"{repo['name']}: {repo.get('description', 'No description')} ({repo['html_url']})"
            for repo in repos
        )
>>>>>>> 70093230aa7dc5fa37876110637ee5bffe9d7dec
    except Exception as e:
        logger.warning(f"GitHub fetch failed: {e}")
        return "See my projects at https://github.com/Navin-2305-dev"

<<<<<<< HEAD

# -------------------------------------------------------------------
# Vector DB Builder
# -------------------------------------------------------------------
def build_vector_db():
    documents = []

    # ------------------ Load Resume ------------------
    resume_path = "Uploads/Navin - Software_resume.pdf"
    if os.path.exists(resume_path):
        loader = PyPDFLoader(resume_path)
        documents.extend(loader.load())
        logger.info("Resume loaded successfully")
    else:
        logger.warning("Resume PDF not found")

    # ------------------ Load GitHub Data ------------------
    github_content = fetch_github_projects()
    documents.append(
        Document(
            page_content=f"GitHub Projects:\n{github_content}",
            metadata={"source": "github"}
        )
    )

    if not documents:
        raise RuntimeError("No documents available for vectorization")

    # ------------------ Split Documents ------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total chunks created: {len(chunks)}")

    # ------------------ Embeddings ------------------
    embeddings = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Qdrant.from_existing_collection(
        collection_name="ChatBot-Portfolio",
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    vector_store.add_documents(chunks)


    logger.info("✅ Vector database built successfully")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    build_vector_db()
=======
def build_vector_db():
    # Load documents
    documents = []
    
    # Resume PDF
    if os.path.exists("Uploads/resume.pdf"):
        loader = PyPDFLoader("Uploads/resume.pdf")
        documents.extend(loader.load())
    
    # GitHub data - Convert to Document object
    github_content = fetch_github_projects()
    if github_content:
        documents.append(Document(
            page_content=f"GitHub Projects:\n{github_content}",
            metadata={"source": "github"}
        ))

    if not documents:
        raise ValueError("No documents found to process")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    # Upload to Qdrant
    Qdrant.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="navin_portfolio",
    )
    logger.info("Successfully built vector database")

if __name__ == "__main__":
    build_vector_db()
>>>>>>> 70093230aa7dc5fa37876110637ee5bffe9d7dec
