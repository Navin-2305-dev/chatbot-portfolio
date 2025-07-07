import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import requests
from langchain.docstore.document import Document  # Import Document class

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def fetch_github_projects():
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"
        }
        response = requests.get(
            "https://api.github.com/users/Navin-2305-dev/repos?sort=updated&per_page=10",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        repos = response.json()
        return "\n".join(
            f"{repo['name']}: {repo.get('description', 'No description')} ({repo['html_url']})"
            for repo in repos
        )
    except Exception as e:
        logger.warning(f"GitHub fetch failed: {e}")
        return "See my projects at https://github.com/Navin-2305-dev"

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