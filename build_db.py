import os
import logging
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

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
            headers=headers,
            timeout=10
        )
        response.raise_for_status()

        repos = response.json()
        if not repos:
            return "No repositories found."

        return "\n".join(
            f"{repo['name']}: {repo.get('description', 'No description')} "
            f"({repo['html_url']})"
            for repo in repos
        )

    except Exception as e:
        logger.warning(f"GitHub fetch failed: {e}")
        return "See my projects at https://github.com/Navin-2305-dev"


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
