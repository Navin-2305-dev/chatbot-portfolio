import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import requests
from dotenv import load_dotenv
import chromadb

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
UPLOAD_FOLDER = 'Uploads'
RESUME_PATH = os.path.join(UPLOAD_FOLDER, 'resume.pdf')
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
GITHUB_API_URL = "https://api.github.com/users/Navin-2305-dev/repos?sort=updated&per_page=10"
GITHUB_PROFILE_URL = "https://github.com/Navin-2305-dev"
CHROMA_PERSIST_DIR = "./chroma_db"

# Fetch GitHub data
def fetch_github_data():
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "NavinBot",
            "Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"
        }
        response = requests.get(GITHUB_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        repos = response.json()
        if not repos:
            logger.warning("No public repositories found.")
            return {
                "section": "GitHub Profile",
                "content": f"GitHub Profile: For a full list of projects, visit {GITHUB_PROFILE_URL}."
            }
        repo_data = [
            f"{repo['name']}: {repo['description'] or 'No description'} (URL: {repo['html_url']})"
            for repo in repos
        ]
        content = f"GitHub Profile: Navin B's recent projects include {', '.join(repo_data)}."
        logger.info("Successfully fetched GitHub repository data via API.")
        return {"section": "GitHub Profile", "content": content}
    except Exception as e:
        logger.error(f"Error fetching GitHub data: {e}")
        return {
            "section": "GitHub Profile",
            "content": f"GitHub Profile: Explore Navin B's projects at {GITHUB_PROFILE_URL}."
        }

# Build the Chroma database
def build_chroma_db():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True}
        )
        all_documents = []

        # Load resume
        if os.path.exists(RESUME_PATH):
            logger.info(f"Loading resume from {RESUME_PATH}")
            loader = PyPDFLoader(RESUME_PATH)
            all_documents.extend(loader.load())
        else:
            logger.warning(f"Resume not found at {RESUME_PATH}. Skipping.")

        # Fetch GitHub data
        github_data = fetch_github_data()
        if github_data and github_data['content']:
            all_documents.append(Document(page_content=github_data['content'], metadata={"source": "github"}))
        else:
            logger.warning("No content obtained from GitHub.")

        if not all_documents:
            logger.error("No documents found. Cannot build vector store.")
            return False

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=80)
        texts = text_splitter.split_documents(all_documents)
        logger.info(f"Split content into {len(texts)} chunks.")

        # Create Chroma database
        logger.info(f"Generating embeddings and saving Chroma DB to {CHROMA_PERSIST_DIR}...")
        vector_db = Chroma.from_documents(
            texts,
            embeddings,
            collection_name="navin_portfolio",
            persist_directory=CHROMA_PERSIST_DIR,
            client_settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        logger.info("Chroma database built and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error building Chroma database: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not build_chroma_db():
        logger.critical("Failed to build Chroma database.")
        exit(1)
    else:
        logger.info(f"Chroma database saved to {CHROMA_PERSIST_DIR}. Ready for deployment.")