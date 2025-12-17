import os
import logging
import requests
from dotenv import load_dotenv

from google.genai import Client

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
from qdrant_client import QdrantClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "ChatBot-Portfolio"

# --------------------------------------------------
# Gemini Client + Embeddings
# --------------------------------------------------
gemini_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbeddings:
    def __init__(self, model="models/text-embedding-004"):
        self.client = gemini_client
        self.model = model

    def embed_documents(self, texts):
        return [
            self.client.models.embed_content(
                model=self.model,
                content=text
            ).embedding
            for text in texts
        ]

    def embed_query(self, text):
        return self.client.models.embed_content(
            model=self.model,
            content=text
        ).embedding

def fetch_github_projects() -> str:
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
        logger.warning(f"GitHub fetch failed: {e}")
        return "See my GitHub: https://github.com/Navin-2305-dev"

def build_vector_db():
    documents = []

    resume_path = "Uploads/Navin - Software_resume.pdf"
    if os.path.exists(resume_path):
        documents.extend(PyPDFLoader(resume_path).load())
        logger.info("Resume loaded")

    documents.append(
        Document(
            page_content=f"GitHub Projects:\n{fetch_github_projects()}",
            metadata={"source": "github"}
        )
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Chunks created: {len(chunks)}")

    embeddings = GeminiEmbeddings()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vector_store.add_documents(chunks)
    logger.info("✅ Vector database built successfully")

if __name__ == "__main__":
    build_vector_db()
