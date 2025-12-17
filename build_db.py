import os
import logging
import requests
from dotenv import load_dotenv

from google.genai import Client

from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document
from qdrant_client import QdrantClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "ChatBot-Portfolio"

# --------------------------------------------------
# Gemini Embeddings
# --------------------------------------------------
gemini_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiEmbeddings(Embeddings):
    def __init__(self, model="models/text-embedding-004"):
        self.client = gemini_client
        self.model = model

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            res = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            vectors.append(res.embeddings[0].values)
        return vectors

    def embed_query(self, text):
        res = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return res.embeddings[0].values

# --------------------------------------------------
# Helpers
# --------------------------------------------------
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
        logger.warning(e)
        return "See my GitHub: https://github.com/Navin-2305-dev"

# --------------------------------------------------
# Build Vector DB
# --------------------------------------------------
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

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=GeminiEmbeddings()
    )

    vector_store.add_documents(chunks)
    logger.info("✅ Vector database built successfully")

if __name__ == "__main__":
    build_vector_db()
