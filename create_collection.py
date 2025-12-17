import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False,  # avoids cloud warning
)

COLLECTION_NAME = "ChatBot-Portfolio"

# Check if collection already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
else:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # all-MiniLM-L6-v2 embedding size
            distance=Distance.COSINE,
        ),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
