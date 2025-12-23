from qdrant_client import QdrantClient
from qdrant_client.http import models
from core.models import bge_model
import os

COLLECTION_NAME = "rag_documents"

class VectorDB:
    def __init__(self, path: str = "backend/data/qdrant_db"):
        self.client = QdrantClient(path=path)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384, # bge-small size
                    distance=models.Distance.COSINE
                )
            )

    def add_documents(self, documents: list[str], ids: list[int], metadatas: list[dict] = None):
        points = []
        for i, text in enumerate(documents):
            vector = bge_model.encode(text)
            points.append(models.PointStruct(
                id=ids[i],
                vector=vector,
                payload={"text": text, **(metadatas[i] if metadatas else {})}
            ))
        
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def search(self, query: str, limit: int = 5):
        query_vector = bge_model.encode_query(query)
        # Fallback to recommended search method if 'search' is missing on this client version/mode
        try:
             results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )
        except AttributeError:
             # Try query_points (newer API)
             results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
             ).points
             
        return [r.payload for r in results]

db = VectorDB()
