from qdrant_client import QdrantClient
from qdrant_client.http import models
from core.models import bge_model
import os

COLLECTION_NAME = "rag_documents"
LEARNED_COLLECTION_NAME = "rag_learned_qa"

class VectorDB:
    def __init__(self, path: str = "backend/data/qdrant_db"):
        self.client = QdrantClient(path=path)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384, # bge-small size
                    distance=models.Distance.COSINE
                )
            )
            
        if LEARNED_COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=LEARNED_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384, # bge-small size
                    distance=models.Distance.COSINE
                )
            )

    def clear_all(self):
        """Delete all documents from the main collection ONLY (preserves learned QA)"""
        # Only clear the main document collection, keep learned answers
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except:
            pass
        # Recreate the main collection (learned collection stays intact)
        self._ensure_collection()

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

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.5):
        query_vector = bge_model.encode_query(query)
        # fallback to recommended search method if 'search' is missing on this client version/mode
        try:
             results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold  # Only return relevant results
            )
        except AttributeError:
             # Try query_points (newer API)
             results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold
            ).points
             
        # Return payload with score for transparency
        return [{"score": r.score, **r.payload} for r in results]

    def store_learned_answer(self, query: str, answer: str):
        # Embed the QUERY, but store the ANSWER
        vector = bge_model.encode_query(query)
        import uuid
        point_id = str(uuid.uuid4())
        
        self.client.upsert(
            collection_name=LEARNED_COLLECTION_NAME,
            points=[models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"query": query, "answer": answer}
            )]
        )

    def search_learned(self, query: str, score_threshold: float = 0.75):
        query_vector = bge_model.encode_query(query)
        try:
             results = self.client.search(
                collection_name=LEARNED_COLLECTION_NAME,
                query_vector=query_vector,
                limit=1,
                score_threshold=score_threshold
            )
        except AttributeError:
             results = self.client.query_points(
                collection_name=LEARNED_COLLECTION_NAME,
                query=query_vector,
                limit=1,
                score_threshold=score_threshold
            ).points
        
        # Debug logging
        if results:
            print(f"✅ Found learned answer for '{query}' with score: {results[0].score}")
        else:
            print(f"❌ No learned answer found for '{query}' (threshold: {score_threshold})")
            
        return [r.payload for r in results]

    def clear_learned(self):
        """Delete all learned Q&A pairs"""
        try:
            self.client.delete_collection(LEARNED_COLLECTION_NAME)
        except:
            pass
        # Recreate the learned collection
        self._ensure_collection()

db = VectorDB()
