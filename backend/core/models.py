from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def encode(self, text: str):
        #  for query 
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def encode_query(self, query: str):
        # specific prompt for BGE queries
        prompt = f"Represent this sentence for searching relevant passages: {query}"
        return self.model.encode(prompt, normalize_embeddings=True).tolist()

# singlton  
bge_model = EmbeddingModel()
