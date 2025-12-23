from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from core.database import db
from core.rag import generate_response, add_feedback
from core.models import bge_model
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
async def query_endpoint(query: str = Form(...)):
    # answer is now a dict {answer: str, id: str}
    return generate_response(query)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    content = await file.read()
    
    text = ""
    if filename.endswith(".pdf"):
        import io
        from pypdf import PdfReader
        pdf_reader = PdfReader(io.BytesIO(content))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = content.decode("utf-8")
    
    if not text.strip():
        return {"error": "Empty or unreadable file"}

    # Improved chunking with larger size and overlap
    CHUNK_SIZE = 1500  # Larger chunks to capture more context
    CHUNK_OVERLAP = 200  # Overlap to avoid cutting concepts mid-sentence
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for sep in ['. ', '.\n', '? ', '!\n', '\n\n']:
                last_sep = chunk.rfind(sep)
                if last_sep > CHUNK_SIZE * 0.7:  # Only if we're past 70% of chunk
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        
        chunks.append(chunk.strip())
        start = end - CHUNK_OVERLAP  # Move back by overlap amount
        
        if start >= len(text):
            break
    
    # Filter out empty chunks
    chunks = [c for c in chunks if c.strip()]
    ids = [hash(chunk + str(uuid.uuid4())) % (10**8) for chunk in chunks]
    
    metadatas = [{"filename": filename} for _ in chunks]
    db.add_documents(chunks, ids, metadatas)
    return {"filename": filename, "chunks": len(chunks)}

@app.post("/feedback")
def feedback_endpoint(query_id: str = Form(...), feedback: str = Form(...)):
    return add_feedback(query_id, feedback)

@app.post("/seed")
def seed_mock_data():
    mock_data = [
        "Vector databases like Qdrant are essential for storing high-dimensional embeddings.",
        "RAG (Retrieval-Augmented Generation) combines retrieval with generative LLMs.",
        "BGE embeddings are state-of-the-art for retrieval tasks.",
        "Advanced Database Systems often cover topics like query optimization and indexing.",
        "Teal is a beautiful color for a modern AI interface."
    ]
    ids = [i for i in range(len(mock_data))]
    metadatas = [{"filename": "System Knowledge"} for _ in mock_data]
    db.add_documents(mock_data, ids, metadatas)
    return {"status": "seeded", "count": len(mock_data)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
