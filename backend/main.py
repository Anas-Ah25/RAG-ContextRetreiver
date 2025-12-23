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
    answer = generate_response(query)
    return {"answer": answer, "id": str(uuid.uuid4())}

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

    # Simple chunking for demo
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    ids = [hash(chunk + str(uuid.uuid4())) % (10**8) for chunk in chunks]
    
    db.add_documents(chunks, ids)
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
    db.add_documents(mock_data, ids)
    return {"status": "seeded", "count": len(mock_data)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
