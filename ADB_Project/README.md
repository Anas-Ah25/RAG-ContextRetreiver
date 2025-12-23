## Project Structure

```text
.
├── backend/                # FastAPI Backend Application
│   ├── core/               # Core RAG Logic and Database interactions
│   │   ├── database.py     # Qdrant Vector Database integration
│   │   ├── models.py       # Embedding model (BGE) initialization
│   │   └── rag.py          # RAG pipeline and Gemini LLM integration
│   ├── data/               # Persistent storage for Qdrant (local mode)
│   ├── main.py             # FastAPI entry point and API routes
│   ├── requirements.txt    # Python dependencies
│   └── .env                # Environment variables (API Keys, etc.)
├── frontend/               # Frontend Application
│   └── index.html          # Single-page interface with Teal theme
└── embedding_comparison.ipynb # Research notebook for embedding model evaluation
```

##  Main Components

### 1. Backend (FastAPI)
The backend serves as the orchestration layer, handling requests from the frontend and coordinating between the vector database and the LLM.
- **`query`**: Processes user questions, retrieves context, and generates answers.
- **`upload`**: Handles PDF and Text file uploads, extracts text, and indexes them into the vector DB.
- **`feedback`**: Captures user feedback for future self-learning optimizations.
- **`seed`**: Populates the database with initial mock data for testing.

### 2. Core Logic (`backend/core`)
- **`database.py`**: Utilizes **Qdrant** for vector storage. It manages collection creation and handles "upsert" operations for document embeddings.
- **`models.py`**: Configures the **BAAI/bge-small-en-v1.5** embedding model via `sentence-transformers`. It includes specific prompting logic to optimize retrieval accuracy.
- **`rag.py`**: Implements the RAG workflow. It retrieves relevant document context from Qdrant and constructs a grounded prompt for the **Google Gemini** model (using Gemini 2.5 Flash Preview).


### 4. Embedding Comparison Notebook
A Jupyter notebook used during the development phase to evaluate different embedding models (`LaBSE`, `all-MiniLM-L6-v2`, `BGE`, etc.) against a custom evaluation dataset to ensure the best retrieval performance.

## Setup & Installation

1. **Backend**:
   ```bash
   pip install uv 
   uv venv 
   source .venv/bin/activate       
   cd backend
   uv pip install -r requirements.txt
   # Add your GEMINI_API_KEY to .env
   python main.py
   ```

2. **Frontend**:
   Simply open `frontend/index.html` in a browser or serve it using a local live server.

## Technologies Used
- **Language**: Python (Backend), Javascript/HTML/CSS (Frontend)
- **Framework**: FastAPI
- **Vector DB**: Qdrant
- **Embeddings**: BGE (HuggingFace)
- **LLM**: Google Gemini
- **UI**: CSS
