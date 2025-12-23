import os
from core.database import db
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def generate_response(query: str):
    # Retrieve
    context_docs = db.search(query, limit=3)
    context_text = "\n\n".join([doc["text"] for doc in context_docs])
    
    prompt = f"""
    You are a helpful assistant grounding your answers in the provided context.
    If the context doesn't contain the answer, say you don't know based on the context.
    
    Context:
    {context_text}
    
    Query: {query}
    
    Response:
    """

    if not GEMINI_API_KEY:
        return f"Insert the gemini api key in the .env file"

    try:
        # User requested specific model version
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error with primary model: {e}")
        try:
             # Fallback to standard Flash
             model = genai.GenerativeModel('gemini-1.5-flash')
             response = model.generate_content(prompt)
             return response.text
        except Exception as e2:
             return f"Error generating response: {e2}"

def add_feedback(query_id: str, feedback: str):
    # Self-learning layer: In a real app, this might update rankings or store for fine-tuning.
    # For now, we'll log it or update a simple metadata field if applicable.
    print(f"Feedback received for {query_id}: {feedback}")
    # Logic to "learn" could go here if we had a persistent feedback store.
    return {"status": "success", "message": "Feedback recorded"}
