import os
from core.database import db
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

import uuid

# Memory Cache for Self-Learning (ID -> {query, answer})
# In production, this should be Redis or a database
interaction_cache = {}

def generate_response(query: str):
    # 1. Search 'Learned QA' Memory for similar past queries
    learned_docs = db.search_learned(query)
    learned_context = ""
    if learned_docs:
        learned_context = f"\n[Previous Verified Answer]: {learned_docs[0]['answer']}\n"

    # 2. Retrieve Standard Documents
    context_docs = db.search(query, limit=3)
    context_text = "\n\n".join([doc["text"] for doc in context_docs])
    
    prompt = f"""
    You are a helpful assistant grounding your answers in the provided context.
    If the context doesn't contain the answer, say you don't know based on the context.
    
    {learned_context}
    
    Context:
    {context_text}
    
    Query: {query}
    
    Response:
    """

    if not GEMINI_API_KEY:
        return {"answer": "Insert the gemini api key in the .env file", "id": "error"}

    try:
        # User requested specific model version
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        response = model.generate_content(prompt)
        text_response = response.text
    except Exception as e:
        print(f"Error with primary model: {e}")
        try:
             # Fallback to standard Flash
             model = genai.GenerativeModel('gemini-1.5-flash')
             response = model.generate_content(prompt)
             text_response = response.text
        except Exception as e2:
             text_response = f"Error generating response: {e2}"
    
    # generate ID and Cache for Feedback
    response_id = str(uuid.uuid4())
    interaction_cache[response_id] = {
        "query": query,
        "answer": text_response
    }
    
    # Keep cache small (simple implementation)
    if len(interaction_cache) > 100:
        interaction_cache.pop(next(iter(interaction_cache)))

    return {"answer": text_response, "id": response_id}

def add_feedback(query_id: str, feedback: str):
    print(f"Feedback received for {query_id}: {feedback}")
    
    if query_id in interaction_cache and feedback.lower() in ["like", "positive", "1"]:
        # Concept: Reinforcement Learning / "Golden" Set
        # If user liked the answer, store it as a 'Learned Answer' in vector DB
        cached = interaction_cache[query_id]
        print(f"Learning from positive interaction: {cached['query']}")
        db.store_learned_answer(cached['query'], cached['answer'])
        return {"status": "success", "message": "Feedback recorded & Answer learned!"}

    return {"status": "success", "message": "Feedback recorded"}
