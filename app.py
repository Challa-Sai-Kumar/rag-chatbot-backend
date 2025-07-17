import subprocess
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_rag_chatbot import GeminiRAGChatbot

app = FastAPI(title="Gemini RAG Chatbot API", version="1.0.0")

# Global chatbot instance
chatbot = None

# Request and Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Initialize ChromaDB and Chatbot on app startup
@app.on_event("startup")
async def startup_event():
    global chatbot
    try:
        if not os.path.exists("chroma_db"):
            print("📦 Chroma DB not found. Running setup...")
            subprocess.run(["python", "setup.py"], check=True)
        else:
            print("Chroma DB already exists. Skipping setup.")

        print("Initializing Gemini RAG Chatbot...")
        chatbot = GeminiRAGChatbot()
        print("Chatbot initialized successfully!")
    except Exception as e:
        print(f"Error during startup: {e}")
        chatbot = None

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_exists": os.path.exists("chroma_db"),
        "chatbot_initialized": chatbot is not None
    }

# Ask Question Endpoint
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer = chatbot.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
