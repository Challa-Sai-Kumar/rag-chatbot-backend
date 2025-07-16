from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_rag_chatbot import GeminiRAGChatbot
import os

app = FastAPI(title="Gemini RAG Chatbot API", version="1.0.0")

# Initialize chatbot
try:
    chatbot = GeminiRAGChatbot()
except Exception as e:
    print(f"Error intializing chatbot: {e}")
    chatbot = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database_exists": os.path.exists("chroma_db")}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)