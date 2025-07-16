import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from dotenv import load_dotenv
import time

load_dotenv()

class GeminiRAGChatbot:
    def __init__(self):
        self.chroma_path = "chroma_db"
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.vectorstore = None
        self.retriever = None
        self.setup_retriever()
        
    def setup_retriever(self):
        """Setup the retriever"""
        if os.path.exists(self.chroma_path):
            self.vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Get more relevant chunks
            )
        else:
            raise Exception("Database not found. Please run python setup.py first.")
    
    def format_docs(self, docs):
        """Format documents for response"""
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)
    
    def query(self, question: str) -> str:
        """Query the chatbot using Gemini"""
        if not self.retriever:
            return "Database not initialized. Please create the database first."
        
        print("input query:", question)

        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                return "I don't know"
            
            # Format context
            context = self.format_docs(docs)

            # print(f"\n[Retrieved Context]:\n{context}\n{'='*50}")
            
            # Create prompt for Gemini
            prompt = f"""You are a helpful customer support assistant for Angel One financial services. 
            Use ONLY the following context to answer the user's question. If you cannot find the answer in the context, 
            respond with "I don't know" - do not make up information.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Only use information from the provided context
            2. If the answer is not in the context, say "I don't know"
            3. Be concise and helpful
            4. Mention the source if relevant

            Answer:"""
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            print("<-----------------------------response:", response)
            answer = response.text.strip()
            print("<--------------------------------answer:", answer)
            
            # Check if the response indicates lack of knowledge
            if any(phrase in answer.lower() for phrase in ["i don't know", "i cannot", "not found", "no information", "not mentioned"]):
                return "I don't know"
            
            return answer
            
        except Exception as e:
            print(f"Error in query: {e}")
            return "I don't know"

# Test the chatbot
if __name__ == "__main__":
    try:
        chatbot = GeminiRAGChatbot()
        
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            
            response = chatbot.query(question)
            print(f"Answer: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your GEMINI_API_KEY in .env file")