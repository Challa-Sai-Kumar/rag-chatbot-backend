# Angel One RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot powered by **Gemini** to assist users with support queries based on Angel One documentation and website content.

## Features

- **Gemini-powered chatbot** using `gemini-1.5-flash` model
- **RAG pipeline** with HuggingFace embeddings and Chroma vector store
- **Document-based responses** - only answers based on provided docs/web data
- **Clean Streamlit UI** with chat history and API health checks
- **FastAPI backend** for scalable deployment

## Setup Instructions

### 1. Backend Setup

Navigate to the backend directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:

```
GEMINI_API_KEY=your_google_gemini_api_key
```

Place your PDF and DOCX files in the `documents/` folder, then build the vector database:

```bash
python setup.py
```

Start the FastAPI server:

```bash
python app.py
```

### 2. Frontend Setup

Navigate to the frontend directory and install dependencies:

```bash
cd frontend
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

**Note:** Make sure the FastAPI backend is running before starting the frontend.

## Usage

1. Open your browser and navigate to the Streamlit app (usually `http://localhost:8501`)
2. Type your questions related to Angel One services
3. The chatbot will provide answers based on the loaded documentation
4. If the information isn't available in the documents, it will respond with "I don't know"

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /ask` - Query the chatbot with a question

## Requirements

- Python 3.8+
- Google Gemini API key
- FastAPI
- Streamlit
- ChromaDB
- HuggingFace Transformers