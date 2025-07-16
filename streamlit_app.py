import streamlit as st
import requests
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Angel One Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

# API configuration
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ask_question(question):
    """Send question to API"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

# Main app
def main():
    st.title("🤖 Angel One Support Chatbot")
    st.markdown("Ask me anything about Angel One services and support!")
    
    # Check API status
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the FastAPI server first.")
        st.code("python app.py")
        return
    
    st.success("✅ API is running")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot can help you with:
        - Angel One support queries
        - Insurance information
        - Account-related questions
        - General support topics
        
        **Note**: I can only answer questions based on the loaded documentation.
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**API Status**: ✅ Connected" if check_api_health() else "**API Status**: ❌ Disconnected")

if __name__ == "__main__":
    main()