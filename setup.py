#!/usr/bin/env python3

import os
import sys
from data_loader import DataLoader

def main():
    print("Setting up RAG Chatbot...")
    
    # Create documents folder if it doesn't exist
    if not os.path.exists('documents'):
        os.makedirs('documents')
        print("Created documents folder")
        print("Place your PDF and DOCX fiels here")
    
    # Load and create database
    try:
        loader = DataLoader()
        loader.create_database()
        print("Database created succesfully!")
    except Exception as e:
        print(f"Error creating database: {e}")
        return
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    print("Start the API server: python app.py")
    print("In another terminal, run: streamlit run streamlit_app.py")
    print("Open your browser to the Streamlit URL")

if __name__ == "__main__":
    main()