import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from ingest import load_and_split_document

# Load the API Key from your .env file
load_dotenv()

def create_vector_database(file_path):
    # 1. Load the chunks we generated from Phase 2
    chunks = load_and_split_document(file_path)
    
    print("\n--- Step 3: Initializing Google Gemini Embedding Model ---")
    # Use a supported Gemini embedding model name for the Google GenAI SDK
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        task_type="retrieval_document"
    )
    
    print("--- Step 4: Storing vectors into local ChromaDB ---")
    # Create the vector store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector database successfully created and saved locally!")
    return vector_db

if __name__ == "__main__":
    # Test creating the DB
    db = create_vector_database("data/earnings_transcript.txt")
    
    # Test semantic searching
    query = "What are the manufacturing delays or logistical issues?"
    print(f"\n--- Testing Search for: '{query}' ---")
    
    # Fetch the single most mathematically relevant chunk
    matching_docs = db.similarity_search(query, k=1) 
    
    print(f"Most Relevant Chunk Found:\n{matching_docs[0].page_content}")