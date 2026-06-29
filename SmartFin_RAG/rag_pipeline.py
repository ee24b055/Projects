import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class FinancialAnalysis(BaseModel):
    sentiment_score: float = Field(description="A score from -1.0 (negative) to 1.0 (positive).")
    key_growth_drivers: List[str] = Field(description="List of positive events or revenue drivers.")
    headwinds_and_risks: List[str] = Field(description="List of negative updates or risks.")
    direct_quote_evidence: str = Field(description="A verbatim quote matching the extracted parameters.")

# NEW: This completely replaces ingest.py and vector_store.py on the fly!
def process_new_pdf_document(pdf_path: str):
    print(f"--- Dynamically Processing: {pdf_path} ---")
    
    # 1. Parse raw PDF sheets into text
    loader = PyPDFLoader(pdf_path)
    raw_pages = loader.load()
    
    # 2. Slice text to maintain clean chunk boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_chunks = text_splitter.split_documents(raw_pages)
    
    # 3. Stream and persist vectors instantly using your working embedding layer string
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Clear old vectors out by recreating the store
    vector_db = Chroma.from_documents(
        documents=split_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_db

def run_financial_rag(user_query: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    retrieved_docs = db.similarity_search(user_query, k=2)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # SPEED OPTIMIZATION: Using 3.1-flash-lite with reasoning budgets turned completely off
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0.0,
        extra_set_arguments={"thinking_budget": 0} 
    )
    
    structured_llm = llm.with_structured_output(FinancialAnalysis)
    
    prompt = f"""
    You are an expert quantitative hedge fund analyst. Analyze the following retrieved financial context 
    and answer the user question by cleanly extracting structural analytical fields.
    
    Retrieved Context:
    {context_text}
    
    User Question: {user_query}
    """
    
    structured_output = structured_llm.invoke(prompt)
    return structured_output