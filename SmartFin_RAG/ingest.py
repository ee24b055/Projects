import os
from dotenv import load_dotenv
# We switched to langchain_core for modern 2026 standards
from langchain_core.document_loaders import Blob
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# A clean, modern way to load standard text files using langchain_core
class SimpleTextLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        yield Document(page_content=text, metadata={"source": self.file_path})

def load_and_split_document(file_path):
    print(f"\n--- Step 1: Loading document safely from {file_path} ---")
    
    loader = SimpleTextLoader(file_path)
    raw_documents = loader.load()
    
    print("--- Step 2: Splitting text into meaningful chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Successfully split document into {len(chunks)} smaller chunks.")
    
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1} Preview]:\n{chunk.page_content}")
        print("-" * 30)
        
    return chunks

if __name__ == "__main__":
    # FIX: Point exactly to the 'data' folder where the file lives!
    sample_chunks = load_and_split_document("data/earnings_transcript.txt")