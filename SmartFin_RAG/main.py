import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from rag_pipeline import run_financial_rag, process_new_pdf_document

app = FastAPI(
    title="SmartFin-RAG Enterprise",
    description="Dynamic Multi-Document Extraction Engine",
    version="2.0.0"
)

# Create a temporary storage directory for uploaded files if it doesn't exist
TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"status": "online", "message": "SmartFin Enterprise Cluster Online"}

# NEW NEW NEW: Upload any PDF statement directly here!
@app.post("/api/v1/upload-statement")
def upload_financial_statement(file: UploadFile = File(...)):
    """
    Accepts an enterprise PDF financial document, processes its structures, 
    and completely refreshes the local vector brain context dynamically.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid format. Please supply a valid PDF file.")
    
    try:
        # Save the uploaded file locally temporarily
        temp_file_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Trigger our new automated ingestion pipeline
        process_new_pdf_document(temp_file_path)
        
        return {
            "success": True,
            "message": f"Successfully parsed {file.filename} and updated vector store context!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analyze")
def analyze_financials(ticker: str, question: str):
    try:
        full_query = f"For stock ticker {ticker}: {question}"
        analysis_report = run_financial_rag(full_query)
        return {
            "success": True,
            "ticker": ticker.upper(),
            "data": analysis_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))