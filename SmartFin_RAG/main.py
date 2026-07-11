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
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def frontend_interface():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SmartFin-RAG Portal</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-50 min-h-screen flex flex-col items-center justify-center p-6">
        <div class="max-w-xl w-full bg-white shadow-md rounded-lg p-6 border border-slate-200">
            <h1 class="text-xl font-bold text-slate-800 mb-4">SmartFin-RAG Execution Terminal</h1>
            
            <form id="uploadForm" class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-slate-600">Ticker Target</label>
                    <input type="text" id="ticker" value="NVDA" class="mt-1 block w-full border border-slate-300 rounded p-2 focus:ring-blue-500" required>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-600">Financial Statement (PDF)</label>
                    <input type="file" id="pdfFile" accept=".pdf" class="mt-1 block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" required>
                </div>
                <button type="submit" class="w-full bg-blue-600 text-white p-2 rounded font-medium hover:bg-blue-700 transition">Submit Statement</button>
            </form>

            <div id="statusResult" class="mt-6 hidden bg-slate-100 p-4 rounded text-xs font-mono text-slate-700 whitespace-pre-wrap"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const ticker = document.getElementById('ticker').value;
                const fileInput = document.getElementById('pdfFile').files[0];
                const statusBox = document.getElementById('statusResult');
                
                statusBox.classList.remove('hidden');
                statusBox.innerText = "Processing ingestion pipeline...";

                const formData = new FormData();
                formData.append('file', fileInput);

                try {
                    const response = await fetch(`/api/v1/upload-statement?ticker=${ticker}`, {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    statusBox.innerText = JSON.stringify(result, null, 2);
                } catch (error) {
                    statusBox.innerText = "Network execution error occurred.";
                }
            });
        </script>
    </body>
    </html>
    """
