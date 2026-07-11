import io
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from main import app

client = TestClient(app)

def test_swagger_docs_accessible():
    """Verify API documentation is online."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_upload_statement_validation():
    """Verify endpoint catches missing file payloads."""
    response = client.post("/api/v1/upload-statement?ticker=NVDA")
    assert response.status_code == 422

def test_mock_pdf_upload():
    """Test full background ingestion process using a structurally valid layout PDF."""
    # Build a structurally valid PDF in memory
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "Record quarterly Data Center revenue of $35.6 billion.")
    c.drawString(100, 730, "Surging customer demand for AI infrastructure.")
    c.drawString(100, 710, "Gross margin fluctuations plus or minus 50 basis points.")
    c.save()
    pdf_buffer.seek(0)
    
    mock_file = ("test_document.pdf", pdf_buffer.read(), "application/pdf")
    
    response = client.post(
        "/api/v1/upload-statement?ticker=NVDA",
        files={"file": mock_file}
    )
    
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert "accepted successfully" in json_data["message"]