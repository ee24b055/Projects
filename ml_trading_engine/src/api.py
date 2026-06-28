from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.data.ingestion import fetch_historical_data
from src.features.features import create_features_and_target
from src.models.model import train_predict_pipeline

app = FastAPI(title="Quant ML Pipeline Engine")

class TickerRequest(BaseModel):
    ticker: str

@app.post("/predict")
async def predict_ticker(payload: TickerRequest):
    try:
        raw = fetch_historical_data(payload.ticker, "2022-01-01", "2026-06-01")
        featured = create_features_and_target(raw)
        predictions = train_predict_pipeline(featured)
        
        latest_row = predictions.iloc[-1]
        direction = "UP" if latest_row['Predicted_Signal'] == 1 else "DOWN"
        
        return {
            "ticker": payload.ticker,
            "prediction": direction,
            "metrics": {
                "rsi": float(latest_row['RSI']),
                "timestamp": str(predictions.index[-1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))