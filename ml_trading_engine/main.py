from src.data.ingestion import fetch_historical_data
from src.features.features import create_features_and_target
from src.models.model import train_predict_pipeline
from src.backtest.backtester import run_backtest

if __name__ == "__main__":
    # Standard engineering configuration wrapper
    raw_data = fetch_historical_data("AAPL", "2020-01-01", "2026-06-01")
    processed_data = create_features_and_target(raw_data)
    test_predictions = train_predict_pipeline(processed_data)
    run_backtest(test_predictions)