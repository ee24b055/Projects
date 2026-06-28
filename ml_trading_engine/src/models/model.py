from sklearn.ensemble import RandomForestClassifier
import pandas as pd 

def train_predict_pipeline(df: pd.DataFrame, train_size: float = 0.75):
    feature_cols = ['Return', 'SMA_Ratio', 'RSI']
    X = df[feature_cols]
    y = df['Target']
    
    # Chronological Time-Series split to avoid forward data leakage
    split_idx = int(len(df) * train_size)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train the Machine Learning Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on out-of-sample data
    test_df = df.iloc[split_idx:].copy()
    test_df['Predicted_Signal'] = model.predict(X_test)
    
    print(f"[+] Out-of-Sample Model Accuracy: {model.score(X_test, y_test):.2%}")
    return test_df