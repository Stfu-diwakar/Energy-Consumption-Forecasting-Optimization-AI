import os
import json
import joblib
import pandas as pd
import numpy as np

from src.data_generator import generate_smart_home_data
from src.preprocessing import create_features, prepare_tabular_data, prepare_lstm_sequences
from src.models import ARIMAModelWrapper, XGBoostModelWrapper, KerasLSTMModelWrapper
from src.evaluation import compute_metrics, print_evaluation_summary

def main():
    print("="*60)
    print(" Smart Home Energy Consumption Forecasting - Model Training Pipeline")
    print("="*60)
    
    # Set paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    raw_path = os.path.join(data_dir, "smart_home_energy.csv")
    processed_path = os.path.join(data_dir, "processed_energy.csv")
    
    # 1. Data Collection / Simulation
    if not os.path.exists(raw_path):
        print("\n[Step 1] Raw dataset not found. Generating 1-year smart home energy data...")
        df_raw = generate_smart_home_data()
        df_raw.to_csv(raw_path, index=False)
    else:
        print("\n[Step 1] Loading raw smart home dataset...")
        df_raw = pd.read_csv(raw_path)
        
    print(f"Raw dataset shape: {df_raw.shape}")
    
    # 2. Preprocessing & Feature Engineering
    print("\n[Step 2] Engineering features (temporal, lag, rolling aggregations, weather)...")
    df_feat = create_features(df_raw)
    df_feat.to_csv(processed_path, index=False)
    print(f"Processed dataset shape: {df_feat.shape}")
    
    # Prepare Tabular Data Splits
    splits, scalers = prepare_tabular_data(df_feat)
    joblib.dump(scalers, os.path.join(models_dir, "scaler.pkl"))
    print(f"Dataset split: Train={len(splits['X_train'])}, Val={len(splits['X_val'])}, Test={len(splits['X_test'])}")
    
    metrics_summary = {}
    test_predictions = {"timestamps": [str(ts)[:19] for ts in splits["ts_test"]], "actual": splits["y_test"].tolist()}
    
    # 3. Model 1: ARIMA / Statistical Time-Series
    print("\n[Step 3] Fitting Statistical ARIMA model...")
    arima_train_series = df_feat["total_consumption_kw"].iloc[:len(splits["X_train"])+len(splits["X_val"])].values
    arima_model = ARIMAModelWrapper(order=(2, 1, 1))
    arima_model.fit(arima_train_series)
    arima_preds = arima_model.predict(steps=len(splits["X_test"]))
    
    metrics_summary["ARIMA"] = compute_metrics(splits["y_test"], arima_preds)
    test_predictions["ARIMA"] = arima_preds.tolist()
    arima_model.save(os.path.join(models_dir, "arima_model.pkl"))
    print("ARIMA Model trained & saved!")

    # 4. Model 2: XGBoost Machine Learning Regressor
    print("\n[Step 4] Training XGBoost Regressor...")
    xgb_model = XGBoostModelWrapper(n_estimators=150, max_depth=6, learning_rate=0.05)
    xgb_model.fit(splits["X_train"], splits["y_train_scaled"], splits["X_val"], splits["y_val_scaled"])
    
    xgb_preds_scaled = xgb_model.predict(splits["X_test"])
    xgb_preds = scalers["scaler_y"].inverse_transform(xgb_preds_scaled.reshape(-1, 1)).flatten()
    
    metrics_summary["XGBoost"] = compute_metrics(splits["y_test"], xgb_preds)
    test_predictions["XGBoost"] = xgb_preds.tolist()
    xgb_model.save(os.path.join(models_dir, "xgboost_model.json"))
    print("XGBoost Model trained & saved!")

    # 5. Model 3: Keras LSTM Neural Network
    print("\n[Step 5] Training Keras Stacked LSTM Neural Network...")
    lstm_splits = prepare_lstm_sequences(df_feat, lookback=24)
    lstm_model = KerasLSTMModelWrapper(lookback=24, num_features=lstm_splits["X_train"].shape[2])
    
    # Train LSTM
    lstm_model.fit(
        lstm_splits["X_train"], lstm_splits["y_train"],
        lstm_splits["X_val"], lstm_splits["y_val"],
        epochs=12, batch_size=32
    )
    
    lstm_preds = lstm_model.predict(lstm_splits["X_test"])
    metrics_summary["LSTM"] = compute_metrics(lstm_splits["y_test"], lstm_preds)
    test_predictions["LSTM"] = lstm_preds.tolist()
    lstm_model.save(os.path.join(models_dir, "lstm_model.keras"))
    print("Keras LSTM Model trained & saved!")

    # 6. Benchmark Comparison Summary
    print_evaluation_summary(metrics_summary)
    
    # Save Metrics & Predictions JSON
    metrics_file = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({
            "metrics": metrics_summary,
            "test_sample_predictions": {
                "timestamps": test_predictions["timestamps"][:168], # First 7 days of test set
                "actual": test_predictions["actual"][:168],
                "ARIMA": test_predictions["ARIMA"][:168],
                "XGBoost": test_predictions["XGBoost"][:168],
                "LSTM": test_predictions["LSTM"][:168]
            }
        }, f, indent=2)
        
    print(f"\nAll models trained successfully! Metrics saved to {metrics_file}")
    print("="*60)

if __name__ == "__main__":
    main()
