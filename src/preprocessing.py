import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_features(df):
    """
    Creates temporal, lag, rolling, and weather interaction features for time-series forecasting.
    Never drops rows or fails due to missing lag periods or NaNs.
    """
    df = df.copy()
    
    # Ensure timestamp column exists and is valid datetime
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start="2025-01-01 00:00:00", periods=len(df), freq="h")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isnull().any():
            df["timestamp"] = pd.date_range(start="2025-01-01 00:00:00", periods=len(df), freq="h")
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    target_col = "total_consumption_kw"
    if target_col not in df.columns:
        df[target_col] = 3.5
        
    if "temperature" not in df.columns:
        df["temperature"] = 20.0
        
    if "humidity" not in df.columns:
        df["humidity"] = 50.0
        
    if "occupancy" not in df.columns:
        df["occupancy"] = 1.5
        
    if "solar_generation_kw" not in df.columns:
        df["solar_generation_kw"] = 0.0
    
    # Temporal features
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["month"] = df["timestamp"].dt.month.fillna(1).astype(int)
    df["dayofyear"] = df["timestamp"].dt.dayofyear.fillna(1).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Peak tariff hours flag (typically 17:00 - 21:00)
    df["is_peak_hour"] = ((df["hour"] >= 17) & (df["hour"] <= 21)).astype(int)
    
    # Weather interaction features
    base_temp = 21.0
    df["cooling_degree_hours"] = np.maximum(0, df["temperature"] - base_temp)
    df["heating_degree_hours"] = np.maximum(0, base_temp - df["temperature"])
    
    # Time-series Lag features for total_consumption_kw
    df["lag_1h"] = df[target_col].shift(1)
    df["lag_2h"] = df[target_col].shift(2)
    df["lag_3h"] = df[target_col].shift(3)
    df["lag_24h"] = df[target_col].shift(24)
    df["lag_168h"] = df[target_col].shift(168)
    
    # Rolling statistics
    df["rolling_mean_6h"] = df[target_col].shift(1).rolling(window=6).mean()
    df["rolling_std_6h"] = df[target_col].shift(1).rolling(window=6).std()
    df["rolling_mean_24h"] = df[target_col].shift(1).rolling(window=24).mean()
    df["rolling_std_24h"] = df[target_col].shift(1).rolling(window=24).std()
    
    # Fill initial NaN values resulting from lag/rolling shifts gracefully
    mean_cols = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h", "rolling_mean_6h", "rolling_mean_24h"]
    for col in mean_cols:
        df[col] = df[col].bfill().ffill().fillna(df[target_col]).fillna(3.5)
        
    std_cols = ["rolling_std_6h", "rolling_std_24h"]
    for col in std_cols:
        df[col] = df[col].bfill().ffill().fillna(0.0)
        
    # Guarantee no NaNs remain across any column
    df = df.ffill().bfill().fillna(0.0)
    return df


def prepare_tabular_data(df, target_col="total_consumption_kw", test_ratio=0.15, val_ratio=0.15):
    """
    Prepares tabular feature matrices X, y with chronological train/val/test split.
    """
    feature_cols = [
        "temperature", "humidity", "occupancy", "solar_generation_kw",
        "hour", "dayofweek", "month", "is_weekend", "is_peak_hour",
        "cooling_degree_hours", "heating_degree_hours",
        "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h",
        "rolling_mean_6h", "rolling_std_6h", "rolling_mean_24h", "rolling_std_24h"
    ]
    
    X = df[feature_cols].values
    y = df[target_col].values
    timestamps = df["timestamp"].values
    
    total_len = len(df)
    test_len = int(total_len * test_ratio)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len - test_len
    
    X_train, y_train = X[:train_len], y[:train_len]
    X_val, y_val = X[train_len:train_len+val_len], y[train_len:train_len+val_len]
    X_test, y_test = X[train_len+val_len:], y[train_len+val_len:]
    
    ts_train = timestamps[:train_len]
    ts_val = timestamps[train_len:train_len+val_len]
    ts_test = timestamps[train_len+val_len:]
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    scalers = {
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols
    }
    
    splits = {
        "X_train": X_train_scaled, "y_train": y_train, "y_train_scaled": y_train_scaled, "ts_train": ts_train,
        "X_val": X_val_scaled, "y_val": y_val, "y_val_scaled": y_val_scaled, "ts_val": ts_val,
        "X_test": X_test_scaled, "y_test": y_test, "y_test_scaled": y_test_scaled, "ts_test": ts_test
    }
    
    return splits, scalers

def prepare_lstm_sequences(df, lookback=24, target_col="total_consumption_kw", test_ratio=0.15, val_ratio=0.15):
    """
    Prepares 3D sliding window sequences (samples, lookback, features) for LSTM model training.
    """
    feature_cols = ["total_consumption_kw", "temperature", "hour", "is_weekend", "solar_generation_kw"]
    data = df[feature_cols].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X_seq, y_seq, timestamps_seq = [], [], []
    for i in range(lookback, len(data_scaled)):
        X_seq.append(data_scaled[i-lookback:i, :])
        y_seq.append(df[target_col].iloc[i]) # Target is original unscaled kW or scaled kW
        timestamps_seq.append(df["timestamp"].iloc[i])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    timestamps_seq = np.array(timestamps_seq)
    
    total_len = len(X_seq)
    test_len = int(total_len * test_ratio)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len - test_len
    
    lstm_splits = {
        "X_train": X_seq[:train_len], "y_train": y_seq[:train_len],
        "X_val": X_seq[train_len:train_len+val_len], "y_val": y_seq[train_len:train_len+val_len],
        "X_test": X_seq[train_len+val_len:], "y_test": y_seq[train_len+val_len:],
        "ts_test": timestamps_seq[train_len+val_len:],
        "scaler": scaler,
        "feature_cols": feature_cols
    }
    
    return lstm_splits

if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "smart_home_energy.csv")
    df = pd.read_csv(raw_path)
    df_feat = create_features(df)
    print(f"Features engineered! Resulting shape: {df_feat.shape}")
    
    processed_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_energy.csv")
    df_feat.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")
