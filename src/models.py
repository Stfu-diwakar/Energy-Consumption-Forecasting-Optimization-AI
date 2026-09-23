import os
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

# Configure Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

class ARIMAModelWrapper:
    """
    Statistical Time-Series ARIMA model wrapper.
    """
    def __init__(self, order=(2, 1, 1)):
        self.order = order
        self.model_res = None
        self.last_train_series = None

    def fit(self, train_series):
        self.last_train_series = train_series
        model = ARIMA(train_series, order=self.order)
        self.model_res = model.fit()
        return self

    def predict(self, steps):
        if self.model_res is None:
            raise ValueError("ARIMA model not trained!")
        forecast = self.model_res.forecast(steps=steps)
        return forecast.values if hasattr(forecast, "values") else np.array(forecast)

    def save(self, filepath):
        joblib.dump({"order": self.order, "params": self.model_res.params, "last_train": self.last_train_series[-100:]}, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.order = data["order"]
        # Reconstruct fit model on sample
        model = ARIMA(data["last_train"], order=self.order)
        self.model_res = model.fit()
        return self


class XGBoostModelWrapper:
    """
    XGBoost Gradient Boosted Decision Trees Regressor for time-series forecasting.
    """
    def __init__(self, n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        self.model.save_model(filepath)

    def load(self, filepath):
        self.model.load_model(filepath)
        return self


class KerasLSTMModelWrapper:
    """
    Keras 3 Deep Learning Sequential LSTM Neural Network.
    """
    def __init__(self, lookback=24, num_features=5):
        self.lookback = lookback
        self.num_features = num_features
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.lookback, self.num_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=15, batch_size=32):
        validation_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)
        return self
