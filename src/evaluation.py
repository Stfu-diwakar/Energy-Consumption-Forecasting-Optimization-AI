import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred):
    """
    Computes RMSE, MAE, MAPE, and R2 performance metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    
    # Avoid division by zero in MAPE calculation
    epsilon = 1e-5
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100)
    
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }

def print_evaluation_summary(metrics_dict):
    """
    Prints a formatted evaluation table comparing all algorithms.
    """
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'RMSE (kW)':<10} | {'MAE (kW)':<10} | {'MAPE (%)':<10} | {'R2 Score':<10}")
    print("="*60)
    for model_name, metrics in metrics_dict.items():
        print(f"{model_name:<15} | {metrics['RMSE']:<10.4f} | {metrics['MAE']:<10.4f} | {metrics['MAPE']:<10.2f} | {metrics['R2']:<10.4f}")
    print("="*60 + "\n")
