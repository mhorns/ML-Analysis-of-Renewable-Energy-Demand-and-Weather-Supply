import pandas as pd
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def safe_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE) while safely handling zero targets.

    :param y_true: Ground truth target values.
    :param y_pred: Predicted values.
    :return: MAPE as a percentage, or NaN if no valid denominator values exist.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid divide-by-zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan

    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    return mape

def evaluate_rnn_model(region: str, model_type: str, units_len: int, DATA_DIR: Path):
    """
    Evaluate a trained LSTM or GRU model on the test set and return performance metrics.

    :param region: Region identifier used to load the model and test data.
    :param model_type: RNN type ("LSTM" or "GRU").
    :param units_len: Number of RNN layers in the model, used in the filename.
    :param DATA_DIR: Path to the directory containing saved models and test data.
    :return: Dictionary containing RMSE, MAE, MAPE, and R2 metrics, or None on failure.
    """
    model_path = DATA_DIR / f"{region}_{model_type.lower()}_{units_len}unit_model.h5"
    print(f"Model path: {model_path}")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        X_test = np.load(DATA_DIR / f"rnn_data_X_test_{region}.npy")
        y_test = np.load(DATA_DIR / f"rnn_data_y_test_{region}.npy")
        y_train = np.load(DATA_DIR / f"rnn_data_y_train_{region}.npy")

        # Re-create scaler from y_train
        scaler_y = StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))

        # Predict and inverse-transform
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        y_true = y_test.ravel()
        y_pred = y_pred.ravel()

        return {
            'Region': region,
            'Model': f"{model_type}_{units_len}unit",
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': safe_mape(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    except Exception as e:
        print(f"[ERROR] {region} | {model_type}: {e}")
        return None

def evaluate_xgboost_model(region: str, DATA_DIR: Path):
    """
    Evaluate a trained XGBoost model on the test set and return performance metrics.

    :param region: Region identifier used to load the model and test data.
    :param DATA_DIR: Path to the directory containing saved models and test data.
    :return: Dictionary containing RMSE, MAE, MAPE, and R2 metrics, or None on failure.
    """
    try:
        model = XGBRegressor()
        model.load_model(DATA_DIR / f"{region}_xgboost_model.json")

        test_df = pd.read_csv(DATA_DIR / f"final_data_test_{region}.csv", parse_dates=["period"])
        drop_cols = ['period', 'respondent', 'respondent-name', 'Region', 'MO', 'DY', 'HR', 'day_of_year', 'day_of_week']
        targ_col = ['Total interchange']

        X = test_df.drop(columns=drop_cols + targ_col)
        y = test_df[targ_col].values.ravel()

        y_pred = model.predict(X)

        return {
            'Region': region,
            'Model': 'XGBoost',
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'MAPE': safe_mape(y, y_pred),
            'R2': r2_score(y, y_pred)
        }

    except Exception as e:
        print(f"[ERROR] XGBoost eval failed for {region}: {e}")
        return None

def plot_test_model_metrics(FIG_DIR: Path, df: pd.DataFrame):
    """
    Generate and save bar plots comparing model test metrics across regions.

    :param FIG_DIR: Path to the directory for saving output plots.
    :param df: DataFrame containing test metrics (RMSE, MAE, MAPE, R2) by model and region.
    :return: None
    """
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']

    palette = {
        'LSTM_3unit': '#08306B',  # navy
        'LSTM_2unit': '#2171B5',  # blue
        'GRU_3unit': '#00441B',  # dark green
        'GRU_2unit': '#238B45',  # green
        'GRU_4unit': '#41ab5d',  # light green
        'ae_3unit': '#e6550d',  # orange
        'XGBoost': '#7F2704',  # rust
        'Naive Pred': '#525252'  # gray
    }

    # Error handling for missing palette keys
    missing_models = set(df['Model']) - set(palette.keys())
    if missing_models:
        print(f"Warning: Missing color mapping for models: {missing_models}")

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x='Region', y=metric,
            hue='Model',
            palette=palette,
            dodge=True,
            errorbar=None  # disable error bars for clean comparison
        )
        plt.title(f'Test Set Model Comparison by Region - {metric.upper()}')
        plt.xlabel('Region')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"Compare_model_test_{metric}_barplot_plot.png")
        plt.close()


def main():
    # Base directory: go up one level from current script (i.e., from 'src/' to project root)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Path to the data directory at the same level as 'src'
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Data Directory: {DATA_DIR}")

    # Path to the figs directory at the same level as 'src'
    FIG_DIR = BASE_DIR / "figs"
    FIG_DIR.mkdir(exist_ok=True)
    print(f"Figures Directory: {FIG_DIR}")
    OUT_FILE = DATA_DIR / "model_test_metrics_summary.csv"

    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']
    model_types = [
        ("LSTM", 3),
        ("LSTM", 2),
        ("GRU", 3),
        ("GRU", 2),
        ("GRU", 4),
    ]

    results = []

    # RNN models
    for model_type, units_len in model_types:
        for region in regions:
            res = evaluate_rnn_model(region, model_type, units_len, DATA_DIR)
            if res:
                results.append(res)

    # XGBoost models
    for region in regions:
        res = evaluate_xgboost_model(region, DATA_DIR)
        if res:
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved test results: {OUT_FILE}")

    plot_test_model_metrics(FIG_DIR, df)


if __name__ == "__main__":
    main()
