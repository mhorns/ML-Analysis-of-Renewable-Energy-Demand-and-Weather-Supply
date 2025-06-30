import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import os
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



def load_final_data(DATA_DIR: Path, region: str):
    """Gets the finalized per region energy and weather data"""
    train_df = pd.read_csv(DATA_DIR / f"final_data_train_{region}.csv", parse_dates=["period"])
    val_df = pd.read_csv(DATA_DIR / f"final_data_val_{region}.csv", parse_dates=["period"])
    test_df = pd.read_csv(DATA_DIR / f"final_data_test_{region}.csv", parse_dates=["period"])

    return train_df, val_df, test_df


def safe_mape(y_true, y_pred):
    """Safely calculate the MAPE when there are zeros"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid divide-by-zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan

    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    return mape


def evaluate_naive_lag_24(DATA_DIR: Path, region, target_col: str = 'Total interchange', naive_pred_col: str = 'lag_interchange_24h'):
    """Create and evaluate naive 24hr lookback prediction of the target"""
    train_df, val_df, test_df = load_final_data(DATA_DIR, region)
    val_df = val_df.copy()
    val_df['naive_pred'] = val_df[target_col].shift(24)

    y_true = val_df[target_col].values
    y_pred = val_df[naive_pred_col].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, mape, r2

def get_naive_results(DATA_DIR: Path, regions):
    """Parse through each region and create naive prediction results based on 24h lookback of target"""
    results = []
    for region in regions:
        rmse, mae, mape, r2 = evaluate_naive_lag_24(DATA_DIR, region)

        results.append({
            'Region': region,
            'Model': 'Naive Pred',
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / f"naive_pred_results.csv"

    if out_path.exists():
        print(f"Skipping {out_path.name} (already exists)")

    else:
        results_df.to_csv(out_path, index=False)
        print(f"Saved: naive_pred_results.csv")

    return results_df


def plot_naive_analytics(FIG_DIR: Path, results_df: pd.DataFrame):
    """Creates comparison plots for each region result evaluation metrics from naive validate predictions"""
    results_df_sorted = results_df.sort_values(by="Region")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    sns.barplot(data=results_df_sorted, x='Region', y='RMSE', ax=axes[0], palette='Blues_r', hue='Region', legend=False)
    axes[0].set_title("RMSE by Region")
    axes[0].set_ylabel("RMSE")

    sns.barplot(data=results_df_sorted, x='Region', y='MAE', ax=axes[1], palette='Greens_r', hue='Region', legend=False)
    axes[1].set_title("MAE by Region")
    axes[1].set_ylabel("MAE")

    sns.barplot(data=results_df_sorted, x='Region', y='MAPE', ax=axes[2], palette='Reds_r', hue='Region', legend=False)
    axes[2].set_title("MAPE by Region")
    axes[2].set_ylabel("MAPE")

    sns.barplot(data=results_df_sorted, x='Region', y='R2', ax=axes[3], palette='Purples_r', hue='Region', legend=False)
    axes[3].set_title("R2 by Region")
    axes[3].set_ylabel("R2")
    axes[3].set_xlabel("Region")

    plt.tight_layout()
    plt.savefig(FIG_DIR / f'Naive_pred_results_by_region.png')
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

    # 13 EIA region codes
    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']

    # Create naive validation set predictions and accumulate results
    results_df = get_naive_results(DATA_DIR, regions)

    # Plot results to compare across regions
    plot_naive_analytics(FIG_DIR, results_df)


if __name__ == "__main__":
    main()