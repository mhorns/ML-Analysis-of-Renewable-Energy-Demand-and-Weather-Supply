import pandas as pd
import numpy as np
import random
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def load_model_metrics(DATA_DIR: Path):
    """Gets the model result metrics and appends each into single table for plotting"""
    types = ['LSTM', 'GRU']
    units = [[128, 64, 32], [64, 32]]

    model_df = []

    for type in types:
        for unit in units:
            df = pd.read_csv(DATA_DIR / f"{type}_{len(unit)}unit_train_results.csv")
            df['Model'] = f"{type}_{len(unit)}unit"
            model_df.append(df)

    xgboost_df = pd.read_csv(DATA_DIR / f"XGBoost_train_results.csv")
    xgboost_df = xgboost_df.drop(columns=['Best_Model'], errors='ignore')
    xgboost_df['Model'] = str("XGBoost")
    model_df.append(xgboost_df)
    naive_df = pd.read_csv(DATA_DIR / f"naive_pred_results.csv")
    model_df.append(naive_df)

    summary_df = pd.concat(model_df, ignore_index=True)
    summary_df = summary_df[~summary_df.apply(lambda row: all(row == summary_df.columns), axis=1)]

    summary_df.to_csv(DATA_DIR / "model_metrics_summary.csv", index=False)

    return summary_df


def plot_model_metrics(FIG_DIR, summary_df):
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']

    palette = {
        'LSTM_3unit': '#08306B',  # navy
        'LSTM_2unit': '#2171B5',  # blue
        'GRU_3unit': '#00441B',  # dark green
        'GRU_2unit': '#238B45',  # green
        'XGBoost': '#7F2704',  # rust
        'Naive Pred': '#525252'  # gray
    }

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=summary_df,
            x='Region', y=metric,
            hue='Model',
            palette=palette,
            dodge=True,
            errorbar=None  # disable error bars for clean comparison
        )
        plt.title(f'Model Comparison by Region - {metric.upper()}')
        plt.xlabel('Region')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"Compare_model_{metric}_barplot_plot.png")
        plt.close()


def main():
    # Setting seeds for the RNN reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

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
    # regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']
    regions = ['MIDW']

    summary_df = load_model_metrics(DATA_DIR)
    plot_model_metrics(FIG_DIR, summary_df)

if __name__ == "__main__":
    main()