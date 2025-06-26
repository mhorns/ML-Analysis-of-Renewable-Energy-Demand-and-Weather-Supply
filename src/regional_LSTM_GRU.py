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

def build_sequences(df, feature_cols, target_col, window_size):
    """Transforms a time series dataframe into sequences"""
    print(f"Building sequences for window size: {window_size}")
    X, y = [], []

    # Create 3D X array shape (n_samples, window_size, n_features) and 1D y array (n_samples,)
    for i in range(window_size, len(df)):
        X.append(df[feature_cols].iloc[i-window_size:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y)





def normalize_features(train_df, val_df, test_df, exclude_cols=None):
    """Fits a StandardScaler on training data, applies to train/val/test sets"""
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler



def train_evaluate_lstm(DATA_DIR, FIG_DIR, X_train, y_train, X_val, y_val, X_test, y_test, region):
    """Creates the RNN model and trains based on parameters given.  Plots the loss over epochs via helper,
    creates evaluation metrics and saves model"""
    # Scale y for training
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Build LSTM model with mse loss as target
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train and save history for plotting and reuse
    history = model.fit(
        X_train, y_train_scaled,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val_scaled),
        verbose=1,
        callbacks=[
            EarlyStopping(patience=6, min_delta=0.001, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=3, factor=0.5, min_delta=0.001, verbose=1)
        ]

    )
    plot_loss(FIG_DIR, history, region)

    # Predict and inverse transform for comparisons between val and test
    # y_pred_scaled = model.predict(X_test)
    y_pred_scaled = model.predict(X_val)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Flatten both for metrics
    # y_true = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else y_test.ravel()
    y_true = y_val.values.ravel() if isinstance(y_val, pd.DataFrame) else y_val.ravel()
    y_pred = y_pred.ravel()

    # MAPE safe calculation
    mask = y_true != 0
    percent_error = (y_pred[mask] - y_true[mask]) / y_true[mask]

    # Metrics calculations
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(percent_error)) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2: {r2:.4f}")

    model.save(DATA_DIR / f"{region}_lstm_model.h5")

    return model, rmse, mae, mape, r2

def plot_loss(FIG_DIR, history, region):
    """Plots training versus validation loss based on history file and saves in FIG_DIR"""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{region} LSTM Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIG_DIR / f"{region}_loss_plot.png")  # Save the plot
    plt.close()

def run_regional_lstm(DATA_DIR: Path, FIG_DIR: Path, regions: list, seq_len: int = 24):
    """Runs RNN training process for each region and given sequence length by importing
    the train/val/test sets, scaling the data and running it through the model building helper"""
    train_start = time.time()

    results = []
    for region in regions:
        print(f"\n--- Running LSTM for {region} ---")
        df_train, df_val, df_test = load_final_data(DATA_DIR, region)

        # Drop label cols and assign target
        drop_cols = ['period', 'respondent', 'respondent-name', 'Region']
        target_col = 'Total interchange'
        df_train = df_train.drop(drop_cols,axis=1)
        df_val = df_val.drop(drop_cols,axis=1)
        df_test = df_test.drop(drop_cols,axis=1)

        # Scale and build sequences
        df_train_scaled, df_val_scaled, df_test_scaled, scaler = normalize_features(df_train, df_val, df_test, target_col)
        feature_cols = [col for col in df_train_scaled.columns if col != target_col]
        X_train, y_train = build_sequences(df_train_scaled, feature_cols, target_col, seq_len)
        X_val, y_val = build_sequences(df_val_scaled, feature_cols, target_col, seq_len)
        X_test, y_test = build_sequences(df_test_scaled, feature_cols, target_col, seq_len)

        # Train and evaluate results
        model, rmse, mae, mape, r2 = train_evaluate_lstm(DATA_DIR, FIG_DIR, X_train, y_train, X_val, y_val, X_test, y_test, region)

        results.append({
            'Region': region,
            'Model': model,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

    results_df = pd.DataFrame(results)
    out_path = DATA_DIR / f"LSTM_train_results.csv"

    if out_path.exists():
        print(f"Skipping {out_path.name} (already exists)")

    else:
        results_df.to_csv(out_path, index=False)
        print(f"Saved: LSTM__train_results.csv")

    train_end = time.time()

    print(f"Total training time for all selected regions: {train_end - train_start} seconds")

    return results_df


def main():
    # Setting seeds for the RNN reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

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

    # regions = ['MIDW']

    results = run_regional_lstm(DATA_DIR, FIG_DIR, regions, seq_len=24)
    print(results)

if __name__ == "__main__":
    main()