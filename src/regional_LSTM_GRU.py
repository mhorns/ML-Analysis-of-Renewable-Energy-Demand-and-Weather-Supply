import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import os
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_final_data(DATA_DIR: Path, region: str):
    """Gets the finalized per region energy and weather data"""
    df = pd.read_csv(DATA_DIR / f"final_data_{region}.csv", parse_dates=["period"])

    return df



def build_sequences(df, feature_cols, target_col, window_size):
    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[feature_cols].iloc[i-window_size:i].values)
        y.append(df[target_col].iloc[i])
    return np.array(X), np.array(y)





def normalize_features(train_df, test_df, exclude_cols=None):
    """
    Fits a StandardScaler on training data, applies to both train and test sets.
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, test_scaled, scaler

'''def train_evaluate_lstm(X_train, y_train, X_test, y_test, region):
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=64,
              validation_split=0.1, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    y_pred = model.predict(X_test).flatten()
    y_true = y_test

    # Avoid division by zero in MAPE
    mask = y_true != 0
    percent_error = np.zeros_like(y_true, dtype=float)
    percent_error[mask] = (y_pred[mask] - y_true[mask]) / y_true[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs(percent_error[mask])) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R2: {r2}")


    return model, rmse, mae, mape, r2
'''

def train_evaluate_lstm(DATA_DIR, X_train, y_train, X_test, y_test, region):
    # --- Scale y ---
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))

    # --- Build LSTM model ---
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # --- Train ---
    model.fit(
        X_train, y_train_scaled,
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    # --- Predict and inverse transform ---
    # y_pred_scaled = model.predict(X_test)
    y_pred_scaled = model.predict(X_train)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # --- Flatten both for metrics ---
    # y_true = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else y_test.ravel()
    y_true = y_train.values.ravel() if isinstance(y_train, pd.DataFrame) else y_train.ravel()
    y_pred = y_pred.ravel()

    # --- MAPE-safe calculation ---
    mask = y_true != 0
    percent_error = (y_pred[mask] - y_true[mask]) / y_true[mask]

    # --- Metrics ---
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


def run_regional_lstm(DATA_DIR: Path, regions: list, split_date: str, seq_len: int = 24):
    train_start = time.time()

    results = []
    for region in regions:
        print(f"\n--- Running LSTM for {region} ---")
        df = load_final_data(DATA_DIR, region)

        # Drop label cols and assign target
        drop_cols = ['period', 'respondent', 'respondent-name', 'Region']
        target_col = 'Total interchange'
        df_train = df[df['period'] < split_date].drop(columns=drop_cols, errors='ignore')
        df_test = df[df['period'] >= split_date].drop(columns=drop_cols, errors='ignore')

        df_train_scaled, df_test_scaled, scaler = normalize_features(df_train, df_test, target_col)
        feature_cols = [col for col in df_train_scaled.columns if col != target_col]
        X_train, y_train = build_sequences(df_train_scaled, feature_cols, target_col, seq_len)
        X_test, y_test = build_sequences(df_test_scaled, feature_cols, target_col, seq_len)


        model, rmse, mae, mape, r2 = train_evaluate_lstm(DATA_DIR, X_train, y_train, X_train, y_train, region)

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
        print(f"Saved: LSTM_train_results.csv")

    train_end = time.time()

    print(f"Total training time for all selected regions: {train_end - train_start} seconds")

    return results_df


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

    # regions = ['MIDW']

    # Time window backward 5 years from 7/31/2024
    end = datetime(2024, 7, 31)
    start = datetime(2019, 8, 2)
    print(f"Start date: {start}")
    print(f"End date: {end}")

    split_date = '2023-08-01'  # Choosing to split after 4 years to give approx 80/20

    results = run_regional_lstm(DATA_DIR, regions, split_date, seq_len = 24)
    print(results.head())

if __name__ == "__main__":
    main()