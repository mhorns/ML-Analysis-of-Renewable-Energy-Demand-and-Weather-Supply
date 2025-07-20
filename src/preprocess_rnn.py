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

def scale_features(scaler, train_df, val_df, test_df, exclude_cols=None):
    """Fits a selected scaler on training data, applies to train/val/test sets"""
    if exclude_cols is None:
        exclude_cols = []

    # Do not scale cyclical or binary cols as they are already within bounds
    no_scale_cols = ['hour_sin',
                     'hour_cos',
                     'month_sin',
                     'month_cos',
                     'doy_sin',
                     'doy_cos',
                     'dow_sin',
                     'dow_cos',
                     'is_weekend']
    all_exclude_cols = list(exclude_cols) + no_scale_cols
    feature_cols = [col for col in train_df.columns if col not in all_exclude_cols]

    # Select scaler type
    if scaler == 'StandardScaler()':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler

def preprocess_RNN(DATA_DIR: Path, regions: list, seq_len: int = 24):
    train_start = time.time()

    for region in tqdm(regions):
        print(f"\n--- Preprocessing data to implement RNN for {region} ---")
        df_train, df_val, df_test = load_final_data(DATA_DIR, region)

        # Drop label cols and assign target
        # drop_cols = ['period', 'respondent', 'respondent-name', 'Region']
        drop_cols = ['period',
                     'respondent',
                     'respondent-name',
                     'Region',
                     'MO',
                     'DY',
                     'HR',
                     'day_of_year',
                     'day_of_week',
                     # 'lag_interchange_1h',
                     # 'lag_interchange_24h',
                     # 'interchange_roll_mean_3h',
                     # 'Net_generation_lag_1',
                     # 'Demand_30d_avg',
                     'Day-ahead demand forecast_30d_avg',
                     # 'Solar_30d_avg',
                     # 'Wind_30d_avg',
                     'Net generation_30d_avg',
                     'Total interchange_30d_avg',
                     'Pct_Solar_30d_avg',
                     'Pct_Wind_30d_avg',
                     'ALLSKY_SFC_SW_DWN_30d_avg',
                     'T2M_30d_avg',
                     'WSC_30d_avg'# ,
                     # 'unexpect_dem_diff_30d_avg'
                     ]
        target_col = ['Total interchange']
        df_train = df_train.drop(drop_cols,axis=1)
        df_val = df_val.drop(drop_cols,axis=1)
        df_test = df_test.drop(drop_cols,axis=1)

        # Scale features using either StandardScaler() or MinMaxScaler() and build sequences
        # MinMaxScaler is default as it bounds the range 0 to 1 and has worked better
        df_train_scaled, df_val_scaled, df_test_scaled, scaler = scale_features('MinMaxScaler()',
                                                                                df_train,
                                                                                df_val,
                                                                                df_test,
                                                                                target_col)
        print(f"shape of df_train: {df_train.shape}, df_train_scaled: {df_train_scaled.shape}")
        feature_cols = [col for col in df_train_scaled.columns if col not in target_col]
        print(f"feature cols: {feature_cols} \n and target_col: {target_col}")

        X_train, y_train = build_sequences(df_train_scaled, feature_cols, target_col, seq_len)
        X_val, y_val = build_sequences(df_val_scaled, feature_cols, target_col, seq_len)
        X_test, y_test = build_sequences(df_test_scaled, feature_cols, target_col, seq_len)

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaler": scaler
        }

        # Save split tensors and scaler for reuse in training
        for split_name, split_df in splits.items():
            if "scaler" in split_name:
                out_path = DATA_DIR / f"rnn_scaler_{region}.joblib"
                if out_path.exists():
                    print(f"Skipping {out_path.name} (already exists)")
                else:
                    joblib.dump(split_df, out_path)
                    print(f"Saved scaler: {out_path.name}")
            else:
                out_path = DATA_DIR / f"rnn_data_{split_name}_{region}.npy"
                if out_path.exists():
                    print(f"Skipping {out_path.name} (already exists)")
                else:
                    np.save(out_path, split_df)
                    print(f"Saved: {out_path.name}")


    train_end = time.time()

    print(f"Total time for all selected regions: {train_end - train_start} seconds")

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

    regions = ['MIDW', 'NW', 'NY']

    preprocess_RNN(DATA_DIR, regions, seq_len=24)

if __name__ == "__main__":
    main()