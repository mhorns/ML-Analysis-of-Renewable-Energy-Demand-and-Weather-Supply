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
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def train_evaluate_rnn(DATA_DIR, FIG_DIR, X_train, y_train, X_val, y_val, X_test, y_test,
                       region, rnn_type='LSTM', units_per_layer=[128, 64, 32],
                       dropout=0.2, batch_size=64, epochs=20, loss='mse', optimizer='adam', l2_lambda=0.0):
    """Trains an LSTM or GRU model with flexible architecture."""
    train_start = time.time()
    # Scale target using standard scaler to perform preds with zero mean and unit variance
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Build model with variable type and unit structure
    model = Sequential()
    RNN = LSTM if rnn_type.upper() == 'LSTM' else GRU

    for i, units in enumerate(units_per_layer):
        return_seq = i < len(units_per_layer) - 1  # only last RNN layer should not return sequences
        if i == 0:
            model.add(RNN(units,
                          return_sequences=return_seq,
                          input_shape=(X_train.shape[1], X_train.shape[2]),
                          kernel_regularizer=regularizers.l2(l2_lambda)))
        else:
            model.add(RNN(units,
                          return_sequences=return_seq,
                          kernel_regularizer=regularizers.l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)

    # Train model using validation data, early stopping and reduce learning rate on plateau
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[
            EarlyStopping(patience=6, min_delta=0.001, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=3, factor=0.5, min_delta=0.001, verbose=1)
        ]
    )

    # Plot loss over epochs using model history
    plot_loss(FIG_DIR, history, region, rnn_type, units_per_layer)

    # Evaluate model based on y scale inverse on validation data
    y_pred_scaled = model.predict(X_val)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    y_true = y_val.values.ravel() if isinstance(y_val, pd.DataFrame) else y_val.ravel()
    y_pred = y_pred.ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")

    # Save model for future use
    out_path = DATA_DIR / f"{region}_{rnn_type.lower()}_{len(units_per_layer)}unit_model.h5"
    if out_path.exists():
        print(f"Skipping {out_path.name} (already exists)")
    else:
        model.save(out_path)
        print(f"Saved: {out_path.name}")

    train_end = time.time()

    print(f"Total time for training region: {train_end - train_start} seconds")

    return model, rmse, mae, mape, r2

def train_evaluate_latent_forecaster(DATA_DIR, FIG_DIR, X_train, y_train, X_val, y_val, X_test, y_test,
                                     region, encoder, latent_dim=32, batch_size=64, epochs=20, loss='mse',
                                     optimizer='adam', l2_lambda=0.0):
    """Trains a dense forecast model using latent inputs from an encoder"""
    train_start = time.time()
    # Encode inputs
    X_train_latent = encoder.predict(X_train)
    X_val_latent = encoder.predict(X_val)
    X_test_latent = encoder.predict(X_test)

    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    # Build forecast model from latent vector
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_lambda)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_lambda)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_lambda)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss=loss)

    # Train
    history = model.fit(
        X_train_latent, y_train_scaled,
        validation_data=(X_val_latent, y_val_scaled),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            EarlyStopping(patience=6, min_delta=0.001, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=3, factor=0.5, min_delta=0.001, verbose=1)
        ]
    )

    plot_loss(FIG_DIR, history, region, f"AELatent_{latent_dim}", [latent_dim])

    # Evaluate on validation
    y_pred_scaled = model.predict(X_val_latent)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    y_true = y_val.ravel()
    y_pred = y_pred.ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"[Autoencoder Forecast] RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")

    model.save(DATA_DIR / f"{region}_ae_latent_forecast_model.h5")

    train_end = time.time()

    print(f"Total time for training region: {train_end - train_start} seconds")

    return model, rmse, mae, mape, r2


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

def plot_loss(FIG_DIR, history, region, rnn_type, units_per_layer):
    """Plots training versus validation loss based on history file and saves in FIG_DIR"""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{region} {rnn_type.upper()} {len(units_per_layer)} Unit Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIG_DIR / f"{region}_{rnn_type.lower()}_{len(units_per_layer)}unit_loss_plot.png")
    plt.close()

def run_regional_rnn(DATA_DIR: Path, FIG_DIR: Path, regions: list, use_autoencoder=False, rnn_type='LSTM',
                     units_per_layer=[128, 64, 32], dropout=0.2, batch_size=32, epochs=20, loss='mse',
                     optimizer='adam', l2_lambda=0.0):
    """Runs RNN training process for each region by importing the scaled train/val/test sets and scaler, and
    running it through the model building train and evaluate helper"""
    train_start = time.time()

    results = []
    for region in tqdm(regions):
        X_train = np.load(DATA_DIR / f"rnn_data_X_train_{region}.npy")
        X_val = np.load(DATA_DIR / f"rnn_data_X_val_{region}.npy")
        X_test = np.load(DATA_DIR / f"rnn_data_X_test_{region}.npy")
        y_train = np.load(DATA_DIR / f"rnn_data_y_train_{region}.npy")
        y_val = np.load(DATA_DIR / f"rnn_data_y_val_{region}.npy")
        y_test = np.load(DATA_DIR / f"rnn_data_y_test_{region}.npy")
        scaler = joblib.load(DATA_DIR / f"rnn_scaler_{region}.joblib")

        # Train and evaluate results for each switch type
        if use_autoencoder:
            encoder = tf.keras.models.load_model(DATA_DIR / f"{region}_encoder.h5")
            latent_dim = encoder.output_shape[-1]
            model, rmse, mae, mape, r2 = train_evaluate_latent_forecaster(
                DATA_DIR, FIG_DIR, X_train, y_train, X_val, y_val, X_test, y_test, region, encoder,
                latent_dim=latent_dim, batch_size=batch_size, epochs=epochs, loss=loss, optimizer=optimizer,
                l2_lambda=l2_lambda
            )
            model_type = f"Autoencoder Forecast (latent_dim={latent_dim})"

        else:
            model, rmse, mae, mape, r2 = train_evaluate_rnn(
                DATA_DIR, FIG_DIR, X_train, y_train, X_val, y_val, X_test, y_test,
                region, rnn_type=rnn_type, units_per_layer=units_per_layer,
                dropout=dropout, batch_size=batch_size, epochs=epochs, loss=loss, optimizer=optimizer,
                l2_lambda=l2_lambda
            )
            model_type = f"{rnn_type} | layers={units_per_layer}"

        results.append({
            'Region': region,
            'Model': model_type,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

    # Save full model results per region
    df = pd.DataFrame(results)
    out_name = f"{'ae' if use_autoencoder else rnn_type.lower()}_{len(units_per_layer)}unit_train_results.csv"
    df.to_csv(DATA_DIR / out_name, index=False)
    print(f"Saved: {out_name}")

    train_end = time.time()

    print(f"Total training time for all selected regions: {train_end - train_start} seconds")

    return df


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
    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']
    regions = ['MIDW', 'NW', 'NY']

    # Run both models for comparison
    '''print("Running Base LSTM Model")
    run_regional_rnn(DATA_DIR, FIG_DIR, regions, use_autoencoder=False,
                            rnn_type='LSTM', units_per_layer=[128, 64, 32], epochs=30)'''

    print("Running Autoencoder Forecast Model")
    run_regional_rnn(DATA_DIR, FIG_DIR, regions, use_autoencoder=True, epochs=30, l2_lambda=0.001)


if __name__ == "__main__":
    main()