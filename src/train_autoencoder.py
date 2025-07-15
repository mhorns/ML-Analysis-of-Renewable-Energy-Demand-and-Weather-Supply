from pathlib import Path
import numpy as np
import random
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_autoencoder(input_shape, latent_dim=64, dropout_rate=0.2):
    """Returns autoencoder and encoder models with dropout and batch normalization"""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)

    x = layers.RepeatVector(input_shape[0])(latent)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.TimeDistributed(layers.Dense(input_shape[-1]))(x)

    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)
    return autoencoder, encoder

def train_autoencoder(DATA_DIR: Path, region: str, latent_dim=64, dropout_rate=0.2):
    print(f"\nTraining autoencoder for region: {region}")
    X_train = np.load(DATA_DIR / f"rnn_data_X_train_{region}.npy")
    X_val = np.load(DATA_DIR / f"rnn_data_X_val_{region}.npy")

    autoencoder, encoder = build_autoencoder(X_train.shape[1:], latent_dim=latent_dim, dropout_rate=dropout_rate)
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=6, min_delta=0.001, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_delta=0.001, verbose=1)
        ]
    )

    # Save encoder and autoencoder
    autoencoder.save(DATA_DIR / f"{region}_autoencoder.h5")
    encoder.save(DATA_DIR / f"{region}_encoder.h5")

    print("Saved encoder and autoencoder models.")
    return encoder

def main():
    # Set seeds for reproducibility
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


    # regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']
    regions = ['MIDW', 'NW', 'NY']

    for region in regions:
        train_autoencoder(DATA_DIR, region)

if __name__ == "__main__":
    main()
