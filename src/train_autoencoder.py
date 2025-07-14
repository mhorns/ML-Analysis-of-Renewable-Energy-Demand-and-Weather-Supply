from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_autoencoder(input_shape, latent_dim=32):
    """Returns autoencoder and encoder models"""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=False)(inputs)
    latent = layers.Dense(latent_dim, activation='relu')(x)

    x = layers.RepeatVector(input_shape[0])(latent)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(input_shape[-1]))(x)

    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)
    return autoencoder, encoder

def train_autoencoder(DATA_DIR: Path, region: str, latent_dim=32):
    print(f"\nTraining autoencoder for region: {region}")
    X_train = np.load(DATA_DIR / f"rnn_data_X_train_{region}.npy")
    X_val = np.load(DATA_DIR / f"rnn_data_X_val_{region}.npy")

    autoencoder, encoder = build_autoencoder(X_train.shape[1:], latent_dim=latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=3)
        ]
    )

    # Save encoder and autoencoder
    autoencoder.save(DATA_DIR / f"{region}_autoencoder.h5")
    encoder.save(DATA_DIR / f"{region}_encoder.h5")

    print("Saved encoder and autoencoder models.")
    return encoder

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    regions = ['MIDW', 'SE', 'NE', 'MIDA', 'NW', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'NY', 'TEN', 'TEX']

    for region in regions:
        train_autoencoder(DATA_DIR, region)

if __name__ == "__main__":
    main()
