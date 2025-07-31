from pathlib import Path
import numpy as np
import random
import json
import time
import joblib
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras_tuner import Hyperband

def build_autoencoder(input_shape: tuple, latent_dim=128, dropout_rate=0.2, l2_lambda=0.0):
    """
    Returns autoencoder and encoder models with dropout and batch normalization
    :param input_shape: Shape of the input data (timesteps, features).
    :param latent_dim: Dimensionality of the latent (bottleneck) layer.
    :param dropout_rate: Dropout rate applied after batch normalization.
    :param l2_lambda: L2 regularization strength for kernel weights.
    :return: Tuple of (autoencoder, encoder) Keras models.
    """

    inputs = layers.Input(shape=input_shape)

    # Encoder stacked layers
    x = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Latent layer
    latent = layers.Dense(latent_dim, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_lambda))(x)
    latent = layers.BatchNormalization()(latent)

    # Decoder unstacking layers
    x = layers.RepeatVector(input_shape[0])(latent)
    x = layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)

    # Output layer
    outputs = layers.TimeDistributed(layers.Dense(input_shape[-1],
                                     kernel_regularizer=regularizers.l2(l2_lambda)))(x)

    autoencoder = models.Model(inputs, outputs)
    encoder = models.Model(inputs, latent)
    return autoencoder, encoder

def train_autoencoder(DATA_DIR: Path, region: str, latent_dim=128, dropout_rate=0.2, l2_lambda=0.0):
    """
    Trains and saves an autoencoder and its encoder for the specified region.
    :param DATA_DIR: Directory where training data is stored and models will be saved.
    :param region: Region name to load data and label saved models.
    :param latent_dim: Dimensionality of the latent space.
    :param dropout_rate: Dropout applied in encoder/decoder layers.
    :param l2_lambda: L2 regularization strength for recurrent and dense layers.
    :return: Trained encoder model.
    """

    train_start = time.time()
    print(f"\nTraining autoencoder for region: {region}")
    X_train = np.load(DATA_DIR / f"rnn_data_X_train_{region}.npy")
    X_val = np.load(DATA_DIR / f"rnn_data_X_val_{region}.npy")

    autoencoder, encoder = build_autoencoder(X_train.shape[1:],
                                             latent_dim=latent_dim,
                                             dropout_rate=dropout_rate,
                                             l2_lambda=l2_lambda)
    autoencoder.compile(optimizer='adam', loss='mae')

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

    train_end = time.time()

    print(f"Total time for training region: {train_end - train_start} seconds")
    return encoder

def load_best_hyperparams(json_path: Path):
    """
    Load the best hyperparameters from a JSON file.
    :param json_path: Path to the JSON file containing the saved hyperparameter dictionary.
    :return: Dictionary of hyperparameters.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def build_autoencoder_tuner(hp, input_shape: tuple, latent_dim_params: list, dropout_params: list, l2_params: list):
    """
    Build and compile an LSTM-based autoencoder using hyperparameter tuning.
    :param hp: Keras Tuner hyperparameter search object.
    :param input_shape: Tuple indicating input shape as (timesteps, features).
    :param latent_dim_params: List of candidate values for latent dimension size.
    :param dropout_params: List of candidate dropout rates.
    :param l2_params: List of candidate L2 regularization strengths.
    :return: Compiled Keras autoencoder model.
    """

    latent_dim = hp.Choice("latent_dim", latent_dim_params)
    dropout_rate = hp.Choice("dropout_rate", dropout_params)
    l2_lambda = hp.Choice("l2_lambda", l2_params)

    inputs = layers.Input(shape=input_shape)

    x = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(64, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    latent = layers.Dense(latent_dim, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_lambda),
                          name="latent_layer")(x)
    x = layers.BatchNormalization()(latent)

    x = layers.RepeatVector(input_shape[0])(x)
    x = layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.TimeDistributed(layers.Dense(input_shape[-1],
                                                  kernel_regularizer=regularizers.l2(l2_lambda)))(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mae')
    return model

def train_autoencoder_tuner(DATA_DIR: Path, region: str, latent_dim_params: list, dropout_params: list, l2_params: list):
    """
    Run hyperparameter tuning for an LSTM-based autoencoder and save best models and config.

    :param DATA_DIR: Directory path for loading/saving model files and tuning logs.
    :param region: Region identifier string to use for loading and saving.
    :param latent_dim_params: List of candidate values for latent dimension size.
    :param dropout_params: List of candidate dropout rates.
    :param l2_params: List of candidate L2 regularization strengths.
    :return: Trained encoder model with the best hyperparameters.
    """
    print(f"\nTuning autoencoder for region: {region}")
    X_train = np.load(DATA_DIR / f"rnn_data_X_train_{region}.npy")
    X_val = np.load(DATA_DIR / f"rnn_data_X_val_{region}.npy")

    tuner = Hyperband(
        lambda hp: build_autoencoder_tuner(hp, X_train.shape[1:],
                                           latent_dim_params=latent_dim_params,
                                           dropout_params=dropout_params,
                                           l2_params=l2_params),
        objective="val_loss",
        max_epochs=20,
        factor=4,
        directory=DATA_DIR,
        project_name=f"ae_tuning_{region}",
        overwrite=True  # True for new training round, False to continue
    )

    stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(
        X_train, X_train,
        validation_data=(X_val, X_val),
        batch_size=32,
        callbacks=[stop_early],
        verbose=2
    )

    best_model = tuner.get_best_models(1)[0]
    encoder = models.Model(inputs=best_model.input,
                           outputs=best_model.get_layer("latent_layer").output)

    best_model.save(DATA_DIR / f"{region}_autoencoder.h5")
    encoder.save(DATA_DIR / f"{region}_encoder.h5")

    # Append best hyperparameters to shared JSON
    best_hp = tuner.get_best_hyperparameters(1)[0].values
    best_hp_path = DATA_DIR / "all_best_hyperparams.json"

    if best_hp_path.exists():
        with open(best_hp_path) as f:
            all_hp = json.load(f)
    else:
        all_hp = {}

    all_hp[region] = best_hp
    with open(best_hp_path, "w") as f:
        json.dump(all_hp, f, indent=2)

    print(f"Saved models and best hyperparameters for region {region}.")

    return encoder

def main():
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)

    FIG_DIR = BASE_DIR / "figs"
    FIG_DIR.mkdir(exist_ok=True)

    # Path for hyperparameter json file
    hp_path = DATA_DIR / "all_best_hyperparams.json"
    print(f"HP Directory: {hp_path}")
    best_hp_dict = load_best_hyperparams(hp_path)

    completed_regions = ['SE', 'NE', 'MIDA', 'MIDW', 'NY', 'CENT', 'NW', 'SW', 'CAR', 'CAL', 'FLA', 'TEN', 'TEX']
    regions = ['CENT']
    regions_left = []

    latent_dim_params = [64, 128, 256]
    dropout_params = [0.1, 0.2]
    l2_params = [0.0, 0.001, 0.01]

    for region in regions:  # The Keras tuner gets the best parameters but for some reason does not reconstruct correctly
        # Get best params from Keras tuner
        # train_autoencoder_tuner(DATA_DIR, region, latent_dim_params, dropout_params, l2_params)

        # Load saved tuned parameters to manually train
        print(f"\nTraining {region} with best tuner parameters")
        hp = best_hp_dict[region]
        print(f"latent_dim = {hp['latent_dim']}, dropout_rate = {hp['dropout_rate']}, l2_lambda = {hp['l2_lambda']}")

        # Train with best tuned parameters for correct reconstruction
        train_autoencoder(DATA_DIR, region,
                          latent_dim=hp['latent_dim'],
                          dropout_rate=hp['dropout_rate'],
                          l2_lambda=hp['l2_lambda']
                          )

if __name__ == "__main__":
    main()