import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
import tensorflow as tf


def evaluate_autoencoder(region: str, DATA_DIR: Path, FIG_DIR: Path):
    """
    Evaluate a trained autoencoder by visualizing reconstruction quality and error distribution.  Loads the saved
    autoencoder and encoder models for a specific region. Uses the autoencoder to reconstruct the input validation
    sequences.  Plots a few examples comparing the original vs reconstructed sequences.
    :param region: Region identifier string (used to load model and data files).
    :param DATA_DIR: Path to the directory containing model and validation data files.
    :param FIG_DIR: Path to the directory where the output plots will be saved.
    """

    autoencoder = tf.keras.models.load_model(DATA_DIR / f"{region}_autoencoder.h5")
    encoder = tf.keras.models.load_model(DATA_DIR / f"{region}_encoder.h5")
    X_val = np.load(DATA_DIR / f"rnn_data_X_val_{region}.npy")

    # Reconstruct
    decoded = autoencoder.predict(X_val)

    # Plot reconstruction examples
    for i in range(3):
        plt.figure(figsize=(12, 3))
        plt.plot(X_val[i].flatten(), label='Original', alpha=0.7)
        plt.plot(decoded[i].flatten(), label='Reconstructed', alpha=0.7)
        plt.title(f"Validation Sample {region} {i}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"Reconstruction_Validation_Sample_{region}_{i}.png")
        plt.close()

    # Error histogram
    errors = [mean_squared_error(x_true, x_pred) for x_true, x_pred in zip(X_val, decoded)]
    plt.figure(figsize=(8, 4))
    sns.histplot(errors, bins=50)
    plt.title(f"Validation Reconstruction Error Distribution {region}")
    plt.xlabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"Validation_Reconstruction_Error_Distribution_{region}.png")
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

    completed_regions = []

    regions = ['MIDW', 'NW', 'NY', 'SE', 'NE', 'MIDA', 'CENT', 'SW', 'CAR', 'CAL', 'FLA', 'TEN', 'TEX']

    for region in regions:
        evaluate_autoencoder(region, DATA_DIR, FIG_DIR)

if __name__ == "__main__":
    main()
