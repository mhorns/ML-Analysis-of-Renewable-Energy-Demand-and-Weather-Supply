import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import tensorflow as tf

def evaluate_autoencoder(region="TEX", base_dir="../data"):
    DATA_DIR = Path(base_dir)
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
        plt.title(f"Validation Sample {i}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Error histogram
    errors = [mean_squared_error(x_true, x_pred) for x_true, x_pred in zip(X_val, decoded)]
    plt.figure(figsize=(8, 4))
    sns.histplot(errors, bins=50)
    plt.title("Validation Reconstruction Error Distribution")
    plt.xlabel("MSE")
    plt.grid(True)
    plt.show()
    '''
    # t-SNE of latent space
    latent = encoder.predict(X_val)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    latent_2d = tsne.fit_transform(latent)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
    plt.title("t-SNE of Latent Representations")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.show()
    '''

if __name__ == "__main__":
    evaluate_autoencoder(region="TEX", base_dir="../data")
