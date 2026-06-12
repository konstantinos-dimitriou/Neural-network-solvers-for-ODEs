# Author: Konstantinos Dimitriou
# Date: 10/6/2026

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import neural_network as nn


def load_data(filepath):
    """
    Load a CSV produced by generate_data.py and return a list of
    (x, y) pairs suitable for the neural network.

    Each row in the CSV has columns: u0, r, t0, t1, ..., t99 (number of time grids currently hard codded in at 100)
    - x (input)  : column vector of shape (2, 1)  → [u0, r]
    - y (target) : column vector of shape (100, 1) → solution values u(t0)..u(t99)

    Parameters:
    filepath : str

    Returns:
    list of (np.ndarray, np.ndarray) pairs
    """
    df = pd.read_csv(filepath)

    inputs  = df[["u0", "r"]].values           # shape (N, 2)
    targets = df.filter(like="t").values        # shape (N, 100)

    data = []
    for x, y in zip(inputs, targets):
        x_col = x.reshape(-1, 1).astype(np.float64)   # (2, 1)
        y_col = y.reshape(-1, 1).astype(np.float64)   # (100, 1)
        data.append((x_col, y_col))

    return data


# Plotting
def plot_loss_curve(loss_history, save_path):
    """Plot and save the test MSE per epoch."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color="steelblue", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE")
    plt.title("Test MSE per epoch")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to '{save_path}'")


def plot_predictions(net, test_data, t_values, n_samples=6,
                     save_path="predictions.png"):
    """
    Plot predicted vs exact solution curves for a few test samples.

    Parameters:
    net       : NeuralNetwork
    test_data : list of (x, y) pairs
    t_values  : np.ndarray, the shared time grid
    n_samples : int, how many samples to show
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    indices = np.random.choice(len(test_data), size=n_samples, replace=False)

    for ax, idx in zip(axes, indices):
        x, y_exact = test_data[idx]
        y_pred = net.predict(x).flatten()
        u0, r  = x.flatten()

        ax.plot(t_values, y_exact.flatten(), label="Exact",     linewidth=2)
        ax.plot(t_values, y_pred,            label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_title(f"u₀={u0:.2f}, r={r:.2f}")
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Predicted vs Exact logistic solutions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to '{save_path}'")



# Custom SGD with loss tracking
def train_with_tracking(net, training_data, test_data,
                         epochs, mini_batch_size, learning_rate):
    """
    Thin wrapper around net.SGD that also records test MSE each epoch
    and returns the loss history for plotting.

    Returns:
    list of float : test MSE per epoch
    """
    loss_history = []
    n = len(training_data)

    for epoch in range(epochs):
        np.random.shuffle(training_data)

        mini_batches = [
            training_data[k : k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]

        for mini_batch in mini_batches:
            net._update_mini_batch(mini_batch, learning_rate)
            #net._update_mini_batch(mini_batch, current_lr)

        mse = net.evaluate_mse(test_data)
        loss_history.append(mse)
        print(f"Epoch {epoch:>3d} | test MSE: {mse:.6f}")

    return loss_history


def main():
    np.random.seed(17)

    # Load data
    print("Loading data...")
    training_data = load_data("DATA/training_data.csv")
    test_data     = load_data("DATA/testing_data.csv")
    print(f"  Training samples : {len(training_data)}")
    print(f"  Test samples     : {len(test_data)}")

    # Time grid (must match what generate_data.py used)
    t_values = np.linspace(0, 1, 100)

    # Define network
    # Input:  (u0, r)         → 2 neurons
    # Output: u(t0)...u(t99)  → 100 neurons
    layer_sizes = (2, 128, 128, 100)    # Best for sigmoid
    layer_sizes = (2, 256, 128, 100)    # Best for tanh
    print(f"\nNetwork architecture: {layer_sizes}")

    #activation_function = "tanh"
    activation_function = "sigmoid"
    
    net = nn.NeuralNetwork(layer_sizes, activation=activation_function)

    # Hyperparameters
    epochs          = 100   # Best for sigmoid
    #epochs          = 200   # Best for tanh

    mini_batch_size = 32    # or 64
    
    #learning_rate   = 0.005 # Best for tanh
    learning_rate   = 0.05   # Best for sigmoid

    print(f"Epochs: {epochs}  |  Mini-batch: {mini_batch_size}  |  lr: {learning_rate} | Actiavtion: {activation_function}\n")

    # Train
    loss_history = train_with_tracking(
        net, training_data, test_data,
        epochs, mini_batch_size, learning_rate
    )

    # Final evaluation
    train_mse = net.evaluate_mse(training_data)
    test_mse  = net.evaluate_mse(test_data)
    print(f"\nFinal train MSE : {train_mse:.6f}")
    print(f"Final test  MSE : {test_mse:.6f}")

    # Save plots
    os.makedirs("PLOTS", exist_ok=True)
    plot_loss_curve(loss_history, save_path="PLOTS/loss_curve.png")
    plot_predictions(net, test_data, t_values, save_path="PLOTS/predictions.png")

    # Save network
    os.makedirs("TRAINED_NN", exist_ok=True)
    net.save("TRAINED_NN/trained_model_logistic_ode.npz")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = (time.time() - start) / 60
    print(f"\nTotal computation time: {elapsed:.2f} minutes.")
