# Author: Konstantinos Dimitriou
# Date: 10/6/2026

import numpy as np
import matplotlib.pyplot as plt
import neural_network as nn
import os

def logistic_exact(t, u0, r):
    return (u0 * np.exp(r * t)) / ((1 - u0) + u0 * np.exp(r * t))


def load_network(model_path):
    """
    Load a trained NeuralNetwork from disk. Call this once.

    Parameters:
    model_path : str, path to the .npz model file

    Returns:
    NeuralNetwork instance
    """
    return nn.NeuralNetwork.load(model_path)


def solve(net, u0, r):
    """
    Use a loaded network to predict the solution vector u(t) for a
    given (u0, r) pair. Returns one value per time point, matching
    the grid the network was trained on (t0, ..., t99).

    Parameters:
    net : NeuralNetwork, already loaded via load_network()
    u0  : float, initial condition (0 < u0 < 1)
    r   : float, growth rate      (0 < r  < 10)

    Returns:
    np.ndarray, shape (100,) — predicted solution vector
    """
    x = np.array([[u0], [r]], dtype=np.float64)   # (2, 1) input vector
    return net.predict(x).flatten()               # (100,) output vector


def main():
    t_values   = np.linspace(0, 1, 100)
    model_path = "TRAINED_NN/trained_model_logistic_ode.npz"

    # Load the network once
    net = load_network(model_path)

    # Define the (u0, r) pairs you want to evaluate
    cases = [
        (0.3, 2.5),
        (0.1, 5.0),
        (0.7, 1.0),
        (0.5, 8.0),
        (0.2, 3.3),
        (0.9, 0.5),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, (u0, r) in zip(axes, cases):
        # Predict solution vector (100 values, one per t[i])
        y_pred  = solve(net, u0, r)
        y_exact = logistic_exact(t_values, u0, r)

        mse = np.mean((y_pred - y_exact) ** 2)
        print(f"u0={u0:.2f}, r={r:.2f} | MSE: {mse:.6f}")

        ax.plot(t_values, y_exact, label="Exact",     linewidth=2)
        ax.plot(t_values, y_pred,  label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_title(f"u₀={u0}, r={r}  (MSE={mse:.2e})")
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    plt.suptitle("Neural network surrogate vs exact logistic solution")
    plt.tight_layout()

    # Save plot
    os.makedirs("PLOTS", exist_ok=True)
    plt.savefig("PLOTS/evaluation.png", dpi=150)
    plt.show()
    print("\nPlot saved to 'PLOTS/evaluation.png'")


if __name__ == "__main__":
    main()