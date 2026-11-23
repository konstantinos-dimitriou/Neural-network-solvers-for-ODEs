import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Settings
SEED = 123

HIDDEN_LAYERS = (128, 128, 128)
ACTIVATION = "tanh"
SOLVER = "adam"
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 500
N_ITER_NO_CHANGE = 20      # early stopping patience (epochs)
TOL = 1e-6                 # minimum improvement of the internal validation score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "figures"


# Data loading
def load_data(filepath):
    """
    Load an .npz file produced by generate_data.py.

    Returns:
    sampled_parameters : (n, 4)      columns A, B, x0, y0   (network input)
    ODE_data           : (n, 2*N_T)  x(t) block, then y(t) block (network target)
    """
    data = np.load(filepath)
    return data["sampled_parameters"], data["ODE_data"]


# Metrics
def relative_l2_error(ODE_true, ODE_pred):
    """Relative L2 error per sample over the full (x, y) vector, in original units."""
    return np.linalg.norm(ODE_pred - ODE_true, axis=1) / np.linalg.norm(ODE_true, axis=1)


def summarize_errors(sampled_parameters, ODE_true, ODE_pred):
    """Mean / median / max relative L2 error: all samples, stable, oscillating."""
    errors = relative_l2_error(ODE_true, ODE_pred)
    # Hopf line: oscillating regime for B > 1 + A^2
    oscillating = sampled_parameters[:, 1] > 1.0 + sampled_parameters[:, 0] ** 2

    summary = {}
    for name, mask in [("all", np.ones_like(oscillating)),
                       ("stable", ~oscillating),
                       ("oscillating", oscillating)]:
        summary[name] = {
            "n": int(mask.sum()),
            "mean": float(errors[mask].mean()),
            "median": float(np.median(errors[mask])),
            "max": float(errors[mask].max()),
        }
    return summary


def print_summary(title, summary):
    print(f"\n{title}  (relative L2 error)")
    for name, s in summary.items():
        print(f"  {name:<12s} n={s['n']:>5d} | mean {s['mean']:.4e} | "
              f"median {s['median']:.4e} | max {s['max']:.4e}")


# Plotting
def plot_loss_curves(mlp, save_path):
    """Training loss (MSE in scaled space) and internal validation score per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(mlp.loss_curve_, color="steelblue", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss (scaled MSE / 2)")
    axes[0].set_title("Training loss per epoch")

    axes[1].plot(mlp.validation_scores_, color="darkorange", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Internal validation score (R²)")
    axes[1].set_title("Early-stopping validation score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to '{save_path}'")


# Model
def build_model(n_t):
    """
    Input scaling and output scaling are both handled here.

    SCALING:
    - Inputs  (A, B, x0, y0): StandardScaler inside the Pipeline.
    - Targets (x-block, y-block): BlockStandardScaler via TransformedTargetRegressor.
      TransformedTargetRegressor scales the targets before training and applies the
      inverse transform in predict(), so model.predict(...) returns values in the
      ORIGINAL units (directly comparable with Runge-Kutta). The MSE training loss
      is computed on the scaled targets.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation=ACTIVATION,
        solver=SOLVER,
        learning_rate_init=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_iter=MAX_EPOCHS,
        early_stopping=True,            # holds out validation_fraction of the TRAINING data
        validation_fraction=0.1,
        n_iter_no_change=N_ITER_NO_CHANGE,
        tol=TOL,
        random_state=SEED,
    )
    regressor = Pipeline([
        ("input_scaler", StandardScaler()),
        ("mlp", mlp),
    ])
    return TransformedTargetRegressor(
        regressor=regressor,
        #transformer=BlockStandardScaler(n_t=n_t),   # <-- target scaling
        transformer=StandardScaler(),   # <-- target scaling
    )


def main():
    # Load data (the test set is not used here, only in the evaluation step)
    print("Loading data...")
    train_parameters, train_ODE_data = load_data(DATA_DIR / "train.npz")
    val_parameters, val_ODE_data = load_data(DATA_DIR / "validation.npz")
    n_t = train_ODE_data.shape[1] // 2
    print(f"  Training samples   : {len(train_parameters)}")
    print(f"  Validation samples : {len(val_parameters)}")
    print(f"  Outputs per sample : {train_ODE_data.shape[1]} (2 x {n_t})")

    # Baseline: always predict the mean training trajectory, ignoring the inputs
    baseline_pred = np.tile(train_ODE_data.mean(axis=0), (len(val_parameters), 1))
    baseline_summary = summarize_errors(val_parameters, val_ODE_data, baseline_pred)
    print_summary("Baseline (mean trajectory), validation", baseline_summary)

    # Define model
    model = build_model(n_t)
    print(f"\nHidden layers: {HIDDEN_LAYERS} | activation: {ACTIVATION} | solver: {SOLVER}")
    print(f"Batch size: {BATCH_SIZE} | lr: {LEARNING_RATE} | max epochs: {MAX_EPOCHS}\n")

    # Train
    print("Training...")
    model.fit(train_parameters, train_ODE_data)
    mlp = model.regressor_.named_steps["mlp"]
    print(f"Stopped after {mlp.n_iter_} epochs "
          f"(best internal validation score: {mlp.best_validation_score_:.6f})")

    # Evaluate (predictions are already in original units)
    train_summary = summarize_errors(train_parameters, train_ODE_data,
                                     model.predict(train_parameters))
    val_summary = summarize_errors(val_parameters, val_ODE_data,
                                   model.predict(val_parameters))
    print_summary("Training set", train_summary)
    print_summary("Validation set", val_summary)

    # Save plots
    FIG_DIR.mkdir(exist_ok=True)
    plot_loss_curves(mlp, FIG_DIR / "training_loss_curves.png")

    # Save model (including scalers) and metrics
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "brusselator_mlp.joblib")
    metrics = {
        "settings": {
            "hidden_layers": HIDDEN_LAYERS, "activation": ACTIVATION, "solver": SOLVER,
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "epochs_run": int(mlp.n_iter_), "seed": SEED,
        },
        "baseline_validation": baseline_summary,
        "train": train_summary,
        "validation": val_summary,
    }
    with open(MODEL_DIR / "brusselator_mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model and metrics saved to '{MODEL_DIR}/'")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = (time.time() - start) / 60
    print(f"\nTotal computation time: {elapsed:.2f} minutes.")