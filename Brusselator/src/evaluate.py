"""Evaluate one trained model (selected from MLflow) on the test set and make comparison plots."""

import argparse
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from . import config
from .dataset import dataset_path, load_dataset
from .metrics import print_summary, relative_l2_error, summarize_errors
from .ode import REGIMES, classify_regimes, solve_brusselator


# Settings specific to the evaluation
N_SAMPLES_PER_FIGURE = 3      # test samples shown per regime (one column each)
N_TIMING = 500                # test samples used for the timing / accuracy table

# Other solvers at SciPy's default tolerances
RTOL_DEFAULT = 1e-3
ATOL_DEFAULT = 1e-6
OTHER_SOLVERS = {
    "RK45": dict(method="RK45", rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT),
    "Radau": dict(method="Radau", rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT),
}

STYLES = {
    "Ground truth":   dict(color="black", linestyle="-", linewidth=2.5),
    "Neural network": dict(color="tab:red", linestyle="--", linewidth=1.5),
    "RK45":           dict(color="tab:green", linestyle=":", linewidth=1.8),
    "Radau":          dict(color="tab:blue", linestyle="-.", linewidth=1.5),
}


# Model selection
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--run-id", help="MLflow run id of the model (default: latest finished run)")
    return parser.parse_args()


def find_run(run_id=None):
    """Return the MLflow run to evaluate: the given run id, else the latest finished run."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    if run_id is not None:
        return client.get_run(run_id)

    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"No MLflow experiment '{config.MLFLOW_EXPERIMENT}' found. Train a model first.")
    runs = client.search_runs([experiment.experiment_id],
                              filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit("No finished run found.")
    return runs[0]


# Runge-Kutta solutions
def solve_with(sampled_parameters, t, **solver_kwargs):
    """Solve the Brusselator for every row. Returns an array of shape (n, 2, len(t))."""
    return np.array([
        solve_brusselator(A, B, x0, y0, t, **solver_kwargs)
        for A, B, x0, y0 in sampled_parameters
    ])


# Plotting
def plot_regime_comparison(save_path, title, parameters, t, curves):
    """
    One column per sample: x(t), y(t), and the pointwise error to the ground truth.

    curves : dict name -> array (n_samples, 2, len(t)).
             Must contain "Ground truth" and "Neural network".
    """
    n = len(parameters)
    truth = curves["Ground truth"]
    nn_errors = relative_l2_error(truth.reshape(n, -1), curves["Neural network"].reshape(n, -1))

    fig, axes = plt.subplots(3, n, figsize=(4.8 * n, 10), sharex=True, squeeze=False)
    for j in range(n):
        A, B, x0, y0 = parameters[j]
        axes[0, j].set_title(
            f"A={A:.2f}, B={B:.2f}, x0={x0:.2f}, y0={y0:.2f}\n"
            f"NN relative L2 error = {nn_errors[j]:.2e}", fontsize=9)
        for name, values in curves.items():
            axes[0, j].plot(t, values[j, 0], label=name, **STYLES[name])
            axes[1, j].plot(t, values[j, 1], label=name, **STYLES[name])
            if name != "Ground truth":
                # pointwise Euclidean error in the (x, y) plane, floored for the log axis
                error = np.linalg.norm(values[j] - truth[j], axis=0)
                axes[2, j].semilogy(t, np.maximum(error, 1e-12), label=name, **STYLES[name])
        axes[2, j].set_xlabel("t")

    axes[0, 0].set_ylabel("x(t)")
    axes[1, 0].set_ylabel("y(t)")
    axes[2, 0].set_ylabel("|(x, y) - ground truth|")
    axes[0, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved '{save_path}'")


# Timing and accuracy table
def compare_timing(model, sampled_parameters, ODE_data, t, rng):
    """Time the NN and each solver on the same test samples and report the mean
    relative L2 error against the stored ground truth."""
    idx = rng.choice(len(sampled_parameters), size=N_TIMING, replace=False)
    params, truth = sampled_parameters[idx], ODE_data[idx]
    rows = []

    start = time.perf_counter()
    nn_pred = model.predict(params)
    seconds = (time.perf_counter() - start) / N_TIMING
    nn_error = relative_l2_error(truth, nn_pred).mean()
    rows.append(("Neural network (one batch)", seconds, nn_error))

    n_single = 100
    start = time.perf_counter()
    for p in params[:n_single]:
        model.predict(p.reshape(1, -1))
    seconds = (time.perf_counter() - start) / n_single
    rows.append(("Neural network (one sample per call)", seconds, nn_error))

    solvers = {"DOP853 (rtol 1e-10, as ground truth)": {}, **OTHER_SOLVERS}
    for name, kwargs in solvers.items():
        start = time.perf_counter()
        solution = solve_with(params, t, **kwargs)
        seconds = (time.perf_counter() - start) / N_TIMING
        error = relative_l2_error(truth, solution.reshape(N_TIMING, -1)).mean()
        rows.append((name, seconds, error))

    print(f"\nTiming and accuracy on {N_TIMING} test samples")
    for name, seconds, error in rows:
        print(f"  {name:<40s} {1e3 * seconds:>9.4f} ms/sample | "
              f"mean relative L2 error vs ground truth: {error:.3e}")


def main():
    args = parse_args()
    rng = np.random.default_rng(config.SEED)

    # Select and load the model (includes input and output scaling)
    run = find_run(args.run_id)
    run_id = run.info.run_id
    print(f"Evaluating MLflow run {run_id} (name: {run.info.run_name})")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    fig_dir = config.FIG_DIR / f"evaluation_{run_id[:8]}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Data
    test_parameters, test_ODE_data, t = load_dataset(dataset_path("test"))
    _, train_ODE_data, _ = load_dataset(dataset_path("train"))
    n_t = len(t)
    regimes = classify_regimes(test_parameters)
    print(f"  Test samples: {len(test_parameters)}")

    # Baseline: always predict the mean training trajectory, ignoring the inputs
    baseline_pred = np.tile(train_ODE_data.mean(axis=0), (len(test_parameters), 1))
    print_summary("Baseline (mean trajectory), test set",
                  summarize_errors(test_parameters, test_ODE_data, baseline_pred))

    # NN predictions are in the original units (the scaling is undone by the model)
    nn_ODE_data = model.predict(test_parameters)
    print_summary("Neural network, test set",
                  summarize_errors(test_parameters, test_ODE_data, nn_ODE_data))

    # Figures: for each regime one image against the ground truth only,
    # and one image that also includes the other Runge-Kutta algorithms
    print()
    for regime in REGIMES:
        candidates = np.flatnonzero(regimes == regime)
        idx = rng.choice(candidates, size=N_SAMPLES_PER_FIGURE, replace=False)
        parameters = test_parameters[idx]

        curves = {
            "Ground truth": test_ODE_data[idx].reshape(-1, 2, n_t),
            "Neural network": nn_ODE_data[idx].reshape(-1, 2, n_t),
        }
        plot_regime_comparison(
            fig_dir / f"{regime}_ground_truth.png",
            f"Regime: {regime} | run {run_id[:8]} | ground truth: DOP853, "
            f"rtol {config.RTOL:g}, atol {config.ATOL:g}",
            parameters, t, curves)

        for name, kwargs in OTHER_SOLVERS.items():
            curves[name] = solve_with(parameters, t, **kwargs)
        plot_regime_comparison(
            fig_dir / f"{regime}_with_other_solvers.png",
            f"Regime: {regime} | run {run_id[:8]} | other solvers at SciPy default "
            f"tolerances (rtol {RTOL_DEFAULT:g}, atol {ATOL_DEFAULT:g})",
            parameters, t, curves)

    compare_timing(model, test_parameters, test_ODE_data, t, rng)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal computation time: {(time.time() - start) / 60:.2f} minutes.")