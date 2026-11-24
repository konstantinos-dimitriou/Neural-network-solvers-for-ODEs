"""Shared settings for all scripts."""

from pathlib import Path

# Paths (this file lives in <root>/src/brusselator/; needs an editable install)
ROOT_DIR = Path(__file__).resolve().parents[1]
#ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
#MODEL_DIR = ROOT_DIR / "models"
FIG_DIR = ROOT_DIR / "figures"

# MLflow (default local file store)
#MLFLOW_TRACKING_URI = (ROOT_DIR / "mlruns").as_uri()
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "brusselator"

# sklearn types that MLflow's safe model format (skops) has to be told to trust
SKOPS_TRUSTED_TYPES = [
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.neural_network._stochastic_optimizers.SGDOptimizer",   # for solver="sgd" runs
]

# Reproducibility
SEED = 123

# Simulation settings
T = 20.0          # final time
N_T = 200         # number of time-grid points on [0, T] (endpoints included)

# Parameter box: name -> (low, high). Order = column order of the network input.
RANGES = {
    "A": (0.5, 1.5),
    "B": (0.5, 3.5),
    "x0": (0.5, 2.0),
    "y0": (0.5, 4.0),
}

# Reference ("ground truth") solver
SOLVER_METHOD = "DOP853"
RTOL = 1e-10
ATOL = 1e-12

# Data generation
LOG2_N = 15                   # 2**15 = 32768 samples (Sobol likes powers of 2)
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1            # the test set is the remainder
SPLIT_NAMES = ("train", "validation", "test")

# Regimes relative to the Hopf line B = 1 + A^2
NEAR_HOPF_DELTA = 0.25