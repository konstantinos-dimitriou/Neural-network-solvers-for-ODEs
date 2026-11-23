"""
Step 1: Generate training / test data for the Brusselator with Runge-Kutta.

ODE system (A, B > 0 constants, x, y >= 0 concentrations):
    dx/dt = A + x^2 y - (B + 1) x
    dy/dt = B x - x^2 y

Learning problem (later steps):  (A, B, x0, y0)  ->  (x(t_1..t_N), y(t_1..t_N))

Saved arrays (in data/train.npz and data/test.npz):
    X : (n_samples, 4)        columns = A, B, x0, y0
    Y : (n_samples, 2*N_T)    columns = x(t_0..t_{N_T-1}), then y(t_0..t_{N_T-1})
    t : (N_T,)                the fixed time grid on [0, T]
plus the settings (T, N_T, seed, ranges, tolerances) for reproducibility.

Run:  python generate_data.py
"""

from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import matplotlib

matplotlib.use("Agg")  # no display needed; figures are saved to files
import matplotlib.pyplot as plt


# Fix data generation settings
SEED = 123
LOG2_N = 13       # total number of samples = 2**13 = 8192 (Sobol likes powers of 2)
TRAIN_FRACTION = 0.7

# Parameter box: name -> (low, high). Order = column order of X.
# Fix simulation settings
T = 20.0          # final time
N_T = 200         # number of time-grid points on [0, T] (endpoints included)
RANGES = {
    "A": (0.5, 1.5),
    "B": (0.5, 3.5),
    "x0": (0.5, 2.0),
    "y0": (0.5, 4.0),
}

# Reference solver settings (tight, so that RK is effectively "ground truth")
METHOD = "DOP853"
RTOL = 1e-10
ATOL = 1e-12

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"



# Setup ODE and solver
def brusselator_rhs(t, u, A, B):
    # Right-hand side of the Brusselator, u = (x, y).
    x, y = u
    return [A + x**2*y - (B + 1.0) * x, B * x - x**2*y]


def solve_brusselator(A, B, x0, y0, t_eval, method=METHOD, rtol=RTOL, atol=ATOL):
    # Solve one initial value problem; returns array of shape (2, len(t_eval)).
    sol = solve_ivp(
        brusselator_rhs,
        (t_eval[0], t_eval[-1]),
        [x0, y0],
        args=(A, B),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for (A,B,x0,y0)=({A},{B},{x0},{y0}): {sol.message}")
    return sol.y


# Sampling parameter space
def sample_parameters(sample_size=LOG2_N, seed=SEED):
    # Scrambled Sobol samples in the parameter box, shape (2**log2_n, 4).
    lower_bounds = np.array([r[0] for r in RANGES.values()])
    upper_bounds = np.array([r[1] for r in RANGES.values()])
    sampler = qmc.Sobol(d=len(RANGES), scramble=True, seed=seed)
    unit = sampler.random_base2(m=sample_size)
    return qmc.scale(unit, lower_bounds, upper_bounds)


# Full ODE data generation
def generate_dataset(sampled_parameters, t):
    # Solve the ODE for every row of X. Returns Y of shape (n, 2*N_T).
    n = sampled_parameters.shape[0]
    Y = np.empty((n, 2 * len(t)))
    for i, (A, B, x0, y0) in enumerate(sampled_parameters):
        sol = solve_brusselator(A, B, x0, y0, t)
        Y[i, : len(t)] = sol[0]   # x(t)
        Y[i, len(t):] = sol[1]    # y(t)
        if (i + 1) % 1000 == 0:
            print(f"  solved {i + 1}/{n}")
    return Y


# Plot for a visual sanity check
def plot_examples(t, path):
    # Trajectories on both sides of the Hopf line B = 1 + A^2 (A = 1 -> B_c = 2).
    cases = [
        ("stable  (B=1.5 < 2)", 1.0, 1.5, 1.0, 3.0),
        ("oscillating  (B=3.0 > 2)", 1.0, 3.0, 1.0, 1.0),
    ]
    t_fine = np.linspace(t[0], t[-1], 2000)  # smooth curves for the plot only
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, (label, A, B, x0, y0) in enumerate(cases):
        x, y = solve_brusselator(A, B, x0, y0, t_fine)
        t = t_fine
        ax = axes[0, col]
        ax.plot(t, x, label="x(t)")
        ax.plot(t, y, label="y(t)")
        ax.set_title(f"A={A}, {label}")
        ax.set_xlabel("t")
        ax.legend()
        ax = axes[1, col]
        ax.plot(x, y)
        ax.plot(A, B / A, "r*", ms=10, label="fixed point (A, B/A)")
        ax.plot(x0, y0, "ko", label="start")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Save generated ODE data to .npz file
def save_npz(path, sampled_parameters, ODE_data, t):
    np.savez(
        path,
        sampled_parameters=sampled_parameters,
        ODE_data=ODE_data,
        t=t,
        T=T,
        N_T=N_T,
        seed=SEED,
        input_names=np.array(list(RANGES.keys())),
        ranges=np.array(list(RANGES.values())),
        method=METHOD,
        rtol=RTOL,
        atol=ATOL,
    )


# Main
def main():
    DATA_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    t = np.linspace(0.0, T, N_T)

    # 1. sample parameters (Sobol, seeded)
    sampled_parameters = sample_parameters()
    n = sampled_parameters.shape[0]
    print(f"Sampled {n} parameter sets with Sobol (seed={SEED}).")

    # 2. solve all IVPs
    print("Generating ODE data ...")
    start = time.perf_counter()
    ODE_data = generate_dataset(sampled_parameters, t)
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f} s ({1e3 * elapsed / n:.2f} ms per trajectory).")

    # 3. random 70/30 split (shuffled, seeded). Sobol points are ordered, so we
    #    must shuffle before splitting.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    save_npz(DATA_DIR / "train.npz", sampled_parameters[train_idx], ODE_data[train_idx], t)
    save_npz(DATA_DIR / "test.npz", sampled_parameters[test_idx], ODE_data[test_idx], t)
    print(f"Saved {len(train_idx)} train and {len(test_idx)} test samples to {DATA_DIR}/")

    # 4. a plot to look at
    plot_examples(t, FIG_DIR / "sample_trajectories.png")
    print(f"Saved example plot to {FIG_DIR / 'generation_sample_trajectories.png'}")

    # quick info on regimes
    osc = sampled_parameters[:, 1] > 1.0 + sampled_parameters[:, 0] ** 2
    print(f"Fraction of samples in the oscillating regime (B > 1 + A^2): {osc.mean():.2f}")


if __name__ == "__main__":
    main()
