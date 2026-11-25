"""
Generate train / validation / test data for the Brusselator with Runge-Kutta.

Learning problem:  (A, B, x0, y0)  ->  (x(t_1..t_N), y(t_1..t_N))
Saved to data/{train,validation,test}.npz, see dataset.py for the layout.
"""

import time

import numpy as np
from scipy.stats import qmc

from . import config
from .dataset import save_dataset, dataset_path
from .ode import solve_brusselator, time_grid


def sample_parameters(exponent_log2_n=config.LOG2_N, seed=config.SEED, ranges=config.RANGES):
    # Scrambled Sobol samples in the parameter box, shape (2**exponent_log2_n, len(ranges)).
    lower_bounds = np.array([low for low, _ in ranges.values()])
    upper_bounds = np.array([high for _, high in ranges.values()])
    sampler = qmc.Sobol(d=len(ranges), scramble=True, seed=seed)
    unit = sampler.random_base2(m=exponent_log2_n)
    return qmc.scale(unit, lower_bounds, upper_bounds)


def generate_trajectories(sampled_parameters, t):
    # Solve the ODE for every parameter set. Returns shape (n, 2 * len(t)).
    n, n_t = len(sampled_parameters), len(t)
    ODE_data = np.empty((n, 2 * n_t))
    for i, (A, B, x0, y0) in enumerate(sampled_parameters):
        x, y = solve_brusselator(A, B, x0, y0, t)
        ODE_data[i, :n_t] = x
        ODE_data[i, n_t:] = y
        if (i + 1) % 1000 == 0:
            print(f"  solved {i + 1}/{n}")
    return ODE_data


def split_indices(n, train_fraction=config.TRAIN_FRACTION,
                  val_fraction=config.VAL_FRACTION, seed=config.SEED):
    # Shuffled (seeded) split. Sobol points are ordered, so we must shuffle first.
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    t = time_grid()

    sampled_parameters = sample_parameters()
    n = len(sampled_parameters)
    print(f"Sampled {n} parameter sets with Sobol (seed={config.SEED}).")

    print("Generating ODE data ...")
    start = time.perf_counter()
    ODE_data = generate_trajectories(sampled_parameters, t)
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f} s ({1e3 * elapsed / n:.2f} ms per trajectory).")

    splits = dict(zip(config.SPLIT_NAMES, split_indices(n)))
    for name, idx in splits.items():
        save_dataset(dataset_path(name), sampled_parameters[idx], ODE_data[idx], t)
        print(f"Saved {len(idx)} {name} samples to {dataset_path(name)}")


if __name__ == "__main__":
    main()