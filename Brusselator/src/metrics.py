import numpy as np

from .ode import REGIMES, classify_regimes


def relative_l2_error(ODE_true, ODE_pred):
    """Relative L2 error per sample over the full (x, y) vector, in original units."""
    return np.linalg.norm(ODE_pred - ODE_true, axis=1) / np.linalg.norm(ODE_true, axis=1)


def summarize_errors(sampled_parameters, ODE_true, ODE_pred):
    """n / mean / median / max of the relative L2 error: all samples and per regime."""
    errors = relative_l2_error(ODE_true, ODE_pred)
    regimes = classify_regimes(sampled_parameters)
    masks = {"all": np.ones(len(errors), dtype=bool)}
    masks.update({name: regimes == name for name in REGIMES})
    return {
        name: {
            "n": int(mask.sum()),
            "mean": float(errors[mask].mean()),
            "median": float(np.median(errors[mask])),
            "max": float(errors[mask].max()),
        }
        for name, mask in masks.items() if mask.any()
    }


def flatten_summary(prefix, summary):
    """{'stable': {'mean': 0.03, ...}} -> {'<prefix>_mean_stable': 0.03, ...} (flat, for MLflow)."""
    return {f"{prefix}_{stat}_{regime}": value
            for regime, stats in summary.items()
            for stat, value in stats.items() if stat != "n"}


def print_summary(title, summary):
    print(f"\n{title}  (relative L2 error)")
    for name, s in summary.items():
        print(f"  {name:<12s} n={s['n']:>5d} | mean {s['mean']:.4e} | "
              f"median {s['median']:.4e} | max {s['max']:.4e}")