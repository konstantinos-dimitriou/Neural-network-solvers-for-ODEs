import numpy as np

from . import config


def dataset_path(split, data_dir=config.DATA_DIR):
    return data_dir / f"{split}.npz"


def save_dataset(path, sampled_parameters, ODE_data, t):
    np.savez(
        path,
        sampled_parameters=sampled_parameters,   # (n, 4): A, B, x0, y0
        ODE_data=ODE_data,                       # (n, 2*N_T): x-block, then y-block
        t=t,
        seed=config.SEED,
        input_names=np.array(list(config.RANGES.keys())),
        ranges=np.array(list(config.RANGES.values())),
        method=config.SOLVER_METHOD,
        rtol=config.RTOL,
        atol=config.ATOL,
    )


def load_metadata(path):
    """Settings stored with the data at generation time."""
    with np.load(path) as data:
        return {
            "seed": int(data["seed"]),
            "ranges": {str(name): (float(low), float(high))
                       for name, (low, high) in zip(data["input_names"], data["ranges"])},
            "method": str(data["method"]),
            "rtol": float(data["rtol"]),
            "atol": float(data["atol"]),
            "T": float(data["t"][-1]),
            "N_T": len(data["t"]),
        }

    
def load_dataset(path):
    """Returns sampled_parameters (n, 4), ODE_data (n, 2*N_T) and the time grid t."""
    with np.load(path) as data:
        sampled_parameters, ODE_data, t = data["sampled_parameters"], data["ODE_data"], data["t"]
    if len(sampled_parameters) != len(ODE_data):
        raise ValueError(f"{path}: {len(sampled_parameters)} parameter sets "
                         f"but {len(ODE_data)} trajectories")
    return sampled_parameters, ODE_data, t