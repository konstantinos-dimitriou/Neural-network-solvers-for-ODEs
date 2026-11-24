import numpy as np
import pytest

from src.dataset import (
    dataset_path,
    save_dataset,
    load_dataset,
    load_metadata,
)
from src import config


def test_save_load_dataset_round_trip(tmp_path):
    """Saving and loading a dataset must preserve all arrays."""
    path = tmp_path / "test_dataset.npz"

    sampled_parameters = np.array([
        [1.0, 2.0, 1.2, 1.5],
        [1.0, 3.0, 1.5, 2.0],
        [2.0, 4.0, 1.0, 1.0],
    ])

    ODE_data = np.array([
        [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        [1.5, 1.6, 1.7, 2.5, 2.6, 2.7],
        [1.0, 1.2, 1.4, 1.0, 1.2, 1.4],
    ])

    t = np.array([0.0, 0.5, 1.0])

    save_dataset(path, sampled_parameters, ODE_data, t)

    loaded_parameters, loaded_ODE_data, loaded_t = load_dataset(path)

    np.testing.assert_array_equal(
        loaded_parameters, sampled_parameters
    )
    np.testing.assert_array_equal(
        loaded_ODE_data, ODE_data
    )
    np.testing.assert_array_equal(loaded_t, t)


def test_load_dataset_mismatched_lengths(tmp_path):
    """Loading must raise ValueError if parameter and trajectory counts differ."""
    path = tmp_path / "invalid_dataset.npz"

    sampled_parameters = np.array([
        [1.0, 2.0, 1.2, 1.5],
        [1.0, 3.0, 1.5, 2.0],
        [2.0, 4.0, 1.0, 1.0],
    ])

    ODE_data = np.array([
        [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        [1.5, 1.6, 1.7, 2.5, 2.6, 2.7],
    ])

    t = np.array([0.0, 0.5, 1.0])

    # Create an invalid dataset directly, bypassing save_dataset.
    np.savez(
        path,
        sampled_parameters=sampled_parameters,
        ODE_data=ODE_data,
        t=t,
    )

    with pytest.raises(ValueError, match="parameter sets.*trajectories"):
        load_dataset(path)


def test_dataset_path(tmp_path):
    """The dataset path must use the split name and data directory."""
    result = dataset_path("train", data_dir=tmp_path)

    assert result == tmp_path / "train.npz"


def test_load_metadata(tmp_path):
    """Metadata must contain the correct generation settings."""
    path = tmp_path / "metadata_test.npz"

    sampled_parameters = np.array([
        [1.0, 2.0, 1.2, 1.5],
        [1.0, 3.0, 1.5, 2.0],
    ])

    ODE_data = np.ones((2, 6))
    t = np.array([0.0, 0.5, 1.0])

    save_dataset(path, sampled_parameters, ODE_data, t)

    metadata = load_metadata(path)

    expected_ranges = {
        str(name): (float(low), float(high))
        for name, (low, high) in config.RANGES.items()
    }

    expected = {
        "seed": config.SEED,
        "ranges": expected_ranges,
        "method": config.SOLVER_METHOD,
        "rtol": config.RTOL,
        "atol": config.ATOL,
        "T": 1.0,
        "N_T": 3,
    }

    assert metadata.keys() == expected.keys()

    assert metadata["seed"] == expected["seed"]
    assert metadata["ranges"] == expected["ranges"]
    assert metadata["method"] == expected["method"]
    assert metadata["rtol"] == pytest.approx(expected["rtol"])
    assert metadata["atol"] == pytest.approx(expected["atol"])
    assert metadata["T"] == expected["T"]
    assert metadata["N_T"] == expected["N_T"]
