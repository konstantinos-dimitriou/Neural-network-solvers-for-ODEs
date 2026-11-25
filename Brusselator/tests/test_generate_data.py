import numpy as np

from src import config
from src.generate_data import (
    sample_parameters,
    generate_trajectories,
    split_indices,
)
from src.ode import solve_brusselator


def test_sample_parameters_shape():
    """The number of Sobol samples must be exactly 2**m."""
    m = 5

    parameters = sample_parameters(exponent_log2_n=m, seed=42)

    assert parameters.shape == (2**m, 4)


def test_sample_parameters_within_ranges():
    """All sampled parameters must lie within their configured ranges."""
    m = 5

    parameters = sample_parameters(exponent_log2_n=m, seed=42)

    for i, (name, (low, high)) in enumerate(config.RANGES.items()):
        assert np.all(parameters[:, i] >= low), (
            f"Some {name} values are below the lower bound."
        )
        assert np.all(parameters[:, i] <= high), (
            f"Some {name} values exceed the upper bound."
        )


def test_sample_parameters_reproducibility():
    """The same seed must produce identical samples."""
    parameters_1 = sample_parameters(exponent_log2_n=5, seed=42)
    parameters_2 = sample_parameters(exponent_log2_n=5, seed=42)

    np.testing.assert_array_equal(parameters_1, parameters_2)


def test_sample_parameters_different_seeds():
    """Different seeds must produce different samples."""
    parameters_1 = sample_parameters(exponent_log2_n=5, seed=42)
    parameters_2 = sample_parameters(exponent_log2_n=5, seed=123)

    assert not np.array_equal(parameters_1, parameters_2)


def test_generate_trajectories_shape():
    """The generated dataset must have shape (n, 2 * len(t))."""
    sampled_parameters = np.array([
        [1.0, 2.0, 1.2, 1.5],
        [1.0, 3.0, 1.5, 2.0],
    ])

    t = np.linspace(0.0, 2.0, 21)

    result = generate_trajectories(sampled_parameters, t)

    assert result.shape == (len(sampled_parameters), 2 * len(t))


def test_generate_trajectories_x_y_layout():
    """The output must contain the complete x-block followed by the y-block."""
    sampled_parameters = np.array([
        [1.0, 2.0, 1.2, 1.5],
        [1.0, 3.0, 1.5, 2.0],
    ])

    t = np.linspace(0.0, 2.0, 21)
    n_t = len(t)

    result = generate_trajectories(sampled_parameters, t)

    for i, (A, B, x0, y0) in enumerate(sampled_parameters):
        reference = solve_brusselator(A, B, x0, y0, t)

        # First block must contain x(t).
        np.testing.assert_allclose(
            result[i, :n_t], reference[0], rtol=1e-7, atol=1e-9
        )

        # Second block must contain y(t).
        np.testing.assert_allclose(
            result[i, n_t:], reference[1], rtol=1e-7, atol=1e-9
        )


def test_split_indices_sizes():
    """The three splits must contain the expected number of samples."""
    n = 1000
    train_fraction = 0.7
    val_fraction = 0.15

    train_idx, val_idx, test_idx = split_indices(
        n,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=42,
    )

    assert len(train_idx) == round(train_fraction * n)
    assert len(val_idx) == round(val_fraction * n)
    assert len(test_idx) == n - len(train_idx) - len(val_idx)

    assert len(train_idx) + len(val_idx) + len(test_idx) == n


def test_split_indices_disjoint():
    """No sample index may occur in more than one split."""
    n = 1000

    train_idx, val_idx, test_idx = split_indices(n, seed=42)

    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0


def test_split_indices_cover_everything():
    """The three splits must collectively contain every index from 0 to n-1."""
    n = 1000

    train_idx, val_idx, test_idx = split_indices(n, seed=42)

    combined = np.concatenate([train_idx, val_idx, test_idx])

    np.testing.assert_array_equal(
        np.sort(combined), np.arange(n)
    )


def test_split_indices_reproducibility():
    """The same seed must produce identical splits."""
    split_1 = split_indices(1000, seed=42)
    split_2 = split_indices(1000, seed=42)

    for idx_1, idx_2 in zip(split_1, split_2):
        np.testing.assert_array_equal(idx_1, idx_2)


def test_split_indices_different_seeds():
    """Different seeds must produce different splits."""
    split_1 = split_indices(1000, seed=42)
    split_2 = split_indices(1000, seed=123)

    assert not all(
        np.array_equal(idx_1, idx_2)
        for idx_1, idx_2 in zip(split_1, split_2)
    )
