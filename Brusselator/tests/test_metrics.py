import numpy as np
import pytest

from src.metrics import (
    relative_l2_error,
    summarize_errors,
    flatten_summary,
)


def test_relative_l2_error_identical_arrays():
    """Identical predictions and reference solutions must give zero error."""
    ODE_true = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    ODE_pred = ODE_true.copy()

    result = relative_l2_error(ODE_true, ODE_pred)

    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)


def test_relative_l2_error_zero_prediction():
    """A zero prediction must give a relative error of one."""
    ODE_true = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    ODE_pred = np.zeros_like(ODE_true)

    result = relative_l2_error(ODE_true, ODE_pred)

    np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-12)


def test_relative_l2_error_scaled_prediction():
    """A prediction scaled by 1.1 must give a relative error of 0.1."""
    ODE_true = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    ODE_pred = 1.1 * ODE_true

    result = relative_l2_error(ODE_true, ODE_pred)

    np.testing.assert_allclose(result, [0.1, 0.1], atol=1e-12)


def test_summarize_errors_group_counts():
    """The regime-specific sample counts must add up to the total count."""
    A = 1.0

    sampled_parameters = np.array([
        [A, 1.5],  # Stable
        [A, 1.6],  # Stable
        [A, 2.0],  # Near Hopf
        [A, 2.5],  # Oscillating
        [A, 3.0],  # Oscillating
    ])

    ODE_true = np.ones((5, 2))
    ODE_pred = 1.1 * ODE_true

    summary = summarize_errors(
        sampled_parameters, ODE_true, ODE_pred
    )

    total_count = sum(
        summary[regime]["n"]
        for regime in ("stable", "near_hopf", "oscillating")
        if regime in summary
    )

    assert total_count == len(sampled_parameters)
    assert summary["all"]["n"] == len(sampled_parameters)


def test_summarize_errors_all_equals_plain_mean():
    """The overall mean must equal the mean of all individual errors."""
    sampled_parameters = np.array([
        [1.0, 1.5],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 1.7],
    ])

    ODE_true = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ])

    ODE_pred = np.array([
        [1.1, 2.0],
        [2.0, 3.2],
        [3.3, 4.0],
        [4.0, 5.5],
    ])

    individual_errors = relative_l2_error(ODE_true, ODE_pred)

    summary = summarize_errors(
        sampled_parameters, ODE_true, ODE_pred
    )

    assert summary["all"]["mean"] == pytest.approx(
        np.mean(individual_errors)
    )


def test_flatten_summary_key_names():
    """Flattened keys must follow the prefix_stat_regime convention."""
    summary = {
        "stable": {
            "n": 10,
            "mean": 0.01,
            "median": 0.008,
            "max": 0.03,
        },
        "near_hopf": {
            "n": 5,
            "mean": 0.02,
            "median": 0.015,
            "max": 0.04,
        },
    }

    result = flatten_summary("validation", summary)

    expected = {
        "validation_mean_stable": 0.01,
        "validation_median_stable": 0.008,
        "validation_max_stable": 0.03,
        "validation_mean_near_hopf": 0.02,
        "validation_median_near_hopf": 0.015,
        "validation_max_near_hopf": 0.04,
    }

    assert result == expected


def test_flatten_summary_excludes_sample_counts():
    """The flattened summary must not contain any n entries."""
    summary = {
        "all": {
            "n": 100,
            "mean": 0.02,
            "median": 0.015,
            "max": 0.05,
        },
        "stable": {
            "n": 60,
            "mean": 0.01,
            "median": 0.008,
            "max": 0.03,
        },
    }

    result = flatten_summary("validation", summary)

    assert all("_n_" not in key for key in result)
    assert len(result) == 6
