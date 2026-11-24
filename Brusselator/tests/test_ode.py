import numpy as np
import pytest

from src.ode import (
    brusselator_rhs,
    solve_brusselator,
    time_grid,
    classify_regimes,
)


def test_rhs_fixed_point():
    """The RHS must vanish at the analytical fixed point (A, B/A)."""
    A, B = 1.0, 3.0
    x, y = A, B / A

    result = brusselator_rhs(0.0, [x, y], A, B)

    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-12)


def test_rhs_hand_computed_value():
    """Check the RHS against a manually calculated example."""
    A, B = 1.0, 3.0
    x, y = 2.0, 1.0

    result = brusselator_rhs(0.0, [x, y], A, B)

    np.testing.assert_allclose(result, [-3.0, 2.0], atol=1e-12)


def test_solver_output_shape():
    """The solver must return an array of shape (2, len(t_eval))."""
    t = np.linspace(0.0, 10.0, 101)

    result = solve_brusselator(
        A=1.0, B=3.0, x0=1.2, y0=2.0, t_eval=t
    )

    assert result.shape == (2, len(t))


def test_solver_initial_condition():
    """The first solution column must equal the initial condition."""
    t = np.linspace(0.0, 10.0, 101)
    initial_condition = np.array([1.2, 2.0])

    result = solve_brusselator(
        A=1.0, B=3.0,
        x0=initial_condition[0],
        y0=initial_condition[1],
        t_eval=t,
    )

    np.testing.assert_allclose(
        result[:, 0], initial_condition, atol=1e-12
    )


def test_solver_fixed_point():
    """A solution starting at the fixed point must remain constant."""
    A, B = 1.0, 3.0
    fixed_point = np.array([A, B / A])
    t = np.linspace(0.0, 50.0, 501)

    result = solve_brusselator(
        A=A, B=B,
        x0=fixed_point[0],
        y0=fixed_point[1],
        t_eval=t,
    )

    expected = np.repeat(fixed_point[:, None], len(t), axis=1)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def rk4_reference(A, B, initial_condition, t):
    """Independent fixed-step classical RK4 implementation."""
    u = np.empty((2, len(t)))
    u[:, 0] = initial_condition

    def f(u):
        return np.asarray(brusselator_rhs(0.0, u, A, B))

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        current = u[:, i]

        k1 = f(current)
        k2 = f(current + 0.5 * h * k1)
        k3 = f(current + 0.5 * h * k2)
        k4 = f(current + h * k3)

        u[:, i + 1] = current + (h / 6.0) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    return u


def test_solver_against_rk4():
    """Compare solve_ivp against an independent fixed-step RK4."""
    A, B = 1.0, 2.0
    initial_condition = np.array([1.2, 2.0])

    # Small fixed time step for the RK4 reference solution.
    t = np.linspace(0.0, 5.0, 501)

    reference = rk4_reference(A, B, initial_condition, t)

    result = solve_brusselator(
        A=A, B=B,
        x0=initial_condition[0],
        y0=initial_condition[1],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result, reference, rtol=1e-5, atol=1e-7
    )


def test_stable_regime_converges_to_fixed_point():
    """Below the Hopf line, trajectories converge to the fixed point."""
    A, B = 1.0, 1.5
    fixed_point = np.array([A, B / A])

    t = np.linspace(0.0, 100.0, 2001)

    result = solve_brusselator(
        A=A, B=B,
        x0=1.5, y0=1.0,
        t_eval=t,
    )

    np.testing.assert_allclose(
        result[:, -1], fixed_point, atol=1e-3
    )


def test_oscillating_regime_keeps_oscillating():
    """Above the Hopf line, trajectories exhibit sustained oscillations."""
    A, B = 1.0, 3.0

    t = np.linspace(0.0, 100.0, 5001)

    result = solve_brusselator(
        A=A, B=B,
        x0=1.2, y0=2.0,
        t_eval=t,
    )

    # Ignore the initial transient.
    x = result[0, len(t) // 2:]

    amplitude = np.max(x) - np.min(x)

    # The oscillation amplitude should remain clearly nonzero.
    assert amplitude > 0.5


def test_time_grid_length():
    """The time grid must contain exactly n_t points."""
    T, n_t = 10.0, 101

    t = time_grid(T=T, n_t=n_t)

    assert len(t) == n_t


def test_time_grid_endpoints():
    """The time grid must start at zero and end at T."""
    T, n_t = 10.0, 101

    t = time_grid(T=T, n_t=n_t)

    assert t[0] == 0.0
    assert t[-1] == T


def test_classify_regimes_clear_examples():
    """Check one clear example of each regime."""
    A = 1.0

    parameters = np.array([
        [A, 1.5],  # Stable: B < 2
        [A, 2.0],  # Near Hopf: B = 2
        [A, 3.0],  # Oscillating: B > 2
    ])

    result = classify_regimes(parameters, delta=0.1)

    expected = np.array([
        "stable",
        "near_hopf",
        "oscillating",
    ])

    np.testing.assert_array_equal(result, expected)


def test_classify_regimes_exact_boundaries():
    """Both boundaries at +/- delta belong to the near-Hopf regime."""
    A = 1.0
    delta = 0.1
    hopf_line = 1.0 + A**2

    parameters = np.array([
        [A, hopf_line - delta - 0.01],  # Just below lower boundary
        [A, hopf_line - delta],         # Exact lower boundary
        [A, hopf_line + delta],         # Exact upper boundary
        [A, hopf_line + delta + 0.01],  # Just above upper boundary
    ])

    result = classify_regimes(parameters, delta=delta)

    expected = np.array([
        "stable",
        "near_hopf",
        "near_hopf",
        "oscillating",
    ])

    np.testing.assert_array_equal(result, expected)
