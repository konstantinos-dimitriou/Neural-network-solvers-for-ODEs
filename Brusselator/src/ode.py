import numpy as np
from scipy.integrate import solve_ivp

from . import config


REGIMES = ("stable", "near_hopf", "oscillating")

def time_grid(T=config.T, n_t=config.N_T):
    """Fixed time grid on [0, T], endpoints included."""
    return np.linspace(0.0, T, n_t)


def brusselator_rhs(t, u, A, B):
    # Right-hand side of the Brusselator, u = (x, y).
    x, y = u
    return [A + x**2 * y - (B + 1.0) * x, B * x - x**2 * y]


def solve_brusselator(A, B, x0, y0, t_eval, method=config.SOLVER_METHOD,
                      rtol=config.RTOL, atol=config.ATOL):
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


def classify_regimes(sampled_parameters, delta=config.NEAR_HOPF_DELTA):
    """
    Regime of each parameter set, relative to the Hopf line B = 1 + A^2:
    stable      : B < 1 + A^2 - delta
    near_hopf   : |B - (1 + A^2)| <= delta
    oscillating : B > 1 + A^2 + delta
    """
    A, B = sampled_parameters[:, 0], sampled_parameters[:, 1]
    margin = B - (1.0 + A ** 2)

    tol = 1e-12

    return np.where(
        margin < -delta - tol, "stable",
        np.where(margin > delta + tol, "oscillating", "near_hopf")
    )
#    return np.where(margin < -delta, "stable",
#                    np.where(margin > delta, "oscillating", "near_hopf"))
