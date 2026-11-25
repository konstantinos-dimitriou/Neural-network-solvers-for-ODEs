# Neural Network Solvers for ODEs

A collection of neural network solvers for ordinary differential equations
and systems, built either from-scratch in pure NumPy, or using Scikit-learn and MLflow.
The goal is to train a network once on a family of ODE solutions and then 
use it as a cheap alternative — replacing repeated numerical integration with
a single forward pass.

---

## Motivation

Solving ODEs numerically is straightforward for a single set of parameters, but becomes
expensive when you need solutions across a large parameter space (e.g. parameter sweeps,
or real-time applications). A trained neural network can approximate the solution map
from initial conditions and parameters to the full solution curve, reducing evaluation
cost to a matrix multiplication.

---

## Projects

### Brusselator
A solver for the Brusselator system of nonlinear ODEs, mapping initial concentrations
of the uknown functions $x$ and $y$, and the constant concentrations of $A$,$B$ to
the full solution trajectories of $x$ and $y$. See the [Brusselator](Brusselator/)
folder for details.

### Logistic Equation
Surrogate solver for the normalised logistic ODE, mapping initial condition and growth
rate directly to the full solution curve. See the [Logistic-equation](Logistic_equation/)
folder for details. This implementation uses only NumPy, Pandas, and Matplotlib, everything
else, including the neural network logic, is build from scratch.

---

## Approach

Each solver follows the same three-step workflow:

1. **Generate data** — sample the parameter space and compute exact solutions analytically
   or numerically, saving training and test sets to CSV.
2. **Train** — a fully connected network is trained via mini-batch SGD to map input
   parameters to the solution vector at a fixed set of time points.
3. **Evaluate** — load the trained network once and query it for any new parameter pair
   in constant time.
4. **Hyperparameter tuning** — For some (e.g. the Brusselator system) hyperparameter tuning is also possible.

---

## Future Work

- Solvers for further ODEs and systems of ODEs (Pendulum equation, Van der Pol oscillator)
- Apply same approach to PDEs (current project).

---

## Requirements

See requirements for each equation seperately. The following cover all:

- Python 3.10 or newer
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- MLflow
- skops
- pytest

```bash
pip install -r requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE).
