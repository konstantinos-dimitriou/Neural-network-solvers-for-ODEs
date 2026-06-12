# Neural Network Solvers for ODEs

A collection of from-scratch neural network solvers for ordinary differential equations,
built in pure NumPy. The goal is to train a network once on a family of ODE solutions
and then use it as a cheap surrogate — replacing repeated numerical integration with a
single forward pass.

---

## Motivation

Solving ODEs numerically is straightforward for a single set of parameters, but becomes
expensive when you need solutions across a large parameter space (e.g. parameter sweeps,
or real-time applications). A trained neural network can approximate the solution map
from initial conditions and parameters to the full solution curve, reducing evaluation
cost to a matrix multiplication.

---

## Projects

### Logistic Equation
Surrogate solver for the normalised logistic ODE, mapping initial condition and growth
rate directly to the full solution curve. See the [Logistic-equation](Logistic_equation/)
folder for details.

---

## Approach

Each solver follows the same three-step workflow:

1. **Generate data** — sample the parameter space and compute exact solutions analytically
   or numerically, saving training and test sets to CSV.
2. **Train** — a fully connected network is trained via mini-batch SGD to map input
   parameters to the solution vector at a fixed set of time points.
3. **Evaluate** — load the trained network once and query it for any new parameter pair
   in constant time.

---

## Future Work

- PyTorch implementations of each solver for direct comparison with the NumPy baseline
- Solvers for further ODEs and systems of ODEs (Pendulum equation, Van der Pol oscillator)

---

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib

```bash
pip install numpy pandas matplotlib
```

---

## License

This project is licensed under the [MIT License](LICENSE).
