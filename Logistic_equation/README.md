# Logistic Equation — Neural Network Surrogate Solver

A from-scratch neural network trained to solve the normalised logistic ODE, replacing
repeated numerical integration with a single forward pass at evaluation time.

---

## The Logistic Equation

The [logistic differential equation](https://en.wikipedia.org/wiki/Logistic_function#Logistic_differential_equation)
models population growth with a carrying capacity:

$$\frac{dP}{dt} = rP(1-\frac{P}{K}) \qquad P(0) = P_0$$

where $r > 0$ is the growth rate, $K > 0$ is the carrying capacity and $P_0 \in (0, K)$ denotes the initial population.
By applying the normalization $u = P/K$ one arrives at the equation

$$\frac{du}{dt} = r \cdot u(1 - u), \qquad u(0) = u_0$$

where $u_0 \in (0, 1)$ is the normalised initial population.
The exact solution is:

$$u(t) = \frac{u_0 \, e^{rt}}{(1 - u_0) + u_0 \, e^{rt}}$$

The network learns the parameter-to-solution map $(u_0, r) \mapsto (u(t_0), \dots, u(t_{99}))$
over a fixed time grid $t \in [0, 1]$ with 100 equally spaced points.

---

## Sample Plots

![Predicted vs Exact logistic solutions](sample_plots/predictions.png)
![Test MSE per epoch](sample_plots/loss_curve.png)

---

## Project Structure

- `generate_data.py` — samples $(u_0, r)$ pairs, computes exact solutions, saves to CSV
- `train_neural_network.py` — builds and trains the network, saves the trained model
- `solve_ode.py` — loads the trained network and evaluates it on new parameter pairs
- `neural_network.py` — self-contained NumPy implementation of the neural network
- `DATA/` — generated training and test CSVs
- `TRAINED_NN/` — saved model weights (`.npz`)
- `PLOTS/` — loss curve and prediction plots produced during training
- `sample_plots/` — example output plots for this README
- `TESTS/` — unit tests for all modules

---

## Approach

This implementation is adapted from the MNIST digit recognition framework described in:

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen
- [Neural Networks playlist](https://www.youtube.com/watch?v=8bNIkfRJZpo) — Sebastian Lague

The core network is a fully connected feedforward network trained with mini-batch SGD
and backpropagation. Hidden layers use `tanh` or `sigmoid` activation; the output layer is linear,
which is appropriate for a regression target. Weights are initialised with Xavier/Glorot scaling
(1/sqrt(fan_in)) to keep activation variance stable across layers and avoid vanishing gradients,
which is appropriate for tanh and sigmoid hidden units.

---

## How to Run

**1 — Generate data**
```bash
python generate_data.py
```
Produces `DATA/training_data.csv` (10 000 samples) and `DATA/testing_data.csv` (1 024 samples).

**2 — Train the network**
```bash
python train_neural_network.py
```
Trains the network and saves the model to `TRAINED_NN/trained_model_logistic_ode.npz`.
Loss curve and prediction plots are saved to `PLOTS/`.

**3 — Evaluate on new parameters**
```bash
python solve_ode.py
```
Loads the trained model and evaluates it on a set of $(u_0, r)$ pairs, comparing the
network output to the exact solution. Similar to prediction an evaluation plot is
saved to `PLOTS/`.

**Run tests**
```bash
python -m pytest TESTS/
```

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

This project is licensed under the [MIT License](../LICENSE).
