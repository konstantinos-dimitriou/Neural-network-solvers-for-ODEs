import numpy as np

from src import config
from src import train_neural_network as training
from src.ode import solve_brusselator, time_grid
from src.metrics import relative_l2_error
from src.generate_data import split_indices


def test_build_model_default_hyperparameters():
    """The model must use all default hyperparameters."""
    model = training.build_model(seed=42)

    mlp = model.regressor.named_steps["mlp"]

    for name, value in training.DEFAULT_HYPERPARAMETERS.items():
        assert mlp.get_params()[name] == value

    assert mlp.random_state == 42
    assert mlp.early_stopping is True

    # Both input and target scaling must be present.
    assert model.regressor.named_steps["input_scaler"] is not None
    assert model.transformer is not None


def test_build_model_custom_hyperparameters():
    """Custom hyperparameters must override defaults without removing others."""
    custom_parameters = {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "learning_rate_init": 0.005,
        "max_iter": 100,
    }

    model = training.build_model(
        hyperparameters=custom_parameters,
        seed=42,
    )

    mlp = model.regressor.named_steps["mlp"]

    # Custom parameters must override the defaults.
    for name, value in custom_parameters.items():
        assert mlp.get_params()[name] == value

    # Unspecified parameters must retain their defaults.
    for name, value in training.DEFAULT_HYPERPARAMETERS.items():
        if name not in custom_parameters:
            assert mlp.get_params()[name] == value


def test_train_model_smoke(monkeypatch):
    """
    Train a small network on a synthetic Brusselator dataset.

    Check prediction shape, original output scale, and performance
    against a mean-trajectory baseline.
    """
    rng = np.random.default_rng(42)

    # Generate 200 parameter sets in a relatively narrow parameter range.
    # This keeps the learning problem manageable for a small network.
    lower_bounds = np.array([0.9, 1.8, 0.9, 0.9])
    upper_bounds = np.array([1.1, 2.2, 1.1, 1.1])

    sampled_parameters = rng.uniform(
        lower_bounds, upper_bounds, size=(200, 4)
    )

    # Generate the corresponding Brusselator trajectories.
    t = np.linspace(0.0, 1.0, 20)
    ODE_data = np.empty((len(sampled_parameters), 2 * len(t)))

    for i, (A, B, x0, y0) in enumerate(sampled_parameters):
        solution = solve_brusselator(A, B, x0, y0, t)

        n_t = len(t)
        ODE_data[i, :n_t] = solution[0]
        ODE_data[i, n_t:] = solution[1]

    # Split into training and validation datasets.
    train_idx, val_idx, _ = split_indices(
        len(sampled_parameters),
        train_fraction=0.8,
        val_fraction=0.2,
        seed=42,
    )

    datasets = {
        "train": (
            sampled_parameters[train_idx],
            ODE_data[train_idx],
            t,
        ),
        "validation": (
            sampled_parameters[val_idx],
            ODE_data[val_idx],
            t,
        ),
    }

    # Replace load_dataset so no actual project datasets are required.
    def fake_load_dataset(path):
        split = path.stem
        return datasets[split]

    monkeypatch.setattr(
        training, "load_dataset", fake_load_dataset
    )

    # Use a deliberately small network to keep the test fast.
    hyperparameters = {
        "hidden_layer_sizes": (32, 32),
        "max_iter": 1000,
        "batch_size": 32,
        "learning_rate_init": 0.003,
        "alpha": 1e-5,
        "n_iter_no_change": 50,
        "tol": 1e-7,
    }

    result = training.train_model(hyperparameters)

    # Retrieve validation data.
    val_parameters, val_ODE_data, _ = datasets["validation"]


    predictions = result.model.predict(val_parameters)

    assert predictions.shape == (len(val_idx), 2 * len(t))


    # The generated Brusselator trajectories are on an O(1) scale.
    # Predictions should therefore also be on that scale, rather
    # than remaining in standardized target space.

    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0.0)
    assert np.all(predictions < 2.0)

    train_ODE_data = datasets["train"][1]

    mean_trajectory = np.mean(train_ODE_data, axis=0)

    baseline_predictions = np.tile(
        mean_trajectory, (len(val_idx), 1)
    )

    model_error = np.mean(
        relative_l2_error(val_ODE_data, predictions)
    )

    baseline_error = np.mean(
        relative_l2_error(val_ODE_data, baseline_predictions)
    )

    assert model_error < baseline_error, (
        f"Model error ({model_error:.4f}) did not beat "
        f"mean-trajectory baseline ({baseline_error:.4f})."
    )

    assert result.n_train == len(train_idx)
    assert result.n_validation == len(val_idx)
    assert result.epochs_run > 0
    assert result.training_seconds > 0.0
    assert len(result.loss_curve) > 0
    assert result.val_score == result.val_summary["all"]["mean"]
