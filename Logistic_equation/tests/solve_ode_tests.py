import sys
import os
import numpy as np

# Adjust path to find scripts in the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import neural_network as nn
from solve_ode import logistic_exact, load_network, solve

def test_logistic_exact():
    # Test initial condition: at t=0, u(t) should equal u0
    u0, r, t = 0.5, 2.0, 0.0
    result = logistic_exact(t, u0, r)
    assert np.isclose(result, u0), f"Failed initial condition: Expected {u0}, got {result}"

    # Test vectorization: passing an array of times should return an array of solutions
    t_values = np.linspace(0, 1, 100)
    result_array = logistic_exact(t_values, u0, r)
    assert isinstance(result_array, np.ndarray), "Failed vectorization: Expected numpy array."
    assert result_array.shape == (100,), f"Failed vectorization shape: Expected (100,), got {result_array.shape}"

    print("All logistic_exact tests passed.")


def test_inference_pipeline():
    # Create and save a dummy network 
    # Architecture: 2 inputs (u0, r), 4 hidden, 100 outputs (t0...t99)
    dummy_net = nn.NeuralNetwork((2, 4, 100), activation="tanh")
    temp_model_path = "temp_dummy_model.npz"
    dummy_net.save(temp_model_path)

    try:
        # Test load_network wrapper
        loaded_net = load_network(temp_model_path)
        assert isinstance(loaded_net, nn.NeuralNetwork), "load_network did not return a NeuralNetwork instance."
        assert loaded_net.num_layers == 3, "Loaded network architecture mismatch."

        # Test solve function wrapper
        u0_test = 0.5
        r_test = 2.0
        prediction = solve(loaded_net, u0_test, r_test)

        # Assertions on the output
        assert isinstance(prediction, np.ndarray), "solve() should return a numpy array."
        assert prediction.shape == (100,), f"solve() should return a flat array of shape (100,), got {prediction.shape}"
        assert prediction.ndim == 1, "solve() output should be exactly 1-dimensional (flattened)."

    finally:
        # Clean up temporary model file
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    print("All inference pipeline tests passed.")


if __name__ == "__main__":
    test_logistic_exact()
    test_inference_pipeline()
    print("--- All solve_ode tests completed successfully! ---")