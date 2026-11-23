import sys
import os
import numpy as np

# Adjust path to find neural_network.py in the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from generate_data import logistic_solution, generate_data_dictionary

def test_logistic_solution():
    # Initial condition test
    u0, r, t = 0.5, 2.0, 0.0
    result = logistic_solution(t, u0, r)
    assert np.isclose(result, u0), f"Failed initial condition: Expected {u0} at t=0, got {result}"

    # Known value test (exp(3) used in internal math)
    u0, r, t = 0.1, 1.5, 2.0
    expected = 0.69056
    result = logistic_solution(t, u0, r)
    assert np.isclose(result, expected, atol=1e-4), f"Failed known value: Expected ~{expected}, got {result}"

    # Vectorized input test
    t_values = np.linspace(0, 1, 50)
    result = logistic_solution(t_values, u0, r)
    assert isinstance(result, np.ndarray), "Failed vectorization: Expected numpy array."
    assert result.shape == (50,), f"Failed vectorization shape: Expected (50,), got {result.shape}"

    print("All logistic_solution tests passed.")

def test_generate_data_dictionary():
    u0_values = [0.1, 0.2]
    r_values = [1.0, 2.0, 3.0]
    
    # FIX: Cast this to a NumPy array so the math works
    t_values = np.array([0.0, 0.5, 1.0, 1.5]) 
    
    data = generate_data_dictionary(u0_values, r_values, t_values)
    expected_length = len(u0_values) * len(r_values)
    
    # Check keys
    assert "u0" in data, "Failed: 'u0' key missing."
    assert "r" in data, "Failed: 'r' key missing."
    for i in range(len(t_values)):
        assert f"t{i}" in data, f"Failed: 't{i}' key missing."
        
    expected_keys = 2 + len(t_values)
    assert len(data.keys()) == expected_keys, f"Failed: Expected {expected_keys} keys, got {len(data.keys())}."
    
    # Check lengths
    for key, value_list in data.items():
        assert len(value_list) == expected_length, f"Failed: Column '{key}' expected length {expected_length}, got {len(value_list)}."

    print("All generate_data_dictionary tests passed.")


if __name__ == "__main__":
    test_logistic_solution()
    test_generate_data_dictionary()
    print("--- All generate_data tests completed successfully! ---")