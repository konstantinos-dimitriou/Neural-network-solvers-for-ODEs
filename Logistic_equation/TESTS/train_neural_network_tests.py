import sys
import os
import numpy as np
import pandas as pd

# Adjust path to find scripts in the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import neural_network as nn
from train_neural_network import load_data, train_with_tracking, plot_loss_curve

def test_load_data():
    # Create a dummy CSV with 2 rows of data
    temp_csv = "temp_test_data.csv"
    
    # Build dictionary 
    data_dict = {
        "u0": [0.1, 0.2],
        "r": [1.5, 2.5]
    }
    for i in range(100):
        data_dict[f"t{i}"] = [0.5, 0.6] 
        
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data_dict)
    df.to_csv(temp_csv, index=False)

    try:
        # Run load function
        data = load_data(temp_csv)
        
        # Assertions
        assert len(data) == 2, f"Expected 2 samples, got {len(data)}"
        
        x0, y0 = data[0]
        assert x0.shape == (2, 1), f"Expected input x shape (2, 1), got {x0.shape}"
        assert y0.shape == (100, 1), f"Expected target y shape (100, 1), got {y0.shape}"
        assert x0.dtype == np.float64, f"Input data must be float64, got {x0.dtype}"
        
    finally:
        #Clean up temporary file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    print("All load_data tests passed.")


def test_train_with_tracking():
    # Create dummy dataset (4 training samples, 2 testing samples)
    training_data = [(np.random.randn(2, 1), np.random.randn(100, 1)) for _ in range(4)]
    test_data     = [(np.random.randn(2, 1), np.random.randn(100, 1)) for _ in range(2)]

    # Initialize dummy network
    net = nn.NeuralNetwork((2, 4, 100), activation="tanh")

    epochs = 2
    mini_batch_size = 2
    learning_rate = 0.1

    # Run the training wrapper
    loss_history = train_with_tracking(
        net, training_data, test_data,
        epochs, mini_batch_size, learning_rate
    )

    # Verify tracking array matches the number of epochs
    assert len(loss_history) == epochs, f"Expected loss history length {epochs}, got {len(loss_history)}"
    
    # Verify the loss is a valid float number
    assert isinstance(loss_history[0], float), "Loss history should contain floats"
    assert not np.isnan(loss_history[0]), "Loss evaluates to NaN."

    print("All train_with_tracking tests passed.")


def test_plot_generation():
    # Test that plotting functions execute and generate files without matplotlib crashing
    temp_plot = "temp_plot.png"
    loss_history = [0.5, 0.4, 0.3]

    try:
        plot_loss_curve(loss_history, save_path=temp_plot)
        assert os.path.exists(temp_plot), "Plot file was not created on disk."
    finally:
        if os.path.exists(temp_plot):
            os.remove(temp_plot)

    print("All plot generation tests passed.")


if __name__ == "__main__":
    test_load_data()
    test_train_with_tracking()
    test_plot_generation()
    print("--- All train_neural_network tests completed successfully! ---")