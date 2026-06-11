import sys
import os
import numpy as np

# Adjust path to find neural_network.py in the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from neural_network import NeuralNetwork

def test_network_initialization():
    # Setup architecture: 2 inputs, 1 hidden layer of 4 neurons, 3 outputs
    layer_sizes = [2, 4, 3]
    net = NeuralNetwork(layer_sizes, activation="tanh")
    
    # Check that layer count is captured correctly
    assert net.num_layers == 3, f"Expected 3 layers, got {net.num_layers}"
    
    # Check weight shapes: layer1->layer2 should be (4, 2), layer2->layer3 should be (3, 4)
    assert net.weights[0].shape == (4, 2), f"Expected weights[0] shape (4,2), got {net.weights[0].shape}"
    assert net.weights[1].shape == (3, 4), f"Expected weights[1] shape (3,4), got {net.weights[1].shape}"
    
    # Check bias shapes: layer2 should be (4, 1), layer3 should be (3, 1)
    assert net.biases[0].shape == (4, 1), f"Expected biases[0] shape (4,1), got {net.biases[0].shape}"
    assert net.biases[1].shape == (3, 1), f"Expected biases[1] shape (3,1), got {net.biases[1].shape}"
    
    # Check that invalid activations trigger a ValueError
    try:
        NeuralNetwork(layer_sizes, activation="relu")
        assert False, "Network should have raised a ValueError for invalid activation string 'relu'"
    except ValueError:
        pass # Exception caught successfully

    print("All network initialization tests passed.")


def test_feedforward_and_prediction():
    layer_sizes = [2, 5, 1] # 2 inputs, 5 hidden, 1 output (typical for your ODE structure)
    net = NeuralNetwork(layer_sizes, activation="sigmoid")
    
    # Create dummy column vector input x with shape (input_size, 1)
    x = np.array([[0.5], [0.1]])
    
    output = net.feedforward(x)
    prediction = net.predict(x)
    
    # Ensure prediction and feedforward map to the same outcome
    assert np.array_equal(output, prediction), "feedforward() and predict() returned different outputs."
    
    # Ensure dimensions are correctly tracked to the single output target (1, 1)
    assert output.shape == (1, 1), f"Expected output shape (1,1), got {output.shape}"
    
    print("All feedforward and prediction tests passed.")


def test_backpropagation_gradient_shapes():
    layer_sizes = [1, 8, 8, 5] # Complex architecture deep network
    net = NeuralNetwork(layer_sizes, activation="tanh")
    
    # Generate dummy input and regression target matching dimension constraints
    x = np.array([[0.5]])    # (1, 1)
    y = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]) # (5, 1)
    
    nabla_b, nabla_w = net._backprop(x, y)
    
    # Validate gradient structural shape mappings against network layer shapes
    assert len(nabla_w) == len(net.weights), "Gradient weights list length mismatch."
    assert len(nabla_b) == len(net.biases), "Gradient biases list length mismatch."
    
    for i in range(len(net.weights)):
        assert nabla_w[i].shape == net.weights[i].shape, f"Weight gradient shape mismatch at layer {i}."
        assert nabla_b[i].shape == net.biases[i].shape, f"Bias gradient shape mismatch at layer {i}."
        
    print("All backpropagation gradient shape tests passed.")


def test_save_and_load_consistency():
    layer_sizes = [1, 10, 1]
    net_original = NeuralNetwork(layer_sizes, activation="tanh")
    
    # Run a test prediction before serialization
    x = np.array([[0.75]])
    original_prediction = net_original.predict(x)
    
    # Save parameters to file system temporarily
    temp_filename = "temp_test_network.npz"
    net_original.save(temp_filename)
    
    # Load model and remove temporary file asset cleanly
    try:
        net_loaded = NeuralNetwork.load(temp_filename)
        loaded_prediction = net_loaded.predict(x)
        
        # Structural validity tests post-load
        assert net_loaded.num_layers == net_original.num_layers, "Loaded layer count does not match original."
        assert net_loaded.activation_name == "tanh", f"Expected activation 'tanh', got '{net_loaded.activation_name}'"
        
        # Critical Test: Verify numerical predictability is perfectly uniform
        assert np.allclose(original_prediction, loaded_prediction), "Loaded network gives different predictions than original network!"
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    print("All save and load consistency tests passed.")


if __name__ == "__main__":
    test_network_initialization()
    test_feedforward_and_prediction()
    test_backpropagation_gradient_shapes()
    test_save_and_load_consistency()
    print("--- All neural_network tests completed successfully! ---")