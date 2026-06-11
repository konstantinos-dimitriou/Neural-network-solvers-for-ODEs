# Author: Konstantinos Dimitriou
# Date: 10/6/2026

# Neural network for regression, adapted from:
# http://neuralnetworksanddeeplearning.com/
# https://www.youtube.com/watch?v=8bNIkfRJZpo
# https://www.youtube.com/watch?v=d9hLNUzLBYI

import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation="tanh"):
        """
        Initialize a fully connected neural network.

        Parameters:
        layer_sizes : tuple or list of ints
            Number of neurons per layer, e.g. (2, 64, 64, 100).
            First entry = input size, last entry = output size.
        activation : str
            Sets the activation function to either "tanh" or "sigmoid".
        """
        # He initialization: scale by 1/sqrt(fan_in) to keep variance stable
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / s[1]**0.5 for s in weight_shapes]
        self.biases  = [np.zeros((size, 1)) for size in layer_sizes[1:]]
        self.num_layers = len(layer_sizes)
        # Store activation and its derivative as attributes
        if activation == "tanh":
            self.activation_name = "tanh"
            self.activation       = lambda z: np.tanh(z)
            self.activation_prime = lambda z: 1.0 - np.tanh(z)**2
        elif activation == "sigmoid":
            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-z))
            self.activation_name = "sigmoid"
            self.activation       = lambda z: 1.0 / (1.0 + np.exp(-z))
            self.activation_prime = lambda z: sigmoid(z) * (1.0 - sigmoid(z))
        else:
            raise ValueError(f"Unknown activation '{activation}'. Choose 'tanh' or 'sigmoid'.")


    def feedforward(self, a):
        """
        Compute the network output for input a.
        Hidden layers use sigmoid; output layer is linear (no activation),
        which is appropriate for regression targets in (0, 1).

        Parameters:
        a : np.ndarray, shape (input_size, 1)

        Returns:
        np.ndarray, shape (output_size, 1)
        """
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            # Apply actiavation to all hidden layers; leave output layer linear
            if i < self.num_layers - 2:
                a = self.activation(z)
                #a = sigmoid(z)
            else:
                a = z   # linear output
        return a

    # Training
    # The following function is currently not used
    def Stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):
        """
        Train via mini-batch stochastic gradient descent.

        Parameters:
        training_data    : list of (x, y) numpy array pairs
        epochs           : int
        mini_batch_size  : int
        learning_rate    : float
        test_data        : list of (x, y) pairs, optional
                           If provided, mean squared error on test set is printed each epoch.
        """
        n = len(training_data)

        for epoch in range(epochs):
            current_lr = learning_rate * (0.98 ** epoch)
            #net._update_mini_batch(mini_batch, current_lr)

            np.random.shuffle(training_data)

            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                #self._update_mini_batch(mini_batch, learning_rate)
                self._update_mini_batch(mini_batch, current_lr)

            if test_data is not None:
                # Compute error
                mse = self.evaluate_mse(test_data)
                #print(f"Epoch {epoch:>3d} | test MSE: {mse:.6f}")
                print(f"Epoch {epoch:>3d} | lr: {current_lr} | test MSE: {mse:.6f}")
            else:
                print(f"Epoch {epoch:>3d} complete")


    def _update_mini_batch(self, mini_batch, eta):
        """
        Apply one gradient descent step averaged over a mini-batch.

        Parameters:
        mini_batch : list of (x, y) pairs
        eta        : float, learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        scale = eta / len(mini_batch)
        self.weights = [w - scale * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - scale * nb for b, nb in zip(self.biases,  nabla_b)]


    def _backprop(self, x, y):
        """
        Compute gradients for a single training sample (x, y) via
        backpropagation.

        Returns:
        (nabla_b, nabla_w) : lists of np.ndarray matching self.biases / self.weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Forward pass, storing activations and pre-activations
        activation = x
        activations = [x]
        zs = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            zs.append(z)
            if i < self.num_layers - 2:
                activation = self.activation(z)
                #activation = sigmoid(z)
            else:
                activation = z      # linear output layer
            activations.append(activation)

        # Backward pass
        # Output layer: linear activation → sigmoid_prime = 1
        delta = self._cost_derivative(activations[-1], y)   # dC/dz at output
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Hidden layers (counting back from second-to-last)
        for l in range(2, self.num_layers):
            z  = zs[-l]
            #sp = sigmoid_prime(z)
            activation_prime = self.activation_prime(z)
            #delta = np.dot(self.weights[-l + 1].T, delta) * sp
            delta = np.dot(self.weights[-l + 1].T, delta) * activation_prime
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_b, nabla_w


    # Evaluation
    def evaluate_mse(self, data):
        """
        Compute mean squared error over a dataset.

        Parameters:
        data : list of (x, y) pairs
        Returns:
        float : average MSE across all samples
        """
        total = sum(
            np.mean((self.feedforward(x) - y) ** 2)
            for x, y in data
        )
        return total / len(data)


    def predict(self, x):
        """Return the network's output for a single input x."""
        return self.feedforward(x)


    # Cost
    def _cost_derivative(self, output_activations, y):
        """Gradient of MSE: dC/da = (a - y)."""
        return output_activations - y
    

    def save(self, filepath):
        """Save weights and biases to a .npz file."""
        np.savez(filepath,
                num_layers=np.array(self.num_layers),
                weights=np.array(self.weights, dtype=object),
                biases=np.array(self.biases,  dtype=object),
                activation=self.activation_name)
        print(f"Network saved to '{filepath}'")


    @classmethod
    def load(cls, filepath):
        """Load a saved network from a .npz file and return a NeuralNetwork instance."""
        data = np.load(filepath, allow_pickle=True)
        activation     = str(data["activation"])
        net = cls.__new__(cls)   # bypass __init__, we fill manually
        net.weights    = list(data["weights"])
        net.biases     = list(data["biases"])
        net.num_layers = int(data["num_layers"])
        # Re-attach the correct activation functions
        if activation == "tanh":
            net.activation       = lambda z: np.tanh(z)
            net.activation_prime = lambda z: 1.0 - np.tanh(z)**2
        elif activation == "sigmoid":
            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-z))
            net.activation       = lambda z: 1.0 / (1.0 + np.exp(-z))
            net.activation_prime = lambda z: sigmoid(z) * (1.0 - sigmoid(z))
        net.activation_name = activation
        print(f"Network loaded from '{filepath}'")
        return net

