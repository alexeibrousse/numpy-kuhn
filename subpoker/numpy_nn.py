import numpy as np
"""
Feedforward NumPy neural network used for policy gradient training.
Input is a numpy ndarray, passed through a ReLU-activated hidden layer for a softmax probability distribution output.
"""

class NumNet():
    """Simple two-layer feedforward network for REINFORCE."""
    def __init__(self, input_size:int, hidden_size:int, output_size:int, learning_rate: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Xavier initialization and random initialization for the biases
        self.W1 = np.random.randn(input_size, hidden_size)  * np.sqrt(6 / (input_size + hidden_size))
        self.b1 = np.random.randn(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt( 6 / (hidden_size + output_size))
        self.b2 = np.random.randn(output_size)


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        X (np.ndarray): Input vector of shape (input_size,).
        """
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        probs = self.softmax(self.z2)
        
        return probs


    def softmax(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to convert logits to a probability distribution.
        Returns np.ndarray of probabilities summing to 1
        """
        e_X = np.exp(X- np.max(X)) 
        return e_X / np.sum(e_X)


    def backward(self, X: np.ndarray, action_taken: int, advantage: float, probs: np.ndarray) -> tuple:
        """
        Computes gradients for all network parameters using REINFORCE method.

        X (np.ndarray): Input vector used in forward pass.
        action_taken (int): Index of the action taken.
        advantage (float): Reward signal
        probs (np.ndarray): Output probabilities from forward().
        """

        dlog = probs.copy()
        dlog[action_taken] -= 1
        dlog *= advantage

        dW2 = np.outer(self.a1, dlog)
        db2 = dlog
        
        da1 = self.W2 @ dlog
        dz1 = da1 * (self.z1 > 0)

        dW1 = np.outer(X, dz1)
        db1 = dz1

        return dW1, db1, dW2, db2
    

    def update(self, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray) -> None:
        """ Applies gradient descent update to all parameters. """
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
