"""
Feedforward NumPy neural network used for policy gradient training.
Input is a numpy ndarray, passed through a ReLU-activated hidden layer for a softmax probability distribution output.
"""

import numpy as np


class NumNet:
    """Simple two-layer feedforward network for REINFORCE."""
    def __init__(self, input_size:int, hidden_size:int, output_size:int, learning_rate: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Storing intermediate values
        self.state_features = np.zeros(input_size)
        self.z1 = np.zeros(hidden_size)
        self.a1 = np.zeros(hidden_size)
        self.z2 = np.zeros(output_size)

        # Input to Hidden
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)

        # Hidden to Output
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)


    def forward(self, state_features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        state_features (np.ndarray): Input vector of shape (input_size,).
        """
        self.state_features = state_features
        self.z1 = state_features @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        probs = self.softmax(self.z2)
        
        return probs


    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to convert logits to a probability distribution.
        Returns np.ndarray of probabilities summing to 1
        """
        exp_logits = np.exp(logits - np.max(logits)) 
        return exp_logits / np.sum(exp_logits)


    def backward(self, action_taken: int, advantage: float, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes gradients for all network parameters using REINFORCE method.

        action_taken (int): Index of the action taken.
        advantage (float): Scalar weighting the policy gradient, return minus baseline
        probs (np.ndarray): Output probabilities from forward().
        """

        dz2 = probs.copy()
        dz2[action_taken] -= 1
        dz2 *= advantage

        dW2 = np.outer(self.a1, dz2)
        db2 = dz2
        
        da1 = self.W2 @ dz2
        dz1 = da1 * (self.z1 > 0)

        dW1 = np.outer(self.state_features, dz1)
        db1 = dz1

        return dW1, db1, dW2, db2
    

    def update(self, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray) -> None:
        """ Applies gradient descent update to all parameters. """
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
