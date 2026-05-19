"""
Feedforward NumPy neural network used for policy gradient training.
Input is a numpy ndarray, passed through a ReLU-activated hidden layer for a softmax probability distribution output.
"""

import numpy as np


class NumNet:
    """Simple two-layer feedforward network with Adam Optimizer for REINFORCE."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float, 
                 beta1: float, beta2: float, epsilon: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

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

        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)


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
    

    def _adam_step(self, param: np.ndarray, grad: np.ndarray, first_moment: np.ndarray, second_moment: np.ndarray,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs a single Adam update step for a given parameter."""
        first_moment = self.beta1 * first_moment + (1 - self.beta1) * grad
        second_moment = self.beta2 * second_moment + (1 - self.beta2) * (grad ** 2)

        first_hat = first_moment / (1 - self.beta1 ** self.t)
        second_hat = second_moment / (1 - self.beta2 ** self.t)
        param -= self.lr * first_hat / (np.sqrt(second_hat) + self.epsilon)

        return param, first_moment, second_moment


    def update(self, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray) -> None:
        """Apply updates to all parameters."""
        self.t += 1

        self.W1, self.mW1, self.vW1 = self._adam_step(self.W1, dW1, self.mW1, self.vW1)
        self.b1, self.mb1, self.vb1 = self._adam_step(self.b1, db1, self.mb1, self.vb1)
        self.W2, self.mW2, self.vW2 = self._adam_step(self.W2, dW2, self.mW2, self.vW2)
        self.b2, self.mb2, self.vb2 = self._adam_step(self.b2, db2, self.mb2, self.vb2)


    def save(self, path: str) -> None:
        """Persist the final network parameters for inference-time reuse."""
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )


    @classmethod
    def load(cls, path: str, cfg: dict) -> "NumNet":
        """Reconstruct a network from config metadata and saved parameters."""
        net = cls(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            output_size=cfg["output_size"],
            learning_rate=cfg["initial_learning_rate"],
            beta1=cfg["adam_beta1"],
            beta2=cfg["adam_beta2"],
            epsilon=cfg["adam_epsilon"],
        )

        params = np.load(path)
        net.W1 = params["W1"]
        net.b1 = params["b1"]
        net.W2 = params["W2"]
        net.b2 = params["b2"]
        return net
