"""Numpy implementation of some activation functions."""

import numpy as np


def sigmoid(x: np.array) -> np.array:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    """Softmax operation."""
    # Stable softmax implementation.
    z = x - x.max(axis=1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator
