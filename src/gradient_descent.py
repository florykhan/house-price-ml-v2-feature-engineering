from __future__ import annotations

from typing import Optional

import numpy as np


class LinearRegressionGD:
    """
    Linear Regression trained with batch gradient descent.

    This is just the skeleton for now.
    We'll implement:
    - fit()
    - predict()
    - loss history tracking
    - optional L1/L2 regularization
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """
        Train the model using gradient descent.
        To be implemented.
        """
        # TODO: implement gradient descent here
        raise NotImplementedError("fit() not implemented yet.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned weights.
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.weights + self.bias
