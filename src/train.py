from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .config import Config
from .data_loader import load_raw_data, train_test_split_data
from .feature_engineering import apply_feature_engineering
from .preprocessing import compute_standardization_params, apply_standardization
from .gradient_descent import LinearRegressionGD
from .evaluation import evaluate_regression


def main() -> None:
    """
    Full training pipeline:
    1. Load config
    2. Load raw data
    3. Apply feature engineering
    4. Split train/test
    5. Compute standardization params on train
    6. Scale train and test
    7. Train model (gradient descent)
    8. Evaluate
    """
    config = Config()

    df = load_raw_data(config)

    df_fe = apply_feature_engineering(df)

    X_train, X_test, y_train, y_test = train_test_split_data(
        df_fe, target_col="median_house_value", config=config
    )

    std_params = compute_standardization_params(X_train)
    X_train_scaled = apply_standardization(X_train, std_params)
    X_test_scaled = apply_standardization(X_test, std_params)

    model = LinearRegressionGD(
        learning_rate=config.learning_rate,
        n_iterations=config.n_iterations,
        l1_lambda=config.lambda_ if config.use_l1 else 0,
        l2_lambda=config.lambda_ if config.use_l2 else 0,
    )

    model.fit(
        X_train_scaled.to_numpy(),
        y_train.to_numpy(),
    )

    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE + Regularization)")
    plt.title("Training Loss Curve")
    plt.show()

    y_pred = model.predict(X_test_scaled.to_numpy())

    metrics = evaluate_regression(y_test.to_numpy(), y_pred)
    print(metrics)


if __name__ == "__main__":
    main()
