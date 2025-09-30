"""
Benchmark Model for Knowledge Distillation Evaluation
====================================================

This module provides a simple logistic regression baseline model
for fair comparison with the student model in knowledge distillation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_benchmark_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 2000,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic regression baseline model for comparison.

    Args:
        X: Training features
        y: Training labels
        max_iter: Maximum iterations for convergence
        class_weight: How to handle class imbalance
        random_state: Random seed for reproducibility

    Returns:
        Trained logistic regression model
    """

    print("Training logistic regression benchmark...")

    # Use 'lbfgs' solver with default multi_class behavior
    # (automatically handles multinomial for multiclass, ovr for binary)
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
        n_jobs=-1,  # Use all available cores
    )

    clf.fit(X, y)

    print("✓ Logistic regression benchmark training complete!")
    return clf


# Alias for backward compatibility
train_benchmark_ensemble = train_benchmark_model
