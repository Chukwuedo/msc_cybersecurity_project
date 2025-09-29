from __future__ import annotations
from typing import Optional, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def train_benchmark_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: Literal["random_forest", "gradient_boosting", "decision_tree"] = "random_forest",
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier | GradientBoostingClassifier | DecisionTreeClassifier:
    """Train a competitive baseline model for fair comparison with the student ensemble."""
    
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 6,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    elif model_type == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    clf.fit(X, y)
    return clf


# Backwards compatibility
def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a simple decision tree (kept for backwards compatibility)."""
    return train_benchmark_ensemble(
        X, y,
        model_type="decision_tree",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
