from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf
