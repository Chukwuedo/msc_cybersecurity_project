"""
Random Forest Student for Knowledge Distillation
===============================================

This module implements a Random Forest student that learns
from the tree ensemble teacher through knowledge distillation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional


class StudentTree:
    """
    Random Forest student for knowledge distillation.

    Uses knowledge from teacher ensemble to improve Random Forest performance
    through feature augmentation and soft target training.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str] = "sqrt",
        class_weight: str = "balanced",
        random_state: int = 42,
        use_soft_targets: bool = True,
        distillation_alpha: float = 0.7,
        temperature: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.use_soft_targets = use_soft_targets
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature

        self.tree = None
        self.n_classes_ = None

    def _augment_features(
        self, X: np.ndarray, teacher_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Augment features with teacher knowledge."""
        if teacher_probs is None or not self.use_soft_targets:
            return X

        # Add teacher probabilities as additional features
        return np.hstack([X, teacher_probs])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        teacher_logits: Optional[np.ndarray] = None,
        teacher_probs: Optional[np.ndarray] = None,
        class_weight: Optional[str] = None,
    ) -> "StudentTree":
        """
        Fit the decision tree with knowledge distillation.

        Args:
            X: Training features
            y: Training labels
            teacher_logits: For compatibility (ignored)
            teacher_probs: Soft targets from teacher
            class_weight: Class weighting (overrides init parameter)
        """

        # Use passed class_weight if provided
        if class_weight is not None:
            self.class_weight = class_weight

        self.n_classes_ = len(np.unique(y))
        print(f"Training Random Forest student for {self.n_classes_} classes...")

        # Augment features with teacher knowledge
        X_augmented = self._augment_features(X, teacher_probs)

        # Prepare sample weights from teacher confidence
        sample_weights = None
        if teacher_probs is not None and self.use_soft_targets:
            teacher_confidence = np.max(teacher_probs, axis=1)
            sample_weights = self.distillation_alpha * teacher_confidence + (
                1 - self.distillation_alpha
            ) * np.ones_like(teacher_confidence)

        # Create and fit Random Forest
        cw = self.class_weight if self.class_weight == "balanced" else None
        mf = self.max_features if self.max_features in ["sqrt", "log2"] else "sqrt"

        self.tree = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=mf,
            class_weight=cw,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Fit with sample weights if available
        if sample_weights is not None:
            self.tree.fit(X_augmented, y, sample_weight=sample_weights)
        else:
            self.tree.fit(X_augmented, y)

        print("✓ Random Forest student training complete!")
        return self

    def predict(
        self, X: np.ndarray, teacher_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Make predictions with feature augmentation."""
        if self.tree is None:
            raise ValueError("Model must be fitted before making predictions")
        X_augmented = self._augment_features(X, teacher_probs)
        return self.tree.predict(X_augmented)

    def predict_proba(
        self, X: np.ndarray, teacher_probs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get prediction probabilities with feature augmentation."""
        if self.tree is None:
            raise ValueError("Model must be fitted before making predictions")
        X_augmented = self._augment_features(X, teacher_probs)
        return self.tree.predict_proba(X_augmented)

    def score(
        self, X: np.ndarray, y: np.ndarray, teacher_probs: Optional[np.ndarray] = None
    ) -> float:
        """Calculate accuracy score."""
        if self.tree is None:
            raise ValueError("Model must be fitted before scoring")
        X_augmented = self._augment_features(X, teacher_probs)
        return float(self.tree.score(X_augmented, y))

    def get_model_names(self) -> list[str]:
        """Get model name for compatibility."""
        return ["random_forest"]


# Alias for compatibility with existing code
StudentEnsemble = StudentTree


def train_student(
    X_train: np.ndarray,
    y_train: np.ndarray,
    teacher_probs: Optional[np.ndarray] = None,
    *,
    n_estimators: int = 50,
    max_depth: int = 10,
    random_state: int = 42,
    use_soft_targets: bool = True,
    distillation_alpha: float = 0.7,
) -> StudentTree:
    """
    Train a Random Forest student for knowledge distillation.

    Args:
        X_train: Training features
        y_train: Training labels
        teacher_probs: Teacher probability predictions
        n_estimators: Number of trees in forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
        use_soft_targets: Whether to use teacher soft targets
        distillation_alpha: Weight for teacher knowledge

    Returns:
        Trained Random Forest student
    """

    student = StudentTree(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        use_soft_targets=use_soft_targets,
        distillation_alpha=distillation_alpha,
    )

    return student.fit(X_train, y_train, teacher_probs=teacher_probs)
