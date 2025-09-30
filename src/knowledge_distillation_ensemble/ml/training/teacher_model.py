"""
Tree Ensemble Teacher for Knowledge Distillation
===============================================

This module implements a powerful tree ensemble teacher that combines:
- LightGBM (200+ estimators)
- Extra Trees (200+ estimators)
- XGBoost (200+ estimators)

With soft voting for knowledge distillation.
"""

import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from typing import Optional
import lightgbm as lgb
import xgboost as xgb


class TreeEnsembleTeacher:
    """
    Powerful tree ensemble teacher for knowledge distillation.
    Combines multiple strong tree-based models with soft voting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        """
        Initialize tree ensemble teacher.

        Args:
            n_estimators: Number of trees per base model
            random_state: Random seed
            class_weight: Class weighting strategy
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.class_weight = class_weight
        self.ensemble: Optional[VotingClassifier] = None

    def _create_base_models(self):
        """Create the base tree models for the ensemble."""
        # Handle class_weight properly for sklearn
        cw = self.class_weight if self.class_weight == "balanced" else None

        # LightGBM Classifier
        lgb_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=15,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=-1,
            verbose=-1,
        )

        # Extra Trees Classifier
        et = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=cw,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=15,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )

        return lgb_model, et, xgb_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeEnsembleTeacher":
        """Fit the ensemble teacher on training data."""

        # Ensure X is numpy array
        X = np.asarray(X)
        y = np.asarray(y)

        n_classes = len(np.unique(y))
        print(f"Training tree ensemble teacher with {n_classes} classes...")

        # Create base models
        lgb_model, et, xgb_model = self._create_base_models()

        # Create voting ensemble with soft voting for probability outputs
        self.ensemble = VotingClassifier(
            estimators=[("lgb", lgb_model), ("et", et), ("xgb", xgb_model)],
            voting="soft",  # Critical for knowledge distillation
        )

        print("Fitting ensemble components...")
        print("  - Training LightGBM...")
        print("  - Training Extra Trees...")
        print("  - Training XGBoost...")

        # Fit the ensemble
        self.ensemble.fit(X, y)

        print("✓ Tree ensemble teacher training complete!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make hard predictions."""
        if self.ensemble is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X)
        return np.asarray(self.ensemble.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get soft targets for knowledge distillation."""
        if self.ensemble is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X)
        return self.ensemble.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        if self.ensemble is None:
            raise ValueError("Model must be fitted before scoring")
        X = np.asarray(X)
        y = np.asarray(y)
        return float(self.ensemble.score(X, y))


def train_teacher(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int = 200,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> TreeEnsembleTeacher:
    """
    Train a powerful tree ensemble teacher for knowledge distillation.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of estimators per base model
        class_weight: How to handle class imbalance
        random_state: Random seed for reproducibility

    Returns:
        Trained tree ensemble teacher
    """

    teacher = TreeEnsembleTeacher(
        n_estimators=n_estimators, random_state=random_state, class_weight=class_weight
    )

    return teacher.fit(X_train, y_train)


def save_teacher_model(
    model: TreeEnsembleTeacher,
    X: np.ndarray,
    y: np.ndarray,
    filepath: str = "teacher_model.joblib",
) -> None:
    """Save the teacher model after training."""
    if model.ensemble is None:
        model.fit(X, y)

    joblib.dump(model, filepath)
    print(f"Teacher model saved to {filepath}")


def load_teacher_model(filepath: str = "teacher_model.joblib") -> TreeEnsembleTeacher:
    """Load a trained teacher model."""
    model = joblib.load(filepath)
    print(f"Teacher model loaded from {filepath}")
    return model
