from __future__ import annotations
from typing import Optional, Any, Literal

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier


class StudentEnsemble:
    """
    LightGBM ensemble with three specialized models joined by VotingClassifier.
    Knowledge distillation via feature augmentation with teacher outputs.

    Three model variants:
    - lgbm_wide: Broad trees, high learning rate, fewer boosting rounds
    - lgbm_deep: Deep trees, moderate learning rate, more boosting rounds
    - lgbm_fast: Fast training, low complexity, regularized
    """

    def __init__(
        self,
        n_members: int = 3,  # Fixed to 3 for wide/deep/fast
        distil_with: Literal["none", "logits", "probs"] = "probs",
        base_params: Optional[dict[str, Any]] = None,
    ):
        if n_members != 3:
            print(
                f"Warning: n_members={n_members} but using fixed 3 variants (wide/deep/fast)"
            )

        self.n_members = 3
        self.distil_with = distil_with

        # Base parameters - will be specialized per variant
        self.base_params = base_params or {}

        self.ensemble: Optional[VotingClassifier] = None
        self.n_classes_: Optional[int] = None

    def _augment(
        self, X: np.ndarray, teacher_outputs: Optional[np.ndarray]
    ) -> np.ndarray:
        """Augment features with teacher outputs if distillation is enabled."""
        if teacher_outputs is None or self.distil_with == "none":
            return X
        if teacher_outputs.ndim == 1:
            teacher_outputs = teacher_outputs[:, None]
        return np.hstack([X, teacher_outputs])

    def _build_specialized_models(self) -> list[tuple[str, lgb.LGBMClassifier]]:
        """Build three specialized LightGBM variants."""

        # Default base configuration
        base_config = {
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            **self.base_params,
        }

        # Wide model: Broad trees, fewer rounds, higher learning rate
        wide_config = {
            **base_config,
            "n_estimators": 400,
            "learning_rate": 0.1,
            "num_leaves": 127,  # Wider trees
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.5,
            "random_state": 42,
        }

        # Deep model: Deeper trees, more rounds, moderate learning rate
        deep_config = {
            **base_config,
            "n_estimators": 800,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 12,  # Deeper trees
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
            "random_state": 43,
        }

        # Fast model: Quick training, regularized, simple
        fast_config = {
            **base_config,
            "n_estimators": 200,
            "learning_rate": 0.15,
            "num_leaves": 31,  # Smaller trees
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
            "random_state": 44,
        }

        models = [
            ("lgbm_wide", lgb.LGBMClassifier(**wide_config)),
            ("lgbm_deep", lgb.LGBMClassifier(**deep_config)),
            ("lgbm_fast", lgb.LGBMClassifier(**fast_config)),
        ]

        return models

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        teacher_logits: Optional[np.ndarray] = None,
        teacher_probs: Optional[np.ndarray] = None,
        class_weight: Optional[str | dict] = "balanced",
    ):
        """Train the ensemble with optional knowledge distillation."""
        self.n_classes_ = int(np.unique(y).size)

        # Choose which teacher signal to append
        teacher_feat = None
        if self.distil_with == "logits":
            teacher_feat = teacher_logits
        elif self.distil_with == "probs":
            teacher_feat = teacher_probs

        X_aug = self._augment(X, teacher_feat)

        # Build specialized models
        models = self._build_specialized_models()

        # Configure objective based on number of classes
        for name, clf in models:
            if self.n_classes_ == 2:
                clf.set_params(objective="binary", class_weight=class_weight)
            else:
                clf.set_params(
                    objective="multiclass",
                    num_class=self.n_classes_,
                    class_weight=class_weight,
                )

        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=models, voting="soft", n_jobs=-1, flatten_transform=True
        )

        # Fit the ensemble
        self.ensemble.fit(X_aug, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        teacher_logits: Optional[np.ndarray] = None,
        teacher_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Make predictions using the ensemble."""
        assert self.ensemble is not None, "Model not fitted"

        teacher_feat = None
        if self.distil_with == "logits":
            teacher_feat = teacher_logits
        elif self.distil_with == "probs":
            teacher_feat = teacher_probs

        X_aug = self._augment(X, teacher_feat)
        return self.ensemble.predict(X_aug)

    def predict_proba(
        self,
        X: np.ndarray,
        teacher_logits: Optional[np.ndarray] = None,
        teacher_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Make probability predictions using the ensemble."""
        assert self.ensemble is not None, "Model not fitted"

        teacher_feat = None
        if self.distil_with == "logits":
            teacher_feat = teacher_logits
        elif self.distil_with == "probs":
            teacher_feat = teacher_probs

        X_aug = self._augment(X, teacher_feat)
        return self.ensemble.predict_proba(X_aug)

    def get_model_names(self) -> list[str]:
        """Get names of the constituent models."""
        if self.ensemble is None:
            return ["lgbm_wide", "lgbm_deep", "lgbm_fast"]
        return [name for name, _ in self.ensemble.estimators]
