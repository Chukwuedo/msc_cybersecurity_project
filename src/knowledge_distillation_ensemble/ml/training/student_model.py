from __future__ import annotations
from typing import Optional, List, Any, Literal

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier


class StudentEnsemble:
    """
    LightGBM **ensemble** joined with a soft **VotingClassifier**.
    Distillation is via **feature augmentation** with teacher outputs:
      - `distil_with="logits"` appends teacher logits
      - `distil_with="probs"` appends (optionally calibrated) probabilities
    """

    def __init__(
        self,
        n_members: int = 5,
        distil_with: Literal["none", "logits", "probs"] = "probs",
        base_params: Optional[dict[str, Any]] = None,
    ):
        self.n_members = n_members
        self.distil_with = distil_with
        base = dict(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        if base_params:
            base.update(base_params)
        self.base_params = base
        self.ensemble: Optional[VotingClassifier] = None
        self.n_classes_: Optional[int] = None

    def _augment(
        self, X: np.ndarray, teacher_outputs: Optional[np.ndarray]
    ) -> np.ndarray:
        if teacher_outputs is None or self.distil_with == "none":
            return X
        if teacher_outputs.ndim == 1:
            teacher_outputs = teacher_outputs[:, None]
        return np.hstack([X, teacher_outputs])

    def _build_members(self) -> List[tuple[str, lgb.LGBMClassifier]]:
        members: List[tuple[str, lgb.LGBMClassifier]] = []
        for i in range(self.n_members):
            params = dict(self.base_params)
            params["random_state"] = (
                self.base_params.get("random_state", 42) + i
            )  # diversify seeds
            clf = lgb.LGBMClassifier(**params)
            members.append((f"lgb{i}", clf))
        return members

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        teacher_logits: Optional[np.ndarray] = None,
        teacher_probs: Optional[np.ndarray] = None,
        class_weight: Optional[str | dict] = "balanced",
    ):
        self.n_classes_ = int(np.unique(y).size)
        # Choose which teacher signal to append
        teacher_feat = None
        if self.distil_with == "logits":
            teacher_feat = teacher_logits
        elif self.distil_with == "probs":
            teacher_feat = teacher_probs

        X_aug = self._augment(X, teacher_feat)

        members = []
        for name, clf in self._build_members():
            clf.set_params(
                **(
                    {"objective": "binary"}
                    if self.n_classes_ == 2
                    else {"objective": "multiclass", "num_class": self.n_classes_}
                ),
                class_weight=class_weight,
            )
            members.append((name, clf))

        self.ensemble = VotingClassifier(
            estimators=members, voting="soft", n_jobs=-1, flatten_transform=True
        )
        self.ensemble.fit(X_aug, y)
        return self

    def predict(
        self,
        X: np.ndarray,
        teacher_logits: Optional[np.ndarray] = None,
        teacher_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert self.ensemble is not None
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
        assert self.ensemble is not None
        teacher_feat = None
        if self.distil_with == "logits":
            teacher_feat = teacher_logits
        elif self.distil_with == "probs":
            teacher_feat = teacher_probs
        X_aug = self._augment(X, teacher_feat)
        return self.ensemble.predict_proba(X_aug)
