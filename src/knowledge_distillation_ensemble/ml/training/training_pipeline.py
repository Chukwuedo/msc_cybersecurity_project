from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Literal, Optional

import numpy as np
import polars as pl
from joblib import dump

from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix
)

from teacher_model import train_teacher, predict_logits, logits_to_probs, calibrate_temperature, logits_to_calibrated_probs
from student_model import StudentEnsemble
from benchmark_model import train_decision_tree

# ---- Configuration (edit as needed) ----
FEATURES: List[str] = [
    "flow_bytes_per_second",
    "flow_packets_per_second",
    "packet_length_mean",
    "flow_duration",
    "average_packet_size",
    "total_packets",
    "total_bytes",
    "header_length_total",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "ece_flag_count",
    "cwr_flag_count",
    "urg_flag_count",
    "packet_length_min",
    "packet_length_max",
    "packet_length_std",
    "packet_length_range",
    "forward_packets_per_second",
    "backward_packets_per_second",
    "flow_iat_mean",
]

TARGETS = {
    "binary": "label_binary",
    "multiclass": "label_multiclass",
}

# Ensure scikit-learn emits Polars from transformers
set_config(transform_output="polars")


def build_preprocessor(robust: bool = True) -> ColumnTransformer:
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(unit_variance=True) if robust else StandardScaler()),
    ]
    preproc = ColumnTransformer([
        ("num", Pipeline(num_steps), FEATURES),
    ], remainder="drop", n_jobs=-1)
    return preproc


def load_frames(train_path: Path, test_path: Path, target_col: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    train = pl.read_parquet(train_path)
    test = pl.read_parquet(test_path)

    X_train = train.select(FEATURES)
    y_train = train.select(target_col).to_series()
    X_test = test.select(FEATURES)
    y_test = test.select(target_col).to_series()
    return X_train, X_test, y_train, y_test


def to_numpy_after_fit_transform(preproc: ColumnTransformer, X_train_pl: pl.DataFrame, X_test_pl: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Fit on train Polars; transform train and test; convert once to NumPy here
    X_train_proc_pl = preproc.fit_transform(X_train_pl)
    X_test_proc_pl = preproc.transform(X_test_pl)

    # Both should be Polars DataFrames thanks to set_output; convert to NumPy
    X_train_np = X_train_proc_pl.to_numpy()
    X_test_np = X_test_proc_pl.to_numpy()
    return X_train_np, X_test_np


def evaluate(task: Literal["binary", "multiclass"], y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }
    try:
        if task == "binary":
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return metrics


def run_task(
    *,
    train_path: Path,
    test_path: Path,
    task: Literal["binary", "multiclass"],
    robust_scaler: bool = True,
    teacher_epochs: int = 15,
    teacher_batch_size: int = 2048,
    n_student_members: int = 5,
    distil_with: Literal["logits", "probs"] = "probs",
    unseen_path: Optional[Path] = None,
) -> None:
    target_col = TARGETS[task]
    print(f"
=== Task: {task} (target: {target_col}) ===")

    # 1) Load pre-split frames
    X_train_pl, X_test_pl, y_train_pl, y_test_pl = load_frames(train_path, test_path, target_col)

    # 2) Preprocess (fit on train only)
    preproc = build_preprocessor(robust=robust_scaler)
    X_train_np, X_test_np = to_numpy_after_fit_transform(preproc, X_train_pl, X_test_pl)
    y_train = y_train_pl.to_numpy()
    y_test = y_test_pl.to_numpy()

    # 3) Train teacher (Lightning)
    teacher = train_teacher(
        X_train_np, y_train,
        X_val=X_test_np, y_val=y_test,
        max_epochs=teacher_epochs, batch_size=teacher_batch_size,
    )

    # 3b) Calibrate teacher by temperature scaling on the held-out test set (or a validation split if available)
    T = calibrate_temperature(teacher, X_test_np, y_test)
    print(f"Calibrated teacher temperature: {T:.3f}")

    # 4) Teacher outputs for distillation
    logits_train = predict_logits(teacher, X_train_np)
    logits_test = predict_logits(teacher, X_test_np)
    if distil_with == "probs":
        teacher_train_signal = logits_to_calibrated_probs(teacher, logits_train)
        teacher_test_signal = logits_to_calibrated_probs(teacher, logits_test)
    else:
        teacher_train_signal = logits_train
        teacher_test_signal = logits_test

    # 5) Student – LightGBM **ensemble** with voting, using teacher signal
    student = StudentEnsemble(n_members=n_student_members, distil_with=distil_with)
    student.fit(X_train_np, y_train,
                teacher_logits=(teacher_train_signal if distil_with == "logits" else None),
                teacher_probs=(teacher_train_signal if distil_with == "probs" else None),
                class_weight="balanced")

    y_pred_student = student.predict(
        X_test_np,
        teacher_logits=(teacher_test_signal if distil_with == "logits" else None),
        teacher_probs=(teacher_test_signal if distil_with == "probs" else None),
    )
    y_proba_student = student.predict_proba(
        X_test_np,
        teacher_logits=(teacher_test_signal if distil_with == "logits" else None),
        teacher_probs=(teacher_test_signal if distil_with == "probs" else None),
    )

    # 6) Benchmark – simple Decision Tree (no logits)
    dt = train_decision_tree(X_train_np, y_train, class_weight="balanced")
    y_pred_dt = dt.predict(X_test_np)

    # 7) Evaluate on test
    metrics_student = evaluate(task, y_test, y_pred_student, y_proba_student)
    metrics_dt = evaluate(task, y_test, y_pred_dt, np.zeros((len(y_test), len(np.unique(y_train)))))

    print("
-- Student (LGBM Ensemble + distillation) [TEST] --")
    for k, v in metrics_student.items():
        if k != "classification_report":
            print(f"{k}: {v}")
    print("
Classification report (student):
" + metrics_student["classification_report"])  # type: ignore

    print("
-- Benchmark (Decision Tree) [TEST] --")
    for k, v in metrics_dt.items():
        if k != "classification_report":
            print(f"{k}: {v}")
    print("
Classification report (decision tree):
" + metrics_dt["classification_report"])  # type: ignore

    # 8) Optional: Evaluate on **unseen** dataset to demonstrate robustness
    if unseen_path is not None:
        _, X_unseen_pl, _, y_unseen_pl = load_frames(unseen_path, unseen_path, target_col)  # only using features/target
        X_unseen_np = preproc.transform(X_unseen_pl).to_numpy()
        y_unseen = y_unseen_pl.to_numpy()
        logits_unseen = predict_logits(teacher, X_unseen_np)
        if distil_with == "probs":
            teacher_unseen_signal = logits_to_calibrated_probs(teacher, logits_unseen)
        else:
            teacher_unseen_signal = logits_unseen
        y_pred_student_u = student.predict(
            X_unseen_np,
            teacher_logits=(teacher_unseen_signal if distil_with == "logits" else None),
            teacher_probs=(teacher_unseen_signal if distil_with == "probs" else None),
        )
        y_proba_student_u = student.predict_proba(
            X_unseen_np,
            teacher_logits=(teacher_unseen_signal if distil_with == "logits" else None),
            teacher_probs=(teacher_unseen_signal if distil_with == "probs" else None),
        )
        metrics_student_u = evaluate(task, y_unseen, y_pred_student_u, y_proba_student_u)
        y_pred_dt_u = dt.predict(X_unseen_np)
        metrics_dt_u = evaluate(task, y_unseen, y_pred_dt_u, np.zeros((len(y_unseen), len(np.unique(y_train)))))

        print("
== Robustness on UNSEEN dataset ==")
        print("-- Student (ensemble) [UNSEEN] --")
        for k, v in metrics_student_u.items():
            if k != "classification_report":
                print(f"{k}: {v}")
        print("
Classification report (student / unseen):
" + metrics_student_u["classification_report"])  # type: ignore

        print("
-- Decision Tree [UNSEEN] --")
        for k, v in metrics_dt_u.items():
            if k != "classification_report":
                print(f"{k}: {v}")
        print("
Classification report (decision tree / unseen):
" + metrics_dt_u["classification_report"])  # type: ignore

    # 9) Persist artefacts
    outdir = Path("models")
    outdir.mkdir(parents=True, exist_ok=True)
    dump(preproc, outdir / f"preproc_{task}.joblib")
    dump(student, outdir / f"student_ensemble_{task}.joblib")
    dump(dt, outdir / f"decision_tree_{task}.joblib")
    dump(teacher, outdir / f"teacher_{task}.pt")
    print(f"Saved: {outdir}/preproc_{task}.joblib, student_ensemble_{task}.joblib, decision_tree_{task}.joblib, teacher_{task}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network intrusion training – teacher/student pipeline")
    parser.add_argument("--train", type=Path, default=Path("train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("test.parquet"))
    parser.add_argument("--unseen", type=Path, default=None)
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--standard", action="store_true", help="Use StandardScaler instead of RobustScaler")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--members", type=int, default=5, help="Number of LightGBM models in the student ensemble")
    parser.add_argument("--distil", choices=["logits", "probs"], default="probs", help="Use teacher logits or calibrated probabilities for distillation")
    args = parser.parse_args()

    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    for t in tasks:
        run_task(
            train_path=args.train,
            test_path=args.test,
            task=t, robust_scaler=not args.standard,
            teacher_epochs=args.epochs,
            teacher_batch_size=args.batch,
            n_student_members=args.members,
            distil_with=args.distil,
            unseen_path=args.unseen,
        )