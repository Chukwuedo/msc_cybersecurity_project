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
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from .teacher_model import (
    train_teacher,
    predict_logits,
    logits_to_probs,
    calibrate_temperature,
    logits_to_calibrated_probs,
)
from .student_model import StudentEnsemble
from .benchmark_model import train_benchmark_ensemble
from .train_test_split import get_stratified_split_lazy, StratifiedSplit
from ..data.analysis_builder import create_analysis_parquet
from ...config.settings import Settings

# ---- Configuration ----
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
    """Build preprocessing pipeline with imputation and scaling."""
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler(unit_variance=True) if robust else StandardScaler()),
    ]
    preproc = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), FEATURES),
        ],
        remainder="drop",
        n_jobs=-1,
    )
    return preproc


def load_from_splits(
    split: StratifiedSplit, unseen_path: Optional[Path], target_col: str
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.Series,
    pl.Series,
    Optional[pl.DataFrame],
    Optional[pl.Series],
]:
    """Load train/test from stratified split and optional unseen data."""
    # Get train/test from lazy splits
    train_df = split.train.select(FEATURES + [target_col]).collect()
    test_df = split.test.select(FEATURES + [target_col]).collect()

    X_train = train_df.select(FEATURES)
    y_train = train_df.select(target_col).to_series()
    X_test = test_df.select(FEATURES)
    y_test = test_df.select(target_col).to_series()

    # Optional unseen data
    X_unseen, y_unseen = None, None
    if unseen_path is not None:
        unseen_df = pl.read_parquet(unseen_path)
        X_unseen = unseen_df.select(FEATURES)
        y_unseen = unseen_df.select(target_col).to_series()

    return X_train, X_test, y_train, y_test, X_unseen, y_unseen


def to_numpy_after_fit_transform(
    preproc: ColumnTransformer,
    X_train_pl: pl.DataFrame,
    X_test_pl: pl.DataFrame,
    X_unseen_pl: Optional[pl.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Fit preprocessor on train and transform all datasets."""
    X_train_proc_pl = preproc.fit_transform(X_train_pl)
    X_test_proc_pl = preproc.transform(X_test_pl)

    X_train_np = X_train_proc_pl.to_numpy()
    X_test_np = X_test_proc_pl.to_numpy()

    X_unseen_np = None
    if X_unseen_pl is not None:
        X_unseen_proc_pl = preproc.transform(X_unseen_pl)
        X_unseen_np = X_unseen_proc_pl.to_numpy()

    return X_train_np, X_test_np, X_unseen_np


def evaluate(
    task: Literal["binary", "multiclass"],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }

    if y_proba is not None:
        try:
            if task == "binary":
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    return metrics


def run_task(
    *,
    task: Literal["binary", "multiclass"],
    dataset_name: str = "ciciot2023",
    unseen_dataset: str = "cicdiad2024",
    robust_scaler: bool = True,
    teacher_epochs: int = 15,
    teacher_batch_size: int = 2048,
    n_student_members: int = 3,
    distil_with: Literal["logits", "probs"] = "probs",
    seed: int = 42,
) -> dict:
    """Run the complete training and evaluation pipeline."""

    target_col = TARGETS[task]
    print(f"\n=== Task: {task} (target: {target_col}) ===")

    # Initialize settings
    settings = Settings()

    # 1) Get stratified split for training dataset
    print("Loading stratified train/test split...")
    split = get_stratified_split_lazy(
        dataset_name=dataset_name, label_col=target_col, train_ratio=0.9, seed=seed
    )

    # 2) Create analysis dataset for unseen evaluation
    unseen_path = None
    if unseen_dataset:
        print(f"Preparing unseen dataset: {unseen_dataset}")
        unseen_path = create_analysis_parquet(
            unseen_dataset, seed=seed, overwrite=False
        )

    # 3) Load and prepare data
    X_train_pl, X_test_pl, y_train_pl, y_test_pl, X_unseen_pl, y_unseen_pl = (
        load_from_splits(split, unseen_path, target_col)
    )

    # 4) Preprocess (fit on train only)
    print("Preprocessing data...")
    preproc = build_preprocessor(robust=robust_scaler)
    X_train_np, X_test_np, X_unseen_np = to_numpy_after_fit_transform(
        preproc, X_train_pl, X_test_pl, X_unseen_pl
    )

    y_train = y_train_pl.to_numpy()
    y_test = y_test_pl.to_numpy()
    y_unseen = y_unseen_pl.to_numpy() if y_unseen_pl is not None else None

    print(f"Train: {X_train_np.shape}, Test: {X_test_np.shape}")
    if X_unseen_np is not None:
        print(f"Unseen: {X_unseen_np.shape}")

    # 5) Train teacher (PyTorch Lightning)
    print("Training teacher model...")
    teacher = train_teacher(
        X_train_np,
        y_train,
        X_val=X_test_np,
        y_val=y_test,
        max_epochs=teacher_epochs,
        batch_size=teacher_batch_size,
        seed=seed,
    )

    # 6) Calibrate teacher temperature
    print("Calibrating teacher temperature...")
    T = calibrate_temperature(teacher, X_test_np, y_test)
    print(f"Calibrated temperature: {T:.3f}")

    # 7) Generate teacher outputs for distillation
    print("Generating teacher predictions...")
    logits_train = predict_logits(teacher, X_train_np)
    logits_test = predict_logits(teacher, X_test_np)

    if distil_with == "probs":
        teacher_train_signal = logits_to_calibrated_probs(teacher, logits_train)
        teacher_test_signal = logits_to_calibrated_probs(teacher, logits_test)
    else:
        teacher_train_signal = logits_train
        teacher_test_signal = logits_test

    # 8) Train student ensemble
    print("Training student ensemble...")
    student = StudentEnsemble(n_members=n_student_members, distil_with=distil_with)
    student.fit(
        X_train_np,
        y_train,
        teacher_logits=(teacher_train_signal if distil_with == "logits" else None),
        teacher_probs=(teacher_train_signal if distil_with == "probs" else None),
        class_weight="balanced",
    )

    # 9) Train benchmark model
    print("Training benchmark model...")
    benchmark = train_benchmark_ensemble(
        X_train_np,
        y_train,
        model_type="random_forest",
        n_estimators=100,
        class_weight="balanced",
    )

    # 10) Evaluate on test set
    print("\n--- Test Set Evaluation ---")

    # Teacher predictions
    teacher_test_probs = logits_to_calibrated_probs(teacher, logits_test)
    teacher_test_pred = np.argmax(teacher_test_probs, axis=1)

    # Student predictions
    student_test_pred = student.predict(
        X_test_np,
        teacher_logits=(teacher_test_signal if distil_with == "logits" else None),
        teacher_probs=(teacher_test_signal if distil_with == "probs" else None),
    )
    student_test_proba = student.predict_proba(
        X_test_np,
        teacher_logits=(teacher_test_signal if distil_with == "logits" else None),
        teacher_probs=(teacher_test_signal if distil_with == "probs" else None),
    )

    # Benchmark predictions
    benchmark_test_pred = benchmark.predict(X_test_np)
    benchmark_test_proba = benchmark.predict_proba(X_test_np)

    # Compute metrics
    results = {
        "teacher": evaluate(task, y_test, teacher_test_pred, teacher_test_probs),
        "student": evaluate(task, y_test, student_test_pred, student_test_proba),
        "benchmark": evaluate(task, y_test, benchmark_test_pred, benchmark_test_proba),
    }

    # Print test results
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} [TEST]:")
        for k, v in metrics.items():
            if k != "classification_report":
                print(f"  {k}: {v}")

    # 11) Evaluate on unseen data if available
    if X_unseen_np is not None and y_unseen is not None:
        print("\n--- Unseen Data Evaluation ---")

        # Generate teacher outputs for unseen data
        logits_unseen = predict_logits(teacher, X_unseen_np)
        if distil_with == "probs":
            teacher_unseen_signal = logits_to_calibrated_probs(teacher, logits_unseen)
        else:
            teacher_unseen_signal = logits_unseen

        # Teacher predictions
        teacher_unseen_probs = logits_to_calibrated_probs(teacher, logits_unseen)
        teacher_unseen_pred = np.argmax(teacher_unseen_probs, axis=1)

        # Student predictions
        student_unseen_pred = student.predict(
            X_unseen_np,
            teacher_logits=(teacher_unseen_signal if distil_with == "logits" else None),
            teacher_probs=(teacher_unseen_signal if distil_with == "probs" else None),
        )
        student_unseen_proba = student.predict_proba(
            X_unseen_np,
            teacher_logits=(teacher_unseen_signal if distil_with == "logits" else None),
            teacher_probs=(teacher_unseen_signal if distil_with == "probs" else None),
        )

        # Benchmark predictions
        benchmark_unseen_pred = benchmark.predict(X_unseen_np)
        benchmark_unseen_proba = benchmark.predict_proba(X_unseen_np)

        # Compute unseen metrics
        unseen_results = {
            "teacher": evaluate(
                task, y_unseen, teacher_unseen_pred, teacher_unseen_probs
            ),
            "student": evaluate(
                task, y_unseen, student_unseen_pred, student_unseen_proba
            ),
            "benchmark": evaluate(
                task, y_unseen, benchmark_unseen_pred, benchmark_unseen_proba
            ),
        }

        results["unseen"] = unseen_results

        # Print unseen results
        for model_name, metrics in unseen_results.items():
            print(f"\n{model_name.upper()} [UNSEEN]:")
            for k, v in metrics.items():
                if k != "classification_report":
                    print(f"  {k}: {v}")

    # 12) Save models
    print("\nSaving models...")
    outdir = settings.model_save_path
    outdir.mkdir(parents=True, exist_ok=True)

    dump(preproc, outdir / f"preproc_{task}.joblib")
    dump(student, outdir / f"student_ensemble_{task}.joblib")
    dump(benchmark, outdir / f"benchmark_{task}.joblib")
    dump(teacher, outdir / f"teacher_{task}.pt")

    print(f"Models saved to: {outdir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge distillation training pipeline"
    )
    parser.add_argument(
        "--task", choices=["binary", "multiclass", "both"], default="both"
    )
    parser.add_argument("--dataset", default="ciciot2023", help="Training dataset name")
    parser.add_argument(
        "--unseen", default="cicdiad2024", help="Unseen evaluation dataset"
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Use StandardScaler instead of RobustScaler",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument(
        "--members",
        type=int,
        default=3,
        help="Number of LightGBM models in student ensemble",
    )
    parser.add_argument(
        "--distil",
        choices=["logits", "probs"],
        default="probs",
        help="Use teacher logits or calibrated probabilities",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(
            task=task,
            dataset_name=args.dataset,
            unseen_dataset=args.unseen,
            robust_scaler=not args.standard,
            teacher_epochs=args.epochs,
            teacher_batch_size=args.batch,
            n_student_members=args.members,
            distil_with=args.distil,
            seed=args.seed,
        )
