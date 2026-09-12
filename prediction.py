"""Predict left- versus right-fist trials from an EEGMMIDB EDF recording.

Usage (Inference on a single raw EDF):
    python prediction.py path/to/S088R11.edf

Diagnostic (Systematically evaluate held-out 20% cohort):
    python prediction.py --evaluate-held-out

To build the shipped model locally:
    python prediction.py --train

The model is trained strictly on Subjects 1--87 (Runs 3 and 7 - Motor Execution).
Subjects 88--109 and test Run 11 are permanently held out and can be used to test it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import mne
import numpy as np
import pandas as pd
from scipy import stats
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# 1. Dataset Split & Parameters (Pure Motor Execution)
TRAINING_SUBJECTS: tuple[int, ...] = tuple(range(1, 88))
HELD_OUT_SUBJECTS: tuple[int, ...] = tuple(range(88, 110))

TRAINING_RUNS: tuple[int, ...] = (3, 7)  # Execution: Left/Right Fist
TEST_RUNS: tuple[int, ...] = (11,)       # Execution: Held-out Run
TARGET_SFREQ = 160.0
L_FREQ = 8.0
H_FREQ = 30.0
TMIN = 0.5
TMAX = 3.5
N_TIMES = 481  # 3.0 seconds at 160 Hz inclusive
LABELS = {0: "Left_Fist", 1: "Right_Fist"}
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "riemannian_lr_execution_s001_s087.joblib"


def _prepare_raw(edf_path: str | Path, expected_channels: Sequence[str] | None = None) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        on_missing="raise",
        verbose=False,
    )

    if expected_channels is not None:
        missing = sorted(set(expected_channels).difference(raw.ch_names))
        if missing:
            raise ValueError(f"EDF missing expected channels: {', '.join(missing[:8])}")
        raw.reorder_channels(list(expected_channels))

    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose=False)

    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge", verbose=False)
    return raw


def _epochs_from_raw(raw: mne.io.BaseRaw) -> tuple[mne.Epochs, np.ndarray]:
    events, event_id = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose=False)
    if len(events) == 0 or not event_id:
        raise ValueError("No T1/T2 cue annotations found.")

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=TMIN, tmax=TMAX, baseline=None,
        preload=True, verbose=False,
    )
    X = epochs.get_data(copy=False)
    if X.shape[-1] < N_TIMES:
        raise ValueError(f"Expected at least {N_TIMES} timepoints per epoch.")
    return epochs, X[:, :, :N_TIMES]


def _load_subject_epochs(
    subject_id: int, runs: Sequence[int], expected_channels: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = mne.datasets.eegbci.load_data(
        subject=subject_id, runs=list(runs), update_path=False, verbose=False
    )
    raws = []
    for path in paths:
        raw_part = mne.io.read_raw_edf(path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw_part)
        raw_part.set_montage(
            mne.channels.make_standard_montage("standard_1005"),
            match_case=False, on_missing="raise", verbose=False,
        )
        if expected_channels is not None:
            raw_part.reorder_channels(list(expected_channels))
        raws.append(raw_part)

    raw = mne.concatenate_raws(raws, verbose=False)
    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose=False)

    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge", verbose=False)
    epochs, X = _epochs_from_raw(raw)
    y = epochs.events[:, -1] - 1
    return X, y, list(raw.ch_names)


def train_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    mne.set_log_level("WARNING")
    X_by_subject: list[np.ndarray] = []
    y_by_subject: list[np.ndarray] = []
    channel_names: list[str] | None = None

    print(f"Training Riemannian Geometry Pipeline on Subjects {TRAINING_SUBJECTS[0]}--{TRAINING_SUBJECTS[-1]}...")
    for subject_id in TRAINING_SUBJECTS:
        X, y, observed_channels = _load_subject_epochs(subject_id, TRAINING_RUNS, channel_names)
        if channel_names is None:
            channel_names = observed_channels
        X_by_subject.append(X)
        y_by_subject.append(y)

    pipeline = Pipeline([
        ("covariances", Covariances(estimator="lwf")),
        ("tangent_space", TangentSpace(metric="riemann")),
        ("logistic_regression", LogisticRegression(solver="lbfgs", max_iter=1000)),
    ])
    pipeline.fit(np.concatenate(X_by_subject), np.concatenate(y_by_subject))

    artifact: dict[str, Any] = {
        "pipeline": pipeline,
        "channel_names": channel_names,
        "training_subjects": list(TRAINING_SUBJECTS),
        "held_out_subjects": list(HELD_OUT_SUBJECTS),
        "training_runs": list(TRAINING_RUNS),
        "label_map": LABELS,
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def predict_edf(edf_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> list[dict[str, Any]]:
    artifact = joblib.load(model_path)
    raw = _prepare_raw(edf_path, artifact["channel_names"])
    epochs, X = _epochs_from_raw(raw)
    pipeline: Pipeline = artifact["pipeline"]
    
    predicted_classes = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    class_to_column = {int(label): index for index, label in enumerate(pipeline.classes_)}

    predictions: list[dict[str, Any]] = []
    for event, predicted_class, probability in zip(epochs.events, predicted_classes, probabilities):
        class_id = int(predicted_class)
        predictions.append({
            "cue_onset_seconds": round(float(event[0] / raw.info["sfreq"]), 6),
            "predicted_label": LABELS[class_id],
            "probability_left_fist": float(probability[class_to_column[0]]),
            "probability_right_fist": float(probability[class_to_column[1]]),
        })
    return predictions


def evaluate_held_out_cohort(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Systematically test the loaded model on the 20% held-out subjects (Run 11)."""
    artifact = joblib.load(model_path)
    pipeline: Pipeline = artifact["pipeline"]
    channel_names = artifact["channel_names"]
    
    records = []
    print(f"\nEvaluating on Held-Out Cohort: Subjects {HELD_OUT_SUBJECTS[0]}--{HELD_OUT_SUBJECTS[-1]}")
    for sub in HELD_OUT_SUBJECTS:
        try:
            X_test, y_test, _ = _load_subject_epochs(sub, TEST_RUNS, channel_names)
            acc = pipeline.score(X_test, y_test)
            records.append({"Subject": f"S{sub:03d}", "Accuracy": acc})
            print(f"Subject {sub:03d} | Run {TEST_RUNS[0]} Accuracy: {acc:.2%}")
        except Exception as e:
            print(f"Subject {sub:03d} failed: {e}")
            
    df = pd.DataFrame(records)
    n = len(df)
    mean_acc = df["Accuracy"].mean()
    sem_acc = df["Accuracy"].std(ddof=1) / np.sqrt(n)
    t_stat, p_val = stats.ttest_1samp(df["Accuracy"], popmean=0.50)
    
    print("\n" + "="*50)
    print(f"Held-Out Cohort Performance (N={n})")
    print(f"Model: {model_path.name}")
    print(f"Mean Accuracy: {mean_acc:.2%} ± {sem_acc:.2%} SEM")
    print(f"Significance vs 50% Chance: p = {p_val/2:.4e} (1-tailed)")
    print("="*50)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict motor execution from an EEGMMIDB EDF file.")
    parser.add_argument("edf_path", nargs="?", help="Path to raw EEGMMIDB EDF file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to joblib artifact")
    parser.add_argument("--train", action="store_true", help="Fit model on Subjects 1--87 (Runs 3, 7)")
    parser.add_argument("--evaluate-held-out", action="store_true", help="Systematically benchmark the held-out 20% cohort")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.train:
        artifact = train_model(args.model)
        print(f"Model saved to {args.model}.")
        return
    if args.evaluate_held_out:
        if not args.model.exists():
            raise SystemExit(f"Artifact not found. Run `python prediction.py --train` first.")
        evaluate_held_out_cohort(args.model)
        return
    if not args.edf_path:
        raise SystemExit("Provide an EDF path, --train, or --evaluate-held-out.")
    if not args.model.exists():
        raise SystemExit("Artifact not found. Run `python prediction.py --train` first.")
    print(json.dumps(predict_edf(args.edf_path, args.model), indent=2))


if __name__ == "__main__":
    main()