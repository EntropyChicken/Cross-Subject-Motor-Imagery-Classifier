"""Predict left- versus right-fist trials from one EEGMMIDB EDF recording.

Usage:
    python prediction.py path/to/S088R11.edf

To build the shipped model locally (requires the EEGMMIDB data available through
MNE):
    python prediction.py --train

The model is deliberately trained only on subjects 1--87. Subjects 88--109
are a permanently held-out set and are never loaded by ``--train``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import mne
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Dataset and preprocessing choices. These match the inter-subject notebook.
TRAINING_SUBJECTS: tuple[int, ...] = tuple(range(1, 88))
HELD_OUT_SUBJECTS: tuple[int, ...] = tuple(range(88, 110))
TRAINING_RUNS: tuple[int, ...] = (4, 8, 12)  # 4 and 8 and 12 are for motor imagery. 3 and 7 and 11 for executed
TARGET_SFREQ = 160.0
L_FREQ = 8.0
H_FREQ = 30.0
TMIN = 0.5
TMAX = 3.5
N_TIMES = 481  # inclusive endpoints at 160 Hz for a 0.5--3.5 s epoch
LABELS = {0: "Left_Fist", 1: "Right_Fist"}
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "riemannian_lr_execution_s001_s087.joblib"


def _prepare_raw(edf_path: str | Path, expected_channels: Sequence[str] | None = None) -> mne.io.BaseRaw:
    """Load, standardize, channel-align, and filter a single EEGMMIDB EDF."""
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
            raise ValueError(
                "EDF does not contain the complete EEGMMIDB channel layout. "
                f"Missing channels include: {', '.join(missing[:8])}"
            )
        # Model coefficients are channel-order dependent, so never silently
        # accept a compatible set in a different order.
        raw.reorder_channels(list(expected_channels))

    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose=False)

    raw.filter(
        L_FREQ,
        H_FREQ,
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=False,
    )
    return raw


def _epochs_from_raw(raw: mne.io.BaseRaw) -> tuple[mne.Epochs, np.ndarray]:
    """Create the exact cue-locked trial windows used by the fitted model."""
    events, event_id = mne.events_from_annotations(
        raw, event_id={"T1": 1, "T2": 2}, verbose=False
    )
    if len(events) == 0 or not event_id:
        raise ValueError(
            "No T1/T2 cue annotations found. This predictor expects an EEGMMIDB "
            "left/right-fist EDF, not unannotated continuous EEG."
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    X = epochs.get_data(copy=False)
    if len(X) == 0:
        raise ValueError("No complete 0.5--3.5 s cue-locked epochs could be created.")
    if X.shape[-1] < N_TIMES:
        raise ValueError(
            f"Expected at least {N_TIMES} samples per epoch after resampling; got {X.shape[-1]}."
        )
    return epochs, X[:, :, :N_TIMES]


def _load_subject_epochs(
    subject_id: int, runs: Sequence[int], expected_channels: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one participant's downloaded EEGMMIDB runs for model training."""
    paths = mne.datasets.eegbci.load_data(
        subject=subject_id, runs=list(runs), update_path=False, verbose=False
    )
    # This mirrors the notebook: concatenate runs before filtering, so filter
    # edge handling is identical in training and in the reported evaluation.
    raws = []
    for path in paths:
        raw_part = mne.io.read_raw_edf(path, preload=True, verbose=False)
        mne.datasets.eegbci.standardize(raw_part)
        raw_part.set_montage(
            mne.channels.make_standard_montage("standard_1005"),
            match_case=False,
            on_missing="raise",
            verbose=False,
        )
        if expected_channels is not None:
            missing = sorted(set(expected_channels).difference(raw_part.ch_names))
            if missing:
                raise ValueError(f"Subject {subject_id} is missing expected EEG channels: {missing[:8]}")
            raw_part.reorder_channels(list(expected_channels))
        raws.append(raw_part)
    raw = mne.concatenate_raws(raws, verbose=False)
    if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
        raw.resample(TARGET_SFREQ, verbose=False)
    raw.filter(
        L_FREQ,
        H_FREQ,
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=False,
    )
    epochs, X = _epochs_from_raw(raw)
    # T1 is left and T2 is right for runs 3, 7, and 11 only.
    y = epochs.events[:, -1] - 1
    return X, y, list(raw.ch_names)


def train_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Fit only on subjects 1--87 and write a self-describing model artifact."""
    mne.set_log_level("WARNING")
    X_by_subject: list[np.ndarray] = []
    y_by_subject: list[np.ndarray] = []
    channel_names: list[str] | None = None

    for subject_id in TRAINING_SUBJECTS:
        X, y, observed_channels = _load_subject_epochs(
            subject_id, TRAINING_RUNS, channel_names
        )
        if channel_names is None:
            channel_names = observed_channels
        X_by_subject.append(X)
        y_by_subject.append(y)

    pipeline = Pipeline(
        [
            ("covariances", Covariances(estimator="lwf")),
            ("tangent_space", TangentSpace(metric="riemann")),
            ("logistic_regression", LogisticRegression(solver="lbfgs", max_iter=1000)),
        ]
    )
    pipeline.fit(np.concatenate(X_by_subject), np.concatenate(y_by_subject))

    artifact: dict[str, Any] = {
        "pipeline": pipeline,
        "channel_names": channel_names,
        "training_subjects": list(TRAINING_SUBJECTS),
        "held_out_subjects": list(HELD_OUT_SUBJECTS),
        "training_runs": list(TRAINING_RUNS),
        "label_map": LABELS,
        "preprocessing": {
            "target_sfreq_hz": TARGET_SFREQ,
            "bandpass_hz": [L_FREQ, H_FREQ],
            "epoch_seconds_after_cue": [TMIN, TMAX],
            "n_times": N_TIMES,
            "montage": "standard_1005",
        },
        "package_versions": {"mne": mne.__version__, "numpy": np.__version__},
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def predict_edf(edf_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> list[dict[str, Any]]:
    """Return a left/right prediction and probabilities for each valid cue epoch.

    The EDF annotations identify trial *times* only. T1/T2 annotations are not
    used as labels for prediction.
    """
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
        predictions.append(
            {
                "cue_onset_seconds": round(float(event[0] / raw.info["sfreq"]), 6),
                "predicted_label": LABELS[class_id],
                "probability_left_fist": float(probability[class_to_column[0]]),
                "probability_right_fist": float(probability[class_to_column[1]]),
            }
        )
    return predictions


# accesses ground truth values, only for development
def compare_with_edf_labels(
    edf_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH
) -> dict[str, Any]:
    """Diagnostic only: compare predictions with the EDF's known T1/T2 labels."""
    artifact = joblib.load(model_path)
    raw = _prepare_raw(edf_path, artifact["channel_names"])
    epochs, X = _epochs_from_raw(raw)
    pipeline: Pipeline = artifact["pipeline"]
    predicted_classes = pipeline.predict(X).astype(int)
    true_classes = (epochs.events[:, -1] - 1).astype(int)

    trials = []
    for event, predicted_class, true_class in zip(epochs.events, predicted_classes, true_classes):
        trials.append(
            {
                "cue_onset_seconds": round(float(event[0] / raw.info["sfreq"]), 6),
                "true_label": LABELS[true_class],
                "predicted_label": LABELS[predicted_class],
                "correct": bool(predicted_class == true_class),
            }
        )
    return {
        "edf_path": str(edf_path),
        "accuracy": float(np.mean(predicted_classes == true_classes)),
        "n_trials": len(trials),
        "true_label_counts": {
            "Left_Fist": int(np.sum(true_classes == 0)),
            "Right_Fist": int(np.sum(true_classes == 1)),
        },
        "predicted_label_counts": {
            "Left_Fist": int(np.sum(predicted_classes == 0)),
            "Right_Fist": int(np.sum(predicted_classes == 1)),
        },
        "trials": trials,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict left/right fist trials from one EEGMMIDB EDF file.",
        epilog="Example: python prediction.py path/to/S088R11.edf",
    )
    parser.add_argument("edf_path", nargs="?", help="Path to a raw EEGMMIDB EDF file")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the joblib model artifact"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train on subjects 1--87 only and save the model artifact",
    )
    parser.add_argument(
        "--compare-labels",
        nargs="+",
        metavar="EDF_PATH",
        help="Diagnostic: compare predictions with known T1/T2 labels for one or more EDFs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.train:
        if args.edf_path:
            raise SystemExit("Provide either --train or an EDF path, not both.")
        artifact = train_model(args.model)
        print(
            f"Saved model trained on {len(artifact['training_subjects'])} subjects to {args.model}. "
            f"Subjects {artifact['held_out_subjects'][0]}--{artifact['held_out_subjects'][-1]} were not used."
        )
        return
    if args.compare_labels:
        if args.edf_path:
            raise SystemExit("Provide either an EDF path or --compare-labels paths, not both.")
        if not args.model.exists():
            raise SystemExit(f"Model artifact not found: {args.model}. Run `python prediction.py --train` first.")
        reports = [compare_with_edf_labels(edf_path, args.model) for edf_path in args.compare_labels]
        print(json.dumps(reports, indent=2))
        return
    if not args.edf_path:
        raise SystemExit("Provide an EDF path, or use --train to create the model artifact.")
    if not args.model.exists():
        raise SystemExit(f"Model artifact not found: {args.model}. Run `python prediction.py --train` first.")
    print(json.dumps(predict_edf(args.edf_path, args.model), indent=2))


if __name__ == "__main__":
    main()
