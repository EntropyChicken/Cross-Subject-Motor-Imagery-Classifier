# EEG Motor Imagery Classification

Cross-subject EEG motor imagery classification using the [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) dataset.

This project evaluates whether a classifier trained on EEG recordings from one group of subjects can generalize to **completely unseen subjects**. The current implementation uses a Riemannian geometry pipeline with logistic regression to classify imagined left- versus right-fist movement.

## Overview

The model takes an EEGMMIDB EDF recording and predicts, for each motor-imagery trial, whether the subject was imagining:

- **Left fist**
- **Right fist**

The primary evaluation is performed across subjects rather than randomly splitting individual trials. This is important because randomly splitting trials from the same subject can produce overly optimistic results when the goal is to generalize to a new person.

### Dataset split

Subjects are divided into a development/training cohort and a permanently held-out cohort:

| Subjects | Runs 4, 8, 12 |
|---|---|
| **S001–S087** | Training |
| **S088–S109** | Held-out evaluation |

Runs **4, 8, and 12** are motor-imagery runs containing left- and right-fist imagery.

The model therefore has access to all three motor-imagery runs from the training subjects, while **no data from Subjects 88–109 is used during training**.

The held-out evaluation uses Runs 4, 8, and 12 from Subjects 88–109.

## Pipeline

The current classifier is:

```text
Raw EDF
   ↓
EEG channel standardization
   ↓
Standard 10-05 montage
   ↓
Resampling to 160 Hz
   ↓
8–30 Hz band-pass filter
   ↓
Epoch 0.5–3.5 s after cue
   ↓
Covariance estimation (Ledoit-Wolf)
   ↓
Riemannian tangent-space mapping
   ↓
Logistic Regression
   ↓
Left / Right fist prediction
```

### Preprocessing

- Sampling frequency: **160 Hz**
- Band-pass filter: **8–30 Hz**
- Epoch: **0.5–3.5 seconds after the cue**
- Montage: `standard_1005`
- Number of time points per epoch: **481**
- Baseline correction: none

### Riemannian geometry

For each EEG epoch, the pipeline estimates a regularized covariance matrix:

```python
Covariances(estimator="lwf")
```

The covariance matrices are then mapped into a Riemannian tangent space:

```python
TangentSpace(metric="riemann")
```

Finally, a logistic regression classifier operates on the tangent-space features:

```python
LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)
```

The complete model is implemented as a scikit-learn `Pipeline`.

## Evaluation

The primary evaluation asks:

> How well does the model classify motor imagery from subjects it has never seen during training?

Subjects **S088–S109** are kept completely separate from training.

Current held-out cohort results:

```text
Held-Out Cohort Performance (N=22)

Mean Accuracy: 57.97%
SEM:            2.10%
SD:             9.83%
95% CI:         [53.62%, 62.33%]

One-tailed t-test vs. 50% chance:
p = 5.1664e-04
```

The mean is calculated across the 22 held-out subjects.

Importantly, training-set accuracy is substantially higher than held-out-subject accuracy. This indicates that the model learns patterns that fit the training subjects well but do not transfer perfectly to unseen subjects. Cross-subject generalization is therefore the primary challenge investigated by this project.

## Running the project

### Install dependencies

Create the provided environment and activate it:

```bash
conda env create -f environment.yml
conda activate ntab-bci
```

### Train the model

```bash
python prediction.py --train
```

This trains the model on:

```text
Subjects 1–87
Runs 4, 8, 12
```

and saves the trained pipeline as:

```text
models/riemannian_lr_imagery_s001_s087.joblib
```

### Predict an EDF

After training:

```bash
python prediction.py path/to/recording.edf
```

For example:

```bash
python prediction.py path/to/S088R12.edf
```

The program outputs JSON containing predictions for each detected T1/T2 trial.

Example structure:

```json
[
  {
    "cue_onset_seconds": 20.0,
    "predicted_label": "Left_Fist_Imagined",
    "probability_left_fist": 0.73,
    "probability_right_fist": 0.27
  }
]
```

The prediction script does not require ground-truth labels to make predictions.

### Evaluate the held-out cohort

To systematically evaluate Subjects 88–109:

```bash
python prediction.py --evaluate-held-out
```

This loads the trained model and evaluates it on Runs 4, 8, and 12 from each held-out subject.

## Project structure

```text
.
├── prediction.py
├── environment.yml
├── models/
│   └── riemannian_lr_imagery_s001_s087.joblib
└── README.md
```

## Reproducibility and data separation

The held-out subjects are deliberately excluded from the training process.

Training:

```text
S001–S087
├── Run 4
├── Run 8
└── Run 12
```

Final evaluation:

```text
S088–S109
├── Run 4
├── Run 8
└── Run 12
```

The distinction is therefore **subject-based**, not run-based. Run 12 is used for training on Subjects 1–87 because using a particular run from a training subject does not expose the model to that run from an unseen subject.

This setup is intended to measure generalization to new people rather than generalization to an unseen run number.

## Limitations

The current results demonstrate that cross-subject EEG motor imagery classification remains difficult.

In particular:

- Training accuracy is substantially higher than held-out-subject accuracy.
- EEG signals vary considerably between subjects.
- Spatial covariance patterns learned from one group of subjects may not transfer directly to another group.
- A model can therefore achieve high accuracy on data from subjects it has already seen while exhibiting a strong class bias on unseen subjects.

For this reason, training accuracy is not used as the primary measure of model quality. The held-out-subject results are the more important evaluation.

## Future work

Potential improvements include:

- More systematic subject-level cross-validation on the development cohort
- CSP + LDA comparison with the Riemannian pipeline
- Score-level fusion between complementary classifiers
- Stronger regularization
- Riemannian domain adaptation / subject alignment
- Investigation of frequency-band selection
- Analysis of per-subject confusion matrices and prediction bias
- More rigorous evaluation of whether additional runs improve cross-subject generalization

## License

See the repository license for details.