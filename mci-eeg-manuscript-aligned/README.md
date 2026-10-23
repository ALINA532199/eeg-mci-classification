# Subject-level EEG classification

[中文说明](README.zh-CN.md)

Python tools for M1/M2-referenced EEG preprocessing, handcrafted features,
P1–P6 classification, full-pipeline permutation testing and post-hoc SHAP.

## Methods

- Reference: average of the recorded M1 and M2 electrodes.
- ICA: complete available scalp montage; label-blind review of ocular, saccadic
  and cardiac components using saved decompositions and review reports.
- EEGNet: independently configured 12 ROI electrodes, 0–1,000 ms.
- Features: ERP amplitude/latency; trial-average Morlet power followed by baseline
  dB normalization; full-epoch Hilbert PAC and electrode-pair dwPLI2 in specified windows.
- EEG screening: training-subject mean-difference permutation tests, family BH,
  absolute Cohen's d ranking and correlation pruning at 0.70.
- P3/P6: retain all three behavioral inputs, share EEG selection, and pass base
  probabilities plus processed original features to the logistic meta-classifier.
- EEGNet early stopping: separate training/validation subjects inside outer train.
- Evaluation: shared 10×5 subject folds; mean repeated OOF probability per subject.
- Permutations: real-label repeatability check, three pilots, 1,000 formal runs;
  refit label-dependent steps and regenerate splits with each permuted label set.

The inner Stacking probabilities condition on outer-training representations.
Only outer OOF predictions evaluate the complete feature-learning pipeline.
Short-window PAC and pooled-time dwPLI2 require the limitations and sensitivity
assessment described in [the methods](docs/METHODS_EN.md).

## Install and test

```console
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Python >=3.11. CPU is the default. See [validation scope](docs/VALIDATION.md).

## Synthetic demonstration

```console
mci-eeg demo --output data/synthetic_demo
mci-eeg validate --data data/synthetic_demo --config configs/smoke.json --output results/demo
mci-eeg formal --data data/synthetic_demo --config configs/smoke.json --output results/demo
mci-eeg explain --data data/synthetic_demo --config configs/smoke.json --output results/demo_explanations
```

The smoke configuration and synthetic signals are for software tests only.

## Prepare study data

Configure actual event codes, scalp/eog/ecg channel types, montage and sampling
rate in a private copy of `configs/preprocessing_example.json`. A null
`scalp_channels` selects all available scalp EEG except M1/M2; it does not force a
particular channel count. `eegnet_channels` is a separate ordered list of 12
channels. A null `sfreq` preserves the recording rate. See the
[data contract](docs/DATA_FORMAT.md) and [ICA instructions](docs/ICA_REVIEW.zh-CN.md).

```console
mci-eeg ica-review --manifest private/manifest.csv --config private/preprocessing.json --output private/ica_review
# Inspect report.html and complete each review.json before proceeding.
mci-eeg prepare --manifest private/manifest.csv --behavior private/behavior.csv --config private/preprocessing.json --ica-directory private/ica_review --output data/study
mci-eeg validate --data data/study --config configs/analysis.json --output results/study_v1
mci-eeg formal --data data/study --config configs/analysis.json --output results/study_v1
```

The reviewed ICA is loaded, not refitted. Input/configuration/ICA hashes and MNE
versions are checked. Data or settings changed after review require a new review.
Continuous FIF/SET are supported; already-epoched SET needs a separate importer.

## Results

Subject OOF probabilities, fold indices/selected features, permutation AUCs,
errors, software versions and hashes are saved. Upper-tail p is
`(1 + count(null_auc >= observed_auc)) / (m + 1)`. Pilots are separate. Empty
selected inputs predict training prevalence; failed fits stop and are logged.
Resume using the same command and unchanged protocol. A stale lock requires
checking that the recorded process is no longer active before removing it.

Compute time must be benchmarked using pilots. Each outer fold retrains EEGNet.
The default unrestricted permutation assumes exchangeable subject labels; use an
appropriate restricted design for matched, familial or site-restricted samples.

## Documentation and release

- [Manuscript correspondence and unresolved settings](docs/MANUSCRIPT_ALIGNMENT.md)
- [English methods](docs/METHODS_EN.md) / [Chinese methods](docs/METHODS_ZH.md)
- [GitHub/Zenodo release](docs/RELEASE.md)

Domain caps are configurable but not numerically specified in the supplied
manuscript. `null` currently means no cap; finalize these values before using a
capped-feature protocol. SHAP uses separate full-data Stacking interpretation
models, not OOF model explanations. Patient data are not included. MIT license.
