# Validation record

Version 0.2.0, local Windows tests, 2026-09-28. Thirteen focused software tests
pass using synthetic signals only. The tests cover:

- M1/M2 subtraction checked numerically against the recorded mastoid average.
- Complete ICA review export, saved-decomposition application and 13-channel
  scalp ICA with an independently selected 12-channel EEGNet input.
- Changed preprocessing invalidates the saved review binding.
- MNE rng interface compatibility (mocked new signature; actual runtime MNE 1.11).
- Permutation feature screening, family BH, Cohen's d and negative dwPLI2 values.
- Short-window feature computation with explicit estimator warnings.
- Shared P3/P6 EEG selection and retention of behavioral inputs.
- Held-out labels do not affect the fitted outer-fold predictions.
- Subject-level EEGNet training/validation partitions are disjoint.
- Full synthetic real-label repeats, pilots, formal permutations and resume.
- Failures are logged without completed results or imputed AUCs.
- Separate post-hoc Stacking SHAP tables and figures.

Runtime: numpy 2.3.4, scipy 1.16.2, pandas 2.3.3, scikit-learn 1.7.2,
torch 2.13.0+cu130 (CPU), mne 1.11.0, shap 0.50.0.

This is software validation, not validation of short-window coupling physiology.
No real-cohort production analysis or 1,000-permutation run has been performed
with this version. MNE 1.13 end-to-end execution and GPU equivalence have not been
tested. GitHub Actions has not yet been run for this package.
