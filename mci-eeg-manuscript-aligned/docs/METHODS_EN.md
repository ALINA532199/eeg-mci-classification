# Methods text for the configured workflow

Use the following text after checking the acquisition records and executing the
corresponding analysis. Acquisition hardware, participant characteristics and
results are not inferred by this software.

## EEG preprocessing and features

Continuous EEG was rereferenced to the average of M1 and M2 using the complete
available scalp montage. Signals were filtered at 0.1–30 Hz with a 48–52 Hz notch.
Manually identified bad segments were annotated, and designated bad channels were
interpolated using the electrode geometry. ICA was fitted to a 1–30 Hz copy of
the scalp recordings. Component maps, time series and spectra were reviewed blind
to diagnosis to identify ocular, saccadic and cardiac artifacts; EOG/ECG scores
were used as supporting information when these channels were available. The
saved, reviewed decomposition was applied to the analysis recordings. Epochs
covered −500 to 1,000 ms relative to probe onset and were baseline corrected over
−500 to 0 ms. Epochs exceeding an absolute amplitude of 80 μV were rejected.

Handcrafted features used the frontal (F3/Fz/F4), frontocentral (FC3/FCz/FC4),
parietal (P3/Pz/P4) and occipital (PO7/Oz/PO8) ROIs. ERP measures included mean
amplitude and peak latency in the predefined N1, P2, N2 and P3 windows. Morlet
power was computed per trial with n_cycles=max(3,f/2), averaged across trials,
and converted to dB relative to the mean baseline power at each electrode and
frequency. This estimates total power, including phase-locked and non-phase-locked
activity. Reflection padding was used for convolution; it does not eliminate
uncertainty near epoch boundaries, especially at low frequencies.

PAC phase and amplitude were obtained by band-pass filtering and Hilbert
transformation over each full epoch, followed by extraction of the target window.
Trial-wise MI used 18 phase bins, with zero mean amplitude assigned to unoccupied
bins. The subject statistic was the mean trial MI. Each of 200 phase-shift
surrogates reproduced this same mean-trial statistic; zMI used the surrogate mean
and sample standard deviation. Short windows can yield sparse phase coverage and
biased MI; these values require sensitivity assessment and cannot independently
establish physiological coupling. Full-epoch filtering does not resolve the
limited number of low-frequency cycles in a short window.

Connectivity used the debiased squared weighted phase-lag index, calculated for
electrode pairs and averaged over each ROI pair. The primary implementation pools
imaginary cross-products over trials and window samples. Adjacent samples are
correlated, so this does not provide that many independent observations; the
finite-sample correction may remain imperfect. A trial-mean cross-product option
is available for a separately reported sensitivity analysis. Finite negative
estimates are retained. Statistical comparisons use subjects, not individual
time samples, as observations.

## Classification and permutation testing

All pipelines shared repeated stratified five-fold subject-level CV with ten
repetitions. Each subject's epochs remained within one partition. EEGNet used
only the 12 ROI electrodes from 0 to 1,000 ms, independently of the scalp montage
used for referencing and ICA. Outer-training subjects were split into network
training and validation subsets for early stopping; channel scaling was fitted
on the network-training subset. Trial probabilities and embeddings were averaged
within subjects. Embedding scaling and eight-component PCA were fitted on the
outer training set and applied to the test set.

Handcrafted EEG selection used two-sided mean-difference permutation tests on
outer-training subjects, followed by BH correction within declared feature
families. EEG candidates were ranked by absolute pooled-SD Cohen's d, subject to
prespecified domain caps, and pruned for absolute Pearson correlations exceeding
0.70. ERP constitutes one family; TF and dwPLI families are band × window, and PAC
families are coupling type × window. The cap values must be reported explicitly.
The current P1 configuration screens behavior using Welch tests and BH; P3 and
P6 retain ACC, RT and IES without behavioral screening and share the same selected
handcrafted features.

P3/P6 used logistic regression, linear SVC, RBF SVC, random forest and gradient
boosting base learners. Five-fold inner CV generated base probabilities conditional
on the representations fitted in the outer-training set. These probabilities were
concatenated with each model's processed input features and standardized before
an L2 logistic meta-classifier (C=1). The outer test set was excluded from all
fitting. Inner scores are not independent estimates of the full feature-learning
pipeline; performance is assessed only using outer OOF predictions.

The ten OOF probabilities per subject were averaged before calculating AUC.
Full-pipeline label permutation repeated selection, preprocessing fits, EEGNet,
PCA and model fitting, with stratified partitions regenerated using permuted
subject labels and the same seed rule. After repeatability checks and three pilot
permutations, 1,000 formal permutations used p=(1+b)/(1+1000), where b counts null
AUCs at least as large as the observed AUC. Empty selected inputs use training
prevalence; training errors stop and are logged. Pilot permutations are separate.

## Items to finalize before using this text

Specify the EEG domain caps, final model hyperparameters, P1 test choice, network
validation fraction/patience, event codes and actual sampling rate. The supplied
configuration records defaults rather than evidence that those values were used
in an already completed study. The PAC surrogate aggregation and retention of
negative dwPLI2 estimates should be stated explicitly. See the correspondence
table for remaining manuscript details.
