# Methods, step by step

Parameter values are the defaults in `config/config.example.yaml` (the values used in the
thesis analysis). File references point to the code that implements each step.

## 0 · Ratings — `scripts/00_ratings.py`, `dotinterest/ratings.py`

- One interest rating per clip (continuous slider). Clip identity from the presentation slot.
- `rating_z_session`: z-score within subject × session — the primary label. It removes the
  day's slider bias and, by construction, any between-session difference in mean interest.
- `rating_z_subject`: z-score within subject (keeps between-session differences).
- `rating_normative`: mean `rating_z_session` of the same clip by all *other* participants
  (≥ 3 raters). Measures how interesting a clip is to people in general.

## 1 · Forward model — `scripts/01_forward_model.py`, `dotinterest/headmodel.py`

- ICBM152 two-surface head model (cedalion) with Schaefer-600 parcels / Yeo-17 labels.
- The montage is aligned on shared landmarks and spring-relaxed onto the scalp (keeps
  source–detector distances).
- Fluence with NIRFASTer (or MCX) → sensitivity matrix. Computed once; all sessions share it.

## 2 · Channel-space preprocessing — `scripts/02_preprocess.py`, `dotinterest/preprocessing.py`

1. Events: markers → fNIRS clock (matched sync pulses, linear fit = offset + drift).
2. Discard data after `end`; repair non-positive/non-finite amplitudes; amplitude → OD.
3. Motion correction: TDDR, then wavelet.
4. High-pass 0.01 Hz (Butterworth, order 4).
5. Channel pruning on rest block 1 minus 120 s at each end, on amplitude rebuilt from the
   high-passed OD: mean amplitude outside [1 mV, 0.84 V] or SNR < 3 → pruned.
6. Measurement variance per block on the high-passed OD; pruned channels inflated ×10⁶.
   This is the noise model of the reconstruction.
7. MBLL (DPF 6), low-pass 0.4 Hz.
8. Blocks with buffers (video ± 60 s, rest ± 30 s).
9. Global component per block: inverse-variance-weighted mean over channels, regressed out
   per channel and chromophore; cleaned data → OD.
10. Parcels whose expected OD change with the retained channels exceeds 0.001 are kept.
11. `alpha_meas` estimated from the noise model (K = 0.01), averaged over video blocks.

## 3 · Image reconstruction — `scripts/03_reconstruct.py`, `dotinterest/recon.py`

- Tikhonov, `mua2conc`, scalp + brain compartments, `alpha_spatial` = 0.001, per-session
  `alpha_meas`, measurement noise model applied.
- Brain vertices → mean over each sensitive parcel.
- Second global component, now in parcel space (inverse-variance weights; Limbic parcels get
  zero weight because only a few are covered), regressed out per block.

**Consequence:** after two global regressions every map is relative to the head-wide average
(zero-sum). "Negative" means "below the average", not deactivation.

## 4 · Single-trial GLM — `scripts/04_glm.py`, `dotinterest/glm.py`

- Window: clip onset − 13 s … clip offset + 13 s (rest: onset − 15 s … onset + 75 s).
- Design: Gaussian kernels with tails (−2 … 55 s, spacing 6 s, width 3 s) + offset and
  linear drift; OLS per parcel and chromophore.
- Feature: AUC = c·β with cᵢ the area under kernel *i*, i.e. the area under the fitted response.
- Skipped trials (listed in `glm_trials.csv`): clip duration outside 45–75 s, window outside
  its block, or window shorter than the basis.
- OLS standard errors ignore residual autocorrelation and are too small; they are carried for
  diagnostics only. All inference is at the group level by permutation.

## 5 · Features — `scripts/05_features.py`, `dotinterest/features.py`

Per trial × parcel × chromophore, after subtracting the mean of −8 … −1 s:

| family | definition |
|---|---|
| `mean`, `auc`, `slope` | over 0–60 s; also `_early` (0–15 s) and `_sustained` (15–55 s) |
| `std`, `stdderiv` | SD of the signal / of its first difference, 0–60 s |
| `ttp` | time of the largest absolute value (negative control) |
| `lowfreq` | Welch power 0.01–0.1 Hz |
| `hbohbr` | HbO–HbR correlation within the parcel |
| `fc` | correlations between network × hemisphere mean time courses |
| `glm_auc` | from step 4 |

Every trial of a session uses the same parcels (cortical, finite in every block). Network
features: Yeo-17 network × hemisphere; magnitudes get `_sum` and `_mean`, `ttp`/`hbohbr` only
`_mean`. No feature selection here — selection belongs inside the CV folds.

## 6 · Group statistics — `scripts/06_group_stats.py`, `dotinterest/stats.py`

- Contrasts: video vs rest (manipulation check), interest continuous (primary), top vs bottom
  tertile (secondary).
- Three levels: within-session effect (Pearson r with the rating, or mean difference) →
  subject mean (Fisher z) → one-sample t across subjects. Units need ≥ 8 subjects.
- Network value = mean of member parcels (≥ 3); parcels sensitive in every session of the
  subject.
- Permutation: labels shuffled within session (5000×) → per-unit p and max-statistic p.
  BH-FDR within each contrast × level × family × chromophore. A unit flagged by both is the
  safe claim.
- Specificity: every magnitude contrast re-run after subtracting the trial's brain-wide mean.
- Also: linear mixed model (feature z-scored within session ~ label + (1 | subject)),
  split-half reliability of the group maps (odd vs even sessions), smallest detectable
  Cohen's d at 80 % power.

## 7 · Decoding — `scripts/07_decoding.py`, `dotinterest/decoding.py`

- Within subject: leave-one-session-out (test clips are never in training), ridge penalty from
  an inner leave-one-session-out loop. Regression scored by Spearman ρ within the held-out
  session (Fisher-z averaged); classification by balanced accuracy.
- Grid: every feature set as-is, as 10 PCs and as its 20 best features (selection inside the
  fold), for both interest targets. Labels shuffled within session and the **whole grid** re-run
  per permutation → max-statistic p for the best cell.
- Positive control: video vs rest with `control_models` (default logistic regression, SVM,
  random forest; unbalanced classes rule out a ridge classifier thresholded at 0).
- Model comparison (stages E, F): every model in `decoding.models` — rating: ridge, elastic net,
  PLS, SVR, random forest; high vs low: logistic regression, ridge classifier, SVM, random
  forest — × feature set, within subject (nested leave-one-session-out) and across subjects
  (`plain`, `clip_blind`). Hyperparameters by inner CV (random forests untuned); sets wider than
  40 features reduced to 10 PCs inside the fold. Descriptive: t-test of subject scores against
  chance, no permutation null.
- Cross subject: leave-one-subject-out per clip group in three conditions (`plain`,
  `clip_blind`, `matched`). Everybody saw the same clips, so `plain` can succeed by
  recognising clips; `clip_blind` cannot; `matched` controls for the smaller training set.
  The confirmatory test is `clip_blind`, min-p corrected over feature sets × targets.

## Video features — `scripts/video/`, `dotinterest/video_features.py`

- Extraction per clip: luminance, RMS contrast, optical flow (cut frames interpolated), cuts,
  faces (YuNet), bodies (YOLOv8n), speech probability (Silero VAD), loudness, spectral flux.
- Diagnostics as GLM regressors: resampled to the DOT rate, fades (first/last 1.5 s) set to the
  clip mean, block timelines, HRF convolution (canonical; gamma τ = 3 s as sensitivity),
  nuisance = clip boxcar, onsets, offsets, rating screens, instructions, drift. Reported:
  correlations, VIF, variance left after nuisance, flat clips, PCA of low-level features.

## Not included

- Interactive repair of interrupted sessions (use `events_override.csv`).
- Thesis QC figures, 3-D brain renderings and exploratory notebooks.
- Inter-subject correlation and the feature-gain GLM (follow-up work, not part of the thesis).
