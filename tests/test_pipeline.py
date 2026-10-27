"""Tests on synthetic data. `pytest -q` (the end-to-end test takes ~2 min; cedalion is needed
only for the GLM step, which is skipped without it)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dotinterest.decoding import (CrossSubject, WithinSubject, cross_subject_sklearn, minp_corrected,
                                  sklearn_models, within_subject_sklearn)
from dotinterest.io import events_from_markers
from dotinterest.paradigm import block_limits, trial_table
from dotinterest.ratings import add_normalisations, parse_psychopy
from dotinterest.stats import bh, group_test

ROOT = Path(__file__).resolve().parents[1]
PARADIGM = dict(n_video_blocks=2, videos_per_block=3, n_rest_blocks=2, rest_trial_s=60)


def test_sync_recovers_drift_and_offset():
    rng = np.random.default_rng(0)
    nirs = np.cumsum(rng.uniform(20, 200, 20))                     # irregular pulse intervals
    lsl = 1.00001 * nirs + 500.0                                   # 10 ppm drift, 500 s offset
    doubled = np.sort(np.r_[nirs, nirs + 0.1])                     # device logs pulses twice
    marker_t = 1.00001 * np.array([100.0, 200.0]) + 500.0
    ev, info = events_from_markers(["a", "b"], marker_t, np.r_[lsl, lsl[-1] + 7.3], doubled)
    assert np.allclose(ev.onset, [100.0, 200.0], atol=1e-3)
    assert abs(info["sync_drift_ppm"] + 10) < 0.5


def test_paradigm_tables():
    rows = [("video_1", 10, 60), ("post_rating_baseline_3", 250, 10), ("video_2", 85, 60), ("video_3", 170, 60),
            ("rest_1", 300, 300), ("video_4", 700, 60), ("video_5", 780, 60), ("video_6", 860, 60),
            ("slider_6", 920, 3), ("rest_2", 1000, 300)]
    ev = pd.DataFrame(rows, columns=["trial_type", "onset", "duration"])
    lims = block_limits(ev, PARADIGM)
    assert lims["video_1"] == (10, 260) and lims["video_2"] == (700, 923) and lims["rest_2"] == (1000, 1300)
    tt = trial_table(ev, PARADIGM)
    assert list(tt.trial) == [f"video_{i}" for i in range(1, 7)] + ["rest_1", "rest_2"]
    assert list(tt.block[:6]) == [1, 1, 1, 2, 2, 2]


def test_ratings_keep_slots_after_unrated_block(tmp_path):
    # block 1 presented but never rated: blocks 2 and 3 must keep their slot numbers
    rows = []
    for b in range(3):
        for pos in range(8):
            rows.append({"main.thisN": b, "video_block.thisTrialN": pos,
                         "circular_slider.response": np.nan if b == 0 else 1 + (pos % 10)})
        rows.append({"main.thisN": np.nan, "video_block.thisTrialN": np.nan, "circular_slider.response": np.nan})
    p = tmp_path / "x.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    cfg = dict(rating_column="circular_slider.response", position_column="video_block.thisTrialN",
               block_columns=["main.thisN"], videos_per_block=8, n_video_blocks=3)
    df, qc = parse_psychopy(p, "001", 2, cfg)
    assert qc["n_presented"] == 24 and qc["n_rated"] == 16
    assert df.video_global.min() == 9 and df.video_uid.min() == 24 + 9
    df = add_normalisations(df)
    assert abs(df.rating_z_session.mean()) < 1e-9


def _toy_study(effect, n_sub=10, n_ses=3, n_trial=24, n_unit=6, seed=0):
    rng = np.random.default_rng(seed)
    rows, X = [], []
    for s in range(n_sub):
        for k in range(n_ses):
            y = rng.standard_normal(n_trial)
            x = rng.standard_normal((n_trial, n_unit))
            x[:, 0] += effect * y
            X.append(x)
            rows += [dict(subject=f"{s:03d}", session=k + 1, session_tag=f"sub-{s:03d}_ses-{k + 1}", y=v) for v in y]
    return pd.DataFrame(rows), np.vstack(X)


def test_group_test_planted_effect_and_null():
    T, X = _toy_study(effect=0.8)
    res, _ = group_test(X, np.arange(X.shape[1]), T, "y", "corr", n_perm=500)
    assert res.p_perm[0] < 0.01 and res.p_maxstat[0] < 0.01
    assert (res.p_maxstat[1:] > 0.05).all()
    # under the null, permutation p-values are roughly uniform
    ps = []
    for i in range(10):
        Tn, Xn = _toy_study(0.0, seed=i)
        ps.append(group_test(Xn, np.arange(6), Tn, "y", "corr", n_perm=200)[0].p_perm)
    ps = np.concatenate(ps)
    assert 0.0 <= np.mean(ps < 0.05) <= 0.15
    rej, q = bh([0.001, 0.02, 0.04, 0.5])
    assert rej.tolist() == [True, True, False, False]


def test_decoders_find_planted_signal():
    T, X = _toy_study(effect=1.0, n_unit=8)
    y = T.y.to_numpy()
    sub, ses = T.subject.to_numpy(), T.session.to_numpy()
    mean, per = WithinSubject(X, sub, ses, np.isfinite(y), "reg").run(y)
    assert mean > 0.4 and np.isfinite(per).all()
    yb = (y > 0).astype(float)
    assert WithinSubject(X, sub, ses, np.isfinite(yb), "clf").run(yb)[0] > 0.65
    assert WithinSubject(X, sub, ses, np.isfinite(yb), "clf", select=3).run(yb)[0] > 0.65
    xs = CrossSubject(X, sub, ses, np.isfinite(y)).run(y, "reg")
    assert set(xs.condition) == {"plain", "clip_blind", "matched"} and xs.score.mean() > 0.4
    for model in ("pls", "rf"):
        assert np.nanmean(within_subject_sklearn(X, sub, ses, y, "reg", model)) > 0.3
    assert np.nanmean(within_subject_sklearn(X, sub, ses, yb, "clf", "svm", pca=50)) > 0.6   # pca capped
    assert np.nanmean(cross_subject_sklearn(X, sub, ses, yb, "clf", "logreg")) > 0.6
    assert set(sklearn_models("reg")) == {"ridge", "enet", "pls", "svr", "rf"}
    p_cell, p_corr = minp_corrected(np.array([0.9, 0.1]), np.random.default_rng(0).normal(0, 0.1, (2, 200)))
    assert p_cell[0] < 0.01 and p_corr[0] < 0.02 and p_corr[1] >= p_cell[1]


@pytest.mark.slow
def test_end_to_end_synthetic(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from synthetic import write_study

    cfg = write_study(tmp_path, n_sub=8)
    run = lambda *a: subprocess.run([sys.executable, *a, "--config", str(cfg)], check=True, cwd=ROOT,
                                    capture_output=True, text=True)
    if importlib.util.find_spec("cedalion"):
        run("scripts/04_glm.py")
    run("scripts/05_features.py")
    run("scripts/06_group_stats.py", "--n-perm", "200", "--no-parcels")
    run("scripts/07_decoding.py", "--n-perm", "10", "--n-perm-cross", "10",
        "--models", "ridge,logreg", "--sets", "B_mean_HbO")
    g = tmp_path / "derivatives" / "group"
    S = pd.read_csv(g / "stats" / "summary_network.csv")
    hit = S[(S.contrast == "interest_cont") & (S.family == "mean") & (S.chromo == "HbO")].iloc[0]
    assert hit.top_unit.startswith("VisCent") and hit.n_sig_fdr >= 1
    pc = pd.read_csv(g / "decoding" / "positive_control.csv")
    assert pc[pc.model == "logreg"]["mean"].min() > 0.9      # (rf needs > 2 rest trials per training set)
