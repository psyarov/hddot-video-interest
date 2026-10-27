"""Interest ratings from PsychoPy output: parsing, clip identity, normalisation.

Clip identity is derived from the presentation slot, never from a running count of rated
trials: video_global = (block - 1) * videos_per_block + position (1 .. n per session), and
video_uid = (session - 1) * n_per_session + video_global (shared across participants).
Subject and session come from the file location, not from the PsychoPy `session` field.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .paradigm import video_uid


def _find(df, exact, pattern):
    for c in exact:
        if c in df.columns:
            return c
    hits = [c for c in df.columns if re.search(pattern, str(c), re.I)]
    return hits[0] if hits else None


def parse_psychopy(csv_path: str | Path, sub: str, ses: str | int, p: dict) -> tuple[pd.DataFrame, dict]:
    """One PsychoPy CSV -> one row per rated clip, plus a QC dict.

    Block membership is read from the explicit outer-loop counter when present; otherwise
    from resets of the within-block counter over all PRESENTED trials (dropping unrated
    rows first would shift every block after a fully unrated one).
    """
    raw = pd.read_csv(csv_path, low_memory=False)
    c_rating = _find(raw, [p["rating_column"]], r"slider.*response$")
    c_pos = _find(raw, [p["position_column"]], r"video.*thisTrialN$")
    c_block = _find(raw, p["block_columns"], r"^$")
    if c_rating is None or c_pos is None:
        raise KeyError(f"{csv_path}: rating or trial-counter column not found")

    trials = raw[raw[c_pos].notna()].reset_index(drop=True)
    pos0 = pd.to_numeric(trials[c_pos]).astype(int).to_numpy()
    if c_block is not None:
        block = pd.to_numeric(trials[c_block]).astype(int).to_numpy() + 1
    else:
        block = np.cumsum(np.r_[1, (np.diff(pos0) <= 0).astype(int)])
    vpb, n_ses = p["videos_per_block"], p["videos_per_block"] * p["n_video_blocks"]
    vg = (block - 1) * vpb + pos0 + 1
    rating = pd.to_numeric(trials[c_rating], errors="coerce")

    df = pd.DataFrame({"session_tag": f"sub-{sub}_ses-{ses}", "subject": str(sub),
                       "session": int(ses), "block": block, "position_in_block": pos0 + 1,
                       "video_global": vg,
                       "video_uid": [video_uid(ses, v, n_ses) for v in vg],
                       "rating": rating.values})
    qc = dict(session_tag=f"sub-{sub}_ses-{ses}", n_presented=len(df),
              n_rated=int(rating.notna().sum()), n_expected=n_ses,
              duplicate_slots=int(df.video_global.duplicated().sum()),
              block_source=c_block or "counter resets")
    return df.dropna(subset=["rating"]).reset_index(drop=True), qc


def add_normalisations(df: pd.DataFrame) -> pd.DataFrame:
    """rating_z_session (primary label: z within subject x session), rating_z_subject,
    within-session percentile and tertile (descriptive)."""
    df = df.copy()
    g_ses = df.groupby(["subject", "session"])["rating"]
    df["rating_z_session"] = g_ses.transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    df["rating_z_subject"] = df.groupby("subject")["rating"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1))
    df["rating_pct_session"] = g_ses.transform(lambda x: x.rank(pct=True))
    df["interest_tertile"] = g_ses.transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "middle", "high"]).astype(str))
    return df


def normative_rating(df: pd.DataFrame, col: str = "rating_z_session", min_raters: int = 3) -> pd.Series:
    """Leave-one-subject-out mean rating of the same clip by everybody else."""
    out = pd.Series(np.nan, index=df.index)
    for uid, g in df.groupby("video_uid"):
        for i, r in g.iterrows():
            others = g.loc[g.subject != r.subject, col].dropna()
            if len(others) >= min_raters:
                out[i] = others.mean()
    return out


def agreement_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per subject: Spearman rho between own rating and the leave-one-out normative rating."""
    rows = []
    for sub, g in df.dropna(subset=["rating_normative"]).groupby("subject"):
        if len(g) >= 10:
            rows.append(dict(subject=sub, n=len(g),
                             rho=g.rating_z_session.corr(g.rating_normative, method="spearman")))
    return pd.DataFrame(rows)
