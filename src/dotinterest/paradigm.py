"""Paradigm structure: block limits and the trial table, derived from the event markers.

Expected markers (configurable in `paradigm:`):
    video_<i>                     clip i of the session, i = 1 .. n_videos
    post_rating_baseline_<i>      fixation after the rating of clip i (ends a video block)
    slider_<i>                    rating screen of clip i (fallback block end)
    rest_<b>                      rest block b
    end                           end of the session (data after it are discarded)
Clip i belongs to video block (i - 1) // videos_per_block + 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row(ev: pd.DataFrame, trial_type: str):
    r = ev.loc[ev.trial_type == trial_type]
    return None if r.empty else r.iloc[0]


def block_limits(ev: pd.DataFrame, p: dict) -> dict[str, tuple[float, float]]:
    """{'video_1': (t0, t1), ..., 'rest_1': (t0, t1), ...} in seconds on the fNIRS clock.

    A video block runs from the onset of its first clip to the end of the baseline after
    its last clip; a rest block is the rest marker itself.
    """
    lims = {}
    vpb = p["videos_per_block"]
    for b in range(1, p["n_video_blocks"] + 1):
        first, last = (b - 1) * vpb + 1, b * vpb
        r0 = _row(ev, f"video_{first}")
        r1 = _row(ev, f"post_rating_baseline_{last}")
        if r1 is None:
            r1 = _row(ev, f"slider_{last}")
        if r0 is not None and r1 is not None:
            lims[f"video_{b}"] = (float(r0.onset), float(r1.onset + r1.duration))
    for b in range(1, p["n_rest_blocks"] + 1):
        r = _row(ev, f"rest_{b}")
        if r is not None:
            lims[f"rest_{b}"] = (float(r.onset), float(r.onset + r.duration))
    return lims


def end_of_session(ev: pd.DataFrame, t_last: float) -> float:
    """Onset of the last 'end' marker; for aborted sessions the last event offset."""
    end = ev.loc[ev.trial_type == "end", "onset"]
    if len(end):
        return float(end.iloc[-1])
    return min(float((ev.onset + ev.duration.fillna(0)).max()), t_last)


def trial_table(ev: pd.DataFrame, p: dict) -> pd.DataFrame:
    """One row per video/rest trial: trial, kind, block, onset, duration, video_global.

    The first marker of each trial type is used (a restarted clip keeps its first onset).
    Rest trials get the fixed analysis length `rest_trial_s`.
    """
    rows = []
    vpb = p["videos_per_block"]
    for i in range(1, p["n_video_blocks"] * vpb + 1):
        r = _row(ev, f"video_{i}")
        if r is not None:
            rows.append(dict(trial=f"video_{i}", kind="video", block=(i - 1) // vpb + 1,
                             onset=float(r.onset), duration=float(r.duration), video_global=i))
    for b in range(1, p["n_rest_blocks"] + 1):
        r = _row(ev, f"rest_{b}")
        if r is not None:
            rows.append(dict(trial=f"rest_{b}", kind="rest", block=b, onset=float(r.onset),
                             duration=float(p["rest_trial_s"]), video_global=np.nan))
    return pd.DataFrame(rows)


def block_key(kind: str, block: int) -> str:
    return f"{kind}_{block}"


def video_uid(session: int, video_global: int, n_videos_per_session: int) -> int:
    """Global clip id: every participant saw the same clips in the same order, so session and
    position within the session identify the clip."""
    return (int(session) - 1) * n_videos_per_session + int(video_global)
