"""Single-trial features from parcel time series, and their long / wide tables.

Model-free battery per trial x parcel x chromophore (after subtracting a pre-stimulus baseline):
    mean, auc, slope           full window and, with suffixes _early / _sustained, sub-windows
    std, stdderiv, ttp         variability, derivative variability, time to |peak|
    lowfreq                    band power in `lowband`
    hbohbr                     within-parcel HbO-HbR correlation (one value per parcel)
    fc                         correlation between network-mean time courses (per chromophore)

No feature selection happens here: selection has to live inside the decoding CV folds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch

from .parcels import is_cortical, network_hemi

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# magnitudes may be summed over parcels; times and correlations may only be averaged
SUMMABLE = {"mean", "auc", "slope", "std", "stdderiv", "lowfreq"}


def block_array(da: xr.DataArray):
    """(parcel, chromo, time) DataArray -> numpy array, time vector, parcel names, chromo names."""
    da = da.transpose("parcel", "chromo", "time")
    t = np.asarray(da.time.values, float)
    if not np.all(np.diff(t) > 0):
        raise ValueError("time axis is not strictly increasing")
    return (np.asarray(da.values, float), t,
            np.array([str(x) for x in da.parcel.values]), np.array([str(x) for x in da.chromo.values]))


def trial_features(seg: np.ndarray, t_rel: np.ndarray, fs: float, p: dict) -> dict:
    """seg (parcel, chromo, time), baselined; t_rel in s from onset. -> {family: array}."""
    out = {}
    for wname, (w0, w1) in p["windows"].items():
        sfx = "" if wname == "full" else f"_{wname}"
        m = (t_rel >= w0) & (t_rel <= w1)
        if m.sum() < 5:
            for f in ("mean", "auc", "slope"):
                out[f + sfx] = np.full(seg.shape[:2], np.nan)
            continue
        s, tt = seg[:, :, m], t_rel[m]
        dead = ~np.isfinite(s).any(2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["mean" + sfx] = np.nanmean(s, axis=2)
        out["auc" + sfx] = np.where(dead, np.nan, _trapz(np.nan_to_num(s), tt, axis=2))
        tc = tt - tt.mean()
        out["slope" + sfx] = np.where(dead, np.nan, (np.nan_to_num(s) * tc).sum(2) / (tc ** 2).sum())

    w0, w1 = p["windows"]["full"]
    m = (t_rel >= w0) & (t_rel <= w1)
    s, tt = seg[:, :, m], t_rel[m]
    dead = ~np.isfinite(s).any(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["std"] = np.nanstd(s, axis=2)
        out["stdderiv"] = np.nanstd(np.diff(s, axis=2), axis=2)
    out["ttp"] = np.where(dead, np.nan, tt[np.argmax(np.abs(np.nan_to_num(s)), axis=2)])
    nper = int(min(256, s.shape[2]))
    f, pxx = welch(np.nan_to_num(s), fs=float(fs), axis=2, nperseg=nper)
    band = (f >= p["lowband"][0]) & (f <= p["lowband"][1])
    out["lowfreq"] = np.where(dead, np.nan, pxx[..., band].sum(-1))
    a = np.nan_to_num(s[:, 0] - np.nanmean(s[:, 0], 1, keepdims=True))
    b = np.nan_to_num(s[:, 1] - np.nanmean(s[:, 1], 1, keepdims=True))
    out["hbohbr"] = (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-30)
    return out


def trial_fc(seg: np.ndarray, groups: np.ndarray, gnames: np.ndarray) -> np.ndarray:
    """Upper triangle of the correlation matrix of network-mean time courses. -> (chromo, edge)."""
    iu = np.triu_indices(len(gnames), 1)
    out = np.full((seg.shape[1], len(iu[0])), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in range(seg.shape[1]):
            nt = np.array([np.nanmean(seg[groups == g, c, :], axis=0) for g in gnames])
            ok = np.isfinite(nt).all(1) & (np.nanstd(nt, axis=1) > 0)
            C = np.full((len(gnames), len(gnames)), np.nan)
            if ok.sum() > 1:
                C[np.ix_(ok, ok)] = np.corrcoef(nt[ok])
            out[c] = C[iu]
    return out


def model_free_session(blocks: dict, trials: pd.DataFrame, p: dict):
    """Features for every trial of one session.

    blocks: {'video_1': DataArray, ...}; trials: paradigm.trial_table().
    Every trial is described by the same parcels: cortical and finite in every usable block.
    -> (long parcel frame, long network-FC frame, per-trial status frame)
    """
    arr = {k: block_array(v) for k, v in blocks.items()}
    pc0 = next(iter(arr.values()))[2]
    keep = np.array([is_cortical(x) for x in pc0])
    for A, _, pc, _ in arr.values():
        if not np.array_equal(pc, pc0):
            raise ValueError("blocks disagree on the parcel axis")
        keep &= np.isfinite(A).any(axis=(1, 2))
    parcels = pc0[keep]
    chromo = next(iter(arr.values()))[3]
    groups = np.array([network_hemi(x) for x in parcels])
    gnames = np.array(sorted(set(groups)))
    edges = [f"{gnames[i]}~{gnames[j]}" for i, j in zip(*np.triu_indices(len(gnames), 1))]

    b0 = p["baseline"][0] if p["baseline"] else 0.0
    long_rows, fc_rows, status = [], [], []
    for r in trials.itertuples():
        key = f"{r.kind}_{r.block}"
        if key not in arr:
            status.append(dict(trial=r.trial, status="block missing"))
            continue
        A, t, _, _ = arr[key]
        A = A[keep]
        fs = 1.0 / float(np.median(np.diff(t)))
        i0, i1 = np.searchsorted(t, [r.onset + b0, r.onset + p["trial_length_s"]])
        i1 = min(int(i1) + 1, t.size)
        if i1 - i0 < 20:
            status.append(dict(trial=r.trial, status="too few samples"))
            continue
        seg, t_rel = A[:, :, i0:i1].copy(), t[i0:i1] - r.onset
        if p["baseline"]:
            bm = (t_rel >= p["baseline"][0]) & (t_rel <= p["baseline"][1])
            if bm.sum() >= 3:
                seg -= np.nanmean(seg[:, :, bm], axis=2, keepdims=True)
        F = trial_features(seg, t_rel, fs, p)
        frac = (i1 - i0) / ((p["trial_length_s"] - b0) * fs)
        status.append(dict(trial=r.trial, status="ok" if frac > 0.99 else "partial", frac_ok=frac))
        d = {"trial": r.trial, "parcel": np.repeat(parcels, len(chromo)),
             "chromo": np.tile(chromo, len(parcels))}
        for f, v in F.items():
            d[f] = (np.repeat(v, len(chromo)) if v.ndim == 1 else v.ravel())
        df = pd.DataFrame(d)
        df.loc[df.chromo != chromo[0], "hbohbr"] = np.nan   # one value per parcel
        long_rows.append(df)
        fc = trial_fc(seg, groups, gnames)
        fc_rows.append(pd.DataFrame({"trial": r.trial, "chromo": np.repeat(chromo, len(edges)),
                                     "edge": np.tile(edges, len(chromo)), "fc": fc.ravel()}))
    return (pd.concat(long_rows, ignore_index=True), pd.concat(fc_rows, ignore_index=True),
            pd.DataFrame(status))


def network_table(long: pd.DataFrame, value_cols: list[str], keys=("session_tag", "trial", "chromo")):
    """Parcel rows -> network x hemisphere rows: `<f>_sum` (magnitudes) and `<f>_mean`, n_parcels."""
    d = long.assign(network=[network_hemi(x) for x in long.parcel])
    g = d.groupby(list(keys) + ["network"], observed=True)
    out = g[value_cols].mean().add_suffix("_mean")
    summ = [c for c in value_cols if c.split("_")[0] in SUMMABLE or c == "glm_auc"]
    if summ:
        s = g[summ].sum(min_count=1).add_suffix("_sum")
        out = out.join(s)
    out["n_parcels"] = g.size()
    return out.reset_index()


def wide_table(long: pd.DataFrame, value_cols: list[str], unit: str,
               keys=("session_tag", "subject", "session", "trial")) -> pd.DataFrame:
    """Long rows -> one row per trial, columns '<value>|<unit>|<chromo>'."""
    ws = []
    for vc in value_cols:
        w = long.pivot_table(index=list(keys), columns=[unit, "chromo"], values=vc, aggfunc="first")
        w.columns = [f"{vc}|{u}|{c}" for u, c in w.columns]
        ws.append(w.dropna(axis=1, how="all"))
    return pd.concat(ws, axis=1).reset_index()
