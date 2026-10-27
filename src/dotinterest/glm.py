"""Single-trial GLM in parcel space.

Each trial is cut from its block with padding and fitted on its own: Gaussian-kernel HRF
basis for that trial + polynomial drift, ordinary least squares. The per-trial feature is the
area under the fitted response, AUC = c . beta with c_i = integral of kernel i.

The OLS standard errors ignore residual autocorrelation and are too small; they are kept for
diagnostics only. Inference happens at the group level with permutation tests.
"""
from __future__ import annotations

import cedalion.models.glm as glm
import numpy as np
import pandas as pd
import xarray as xr
from cedalion import units

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def make_basis(p: dict):
    return glm.basis_functions.GaussianKernelsWithTails(
        t_pre=p["t_pre"] * units.s, t_post=p["t_post"] * units.s,
        t_delta=p["t_delta"] * units.s, t_std=p["t_std"] * units.s)


def trial_window(row, p: dict) -> tuple[float, float]:
    if row.kind == "video":
        return row.onset - p["pad_video_s"], row.onset + row.duration + p["pad_video_s"]
    return row.onset - p["pad_rest_s"], row.onset + row.duration + p["pad_rest_s"]


def check_trial(row, y, p: dict) -> str:
    """'' if the trial can be fitted, otherwise the reason it is skipped."""
    if row.kind == "video" and not (p["video_dur_range_s"][0] <= row.duration <= p["video_dur_range_s"][1]):
        return f"implausible clip duration {row.duration:.1f} s (aborted/restarted?)"
    if y.sizes["time"] < 2:
        return "window outside its block"
    span = float(y.time.values[-1] - y.time.values[0])
    if span < p["t_pre"] + p["t_post"]:
        return f"window ({span:.1f} s) shorter than the HRF basis (rank-deficient design)"
    return ""


def fit_trial(y: xr.DataArray, row, basis, drift_order: int, noise_model: str = "ols", n_jobs: int = 1):
    """-> (auc, se, t) as (parcel, chromo) DataArrays."""
    stim = pd.DataFrame({"onset": [row.onset], "duration": [row.duration], "value": [1.0],
                         "trial_type": [row.trial]})
    dms = (glm.design_matrix.hrf_regressors(y, stim, basis)
           & glm.design_matrix.drift_regressors(y, drift_order=drift_order))
    res = glm.fit(y, dms, noise_model=noise_model, max_jobs=n_jobs)
    t = dms.common.time.values
    regs = [str(r) for r in res.sm.params.regressor.values]
    c = np.array([_trapz(dms.common.sel(regressor=r, chromo="HbO").values, t)
                  if r.startswith("HRF") else 0.0 for r in regs])
    tt = res.sm.t_test([c])
    eff = xr.apply_ufunc(lambda r: float(np.ravel(r.effect)[0]), tt, vectorize=True)
    se = xr.apply_ufunc(lambda r: float(np.ravel(r.sd)[0]), tt, vectorize=True)
    return eff, se, eff / se
