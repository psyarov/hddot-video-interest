"""Group-level inference with the subject as the unit.

Three-level summary statistics: a within-session effect per unit (Pearson r with a
continuous label, or a mean difference for a binary label) -> subject mean (Fisher-z for
correlations) -> one-sample t across subjects. Labels are permuted WITHIN SESSION, which keeps
every session's own mean, variance and label composition and only breaks the trial-label
pairing. Each unit gets a permutation p and a max-statistic (family-wise) p; Benjamini-Hochberg
is applied within a family.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps


def _fisher(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def _weights(y, kind):
    """Vector w such that X_centred.T @ w is the session statistic."""
    if kind == "corr":
        yc = y - y.mean()
        n = np.sqrt((yc ** 2).sum())
        return yc / n if n > 0 else np.zeros_like(yc)
    return np.where(y == 1, 1.0 / max((y == 1).sum(), 1), -1.0 / max((y == 0).sum(), 1))


def group_test(X, units, trials: pd.DataFrame, ycol: str, kind: str, n_perm: int = 5000,
               min_trials: int = 8, min_subjects: int = 8, seed: int = 0):
    """X (n_trial, n_unit) aligned with `trials` (needs session_tag, subject, ycol).

    kind: 'corr' (continuous label) or 'diff' (0/1 label).
    -> (one row per unit, raw dict with session/subject-level effects for later use)
    """
    rng = np.random.default_rng(seed)
    use = trials[ycol].notna().to_numpy()
    tags, subs = trials.session_tag.to_numpy()[use], trials.subject.astype(str).to_numpy()[use]
    Y, Xu = trials[ycol].to_numpy(float)[use], X[use]
    usess, usubj = np.unique(tags), np.unique(subs)
    nU = X.shape[1]

    S = np.full((len(usess), nU), np.nan)
    sess_subj, prep = [], []
    for i, tag in enumerate(usess):
        m = tags == tag
        sess_subj.append(subs[m][0])
        Xi, yi = Xu[m], Y[m]
        ok_label = (m.sum() >= min_trials and np.std(yi) > 0) if kind == "corr" \
            else min((yi == 1).sum(), (yi == 0).sum()) >= 2
        good = np.isfinite(Xi).all(0)
        if not ok_label or not good.any():
            prep.append(None)
            continue
        Xc = Xi[:, good] - Xi[:, good].mean(0)
        if kind == "corr":
            den = np.sqrt((Xc ** 2).sum(0))
            den[den == 0] = np.inf
            Xc = Xc / den
        S[i, good] = Xc.T @ _weights(yi, kind)
        prep.append((good, Xc, yi))
    sess_subj = np.array(sess_subj)

    def subject_level(Sm):
        Z = _fisher(Sm) if kind == "corr" else Sm
        out = np.full(Z.shape[:-2] + (len(usubj), nU), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j, s in enumerate(usubj):
                out[..., j, :] = np.nanmean(Z[..., sess_subj == s, :], axis=-2)
        return out

    def group_t(B):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = np.sum(np.isfinite(B), axis=-2)
            m, sd = np.nanmean(B, axis=-2), np.nanstd(B, axis=-2, ddof=1)
            t = np.where((n >= min_subjects) & (sd > 0), m / (sd / np.sqrt(np.maximum(n, 1))), np.nan)
        return t, m, sd, n

    B = subject_level(S)
    t_obs, m_obs, sd_obs, n_obs = group_t(B)

    T_null = np.full((n_perm, nU), np.nan)
    for start in range(0, n_perm, 500):
        c = min(500, n_perm - start)
        Sp = np.full((c, len(usess), nU), np.nan)
        for i, pr in enumerate(prep):
            if pr is None:
                continue
            good, Xc, yi = pr
            W = np.column_stack([_weights(rng.permutation(yi), kind) for _ in range(c)])
            Sp[:, i, good] = (Xc.T @ W).T
        T_null[start:start + c] = group_t(subject_level(Sp))[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = np.isfinite(t_obs)
        fin = np.isfinite(T_null)
        ge = fin & (np.abs(T_null) >= np.abs(np.where(ok, t_obs, np.inf))[None, :])
        p_perm = np.where(ok, (1 + ge.sum(0)) / (fin.sum(0) + 1), np.nan)
        maxnull = np.nanmax(np.where(fin, np.abs(T_null), np.nan), axis=1)
        maxnull = maxnull[np.isfinite(maxnull)]
        p_max = np.array([(1 + np.sum(maxnull >= abs(t))) / (len(maxnull) + 1) if np.isfinite(t) else np.nan
                          for t in t_obs])
        d = np.where(ok, m_obs / sd_obs, np.nan)
        se_d = np.sqrt(1 / np.maximum(n_obs, 1) + d ** 2 / (2 * np.maximum(n_obs - 1, 1)))
        agree = np.nansum(np.sign(B) == np.sign(m_obs)[None, :], 0)
    res = pd.DataFrame(dict(unit=units, effect=m_obs, t=t_obs, n_subjects=n_obs,
                            cohen_d=d, d_lo=d - 1.96 * se_d, d_hi=d + 1.96 * se_d,
                            p_perm=p_perm, p_maxstat=p_max,
                            frac_agree=np.where(n_obs > 0, agree / np.maximum(n_obs, 1), np.nan)))
    return res, dict(S=S, B=B, sessions=usess, subjects=usubj, sess_subj=sess_subj)


def bh(p, q: float = 0.05):
    """Benjamini-Hochberg. -> (reject, q-values); NaNs pass through."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    rej, qv = np.zeros(p.shape, bool), np.full(p.shape, np.nan)
    if ok.any():
        pp = p[ok]
        o = np.argsort(pp)
        ranked = pp[o] * len(pp) / np.arange(1, len(pp) + 1)
        qs = np.minimum.accumulate(ranked[::-1])[::-1]
        qb = np.empty(len(pp))
        qb[o] = np.minimum(qs, 1.0)
        qv[ok], rej[ok] = qb, qb <= q
    return rej, qv


def lme_slope(v, x, groups, tags):
    """Trial-level mixed model, feature (z-scored within session) ~ label + (1 | subject)."""
    from statsmodels.regression.mixed_linear_model import MixedLM

    vz = pd.Series(v).groupby(tags).transform(lambda s: (s - s.mean()) / s.std()).to_numpy()
    ok = np.isfinite(vz) & np.isfinite(x)
    df = pd.DataFrame(dict(y=vz[ok], x=x[ok], g=groups[ok]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = MixedLM.from_formula("y ~ x", groups="g", data=df).fit(method="lbfgs")
    return dict(beta=fit.params["x"], se=fit.bse["x"], t=fit.tvalues["x"], p=fit.pvalues["x"],
                n_obs=int(ok.sum()), converged=bool(fit.converged))


def mdes(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest Cohen's d a one-sample two-sided t-test detects with n subjects."""
    df = n - 1
    crit = sps.t.ppf(1 - alpha / 2, df)
    for d in np.linspace(1e-3, 4, 4000):
        nc = d * np.sqrt(n)
        with np.errstate(all="ignore"):
            pw = sps.nct.sf(crit, df, nc) + sps.nct.cdf(-crit, df, nc)
        if not np.isfinite(pw):
            pw = sps.norm.sf(crit - nc)
        if pw >= power:
            return float(d)
    return np.nan


def split_half_reliability(raw: dict) -> tuple[float, float]:
    """Correlation of the group map from odd vs even sessions -> (r, Spearman-Brown r)."""
    S, subs, ss = raw["S"], raw["subjects"], raw["sess_subj"]
    ses_no = np.array([int(t.split("ses-")[1]) for t in raw["sessions"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.nanmean([np.nanmean(S[(ss == s) & (ses_no % 2 == 1)], 0) for s in subs], 0)
        b = np.nanmean([np.nanmean(S[(ss == s) & (ses_no % 2 == 0)], 0) for s in subs], 0)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan, np.nan
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return r, 2 * r / (1 + r)
