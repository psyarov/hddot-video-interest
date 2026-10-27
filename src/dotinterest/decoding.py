"""Single-trial decoding with nested cross-validation and permutation inference.

Within subject: leave-one-session-out (each session holds different clips, so test clips are
never seen in training); the ridge penalty is chosen by an inner leave-one-session-out loop.
Cross subject: leave-one-subject-out, scored separately for each clip group (= session
number), in three conditions that share the same test trials:
    plain       train on all other subjects
    clip_blind  train on other subjects but only on clips the test trials do not contain
    matched     drop as many training trials as clip_blind, at random (training-size control)
Every participant saw the same clips in the same order, so `plain` can succeed by recognising
the clip and recalling how others rated it; `clip_blind` cannot.

Any scikit-learn model (elastic net, PLS, SVM/SVR, random forest, logistic regression, ...)
can be run through the same within-/cross-subject schemes with nested hyperparameter search
(`within_subject_sklearn`, `cross_subject_sklearn`); these are descriptive model comparisons.
Ridge regression (continuous target) and ridge classification (±1 target, threshold 0) have
closed forms, so all penalties are evaluated from one SVD per fold. The y-independent part
(standardisation + SVD) is cached across permutations; target-dependent feature selection is
recomputed. Scores: Spearman rho within each held-out session (Fisher-z averaged), or
balanced accuracy.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.model_selection import GroupKFold, KFold

ALPHAS = np.logspace(-2, 5, 15)
CHANCE = {"reg": 0.0, "clf": 0.5}


# --------------------------------------------------------------------------------------- #
# linear algebra
# --------------------------------------------------------------------------------------- #
def _standardise(A, B):
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    return (A - mu) / sd, (B - mu) / sd


def _pca(A, B, n):
    n = int(min(n, A.shape[1], max(A.shape[0] - 1, 1)))
    m = A.mean(0)
    _, _, Vt = np.linalg.svd(A - m, full_matrices=False)
    return (A - m) @ Vt[:n].T, (B - m) @ Vt[:n].T


def _select(A, B, y, k):
    """k features most correlated with the training target (same ranking as SelectKBest)."""
    if not k or k >= A.shape[1]:
        return A, B
    idx = np.sort(np.argsort(-np.abs(A.T @ (y - y.mean())), kind="stable")[:k])
    return A[:, idx], B[:, idx]


def _svd(A, B):
    m = A.mean(0)
    U, s, Vt = np.linalg.svd(A - m, full_matrices=False)
    return U, s, (B - m) @ Vt.T


def _ridge_path(U, s, Z, y, alphas=ALPHAS):
    """Ridge predictions for every alpha. -> (n_alpha, n_test)"""
    D = s[None, :] / (s[None, :] ** 2 + np.asarray(alphas, float)[:, None])
    return (D * (U.T @ (y - y.mean()))[None, :]) @ Z.T + y.mean()


def _bacc(t, P):
    P = np.atleast_2d(P)
    npos, nneg = max(int(t.sum()), 1), max(int((~t).sum()), 1)
    return 0.5 * ((P & t).sum(1) / npos + ((~P) & (~t)).sum(1) / nneg)


def _alpha_scores(P, y, task):
    """What GridSearchCV maximises: balanced accuracy, or R^2 against the fold's own mean."""
    if task == "clf":
        return _bacc(y > 0, P > 0)
    sst = float(((y - y.mean()) ** 2).sum())
    return np.zeros(P.shape[0]) if sst <= 0 else 1 - ((P - y[None, :]) ** 2).sum(1) / sst


def score(y, pred, task):
    if task == "clf":
        t = y > 0
        return float(_bacc(t, pred > 0)[0]) if 0 < t.sum() < len(t) else np.nan
    if np.std(pred) == 0 or np.std(y) == 0:
        return np.nan
    return float(sps.spearmanr(y, pred).statistic)


def _combine(scores, task):
    v = np.asarray([s for s in scores if np.isfinite(s)], float)
    if not len(v):
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(v, -.999, .999))))) if task == "reg" else float(v.mean())


def _prep(Xtr, Xte, ytr, pca=None, select=None):
    A, B = _standardise(Xtr, Xte)
    if select:
        A, B = _select(A, B, ytr, select)
    if pca:
        A, B = _pca(A, B, pca)
    return A, B


def _encode(y, task):
    return np.where(y > 0.5, 1.0, -1.0) if task == "clf" else np.asarray(y, float)


# --------------------------------------------------------------------------------------- #
# within subject
# --------------------------------------------------------------------------------------- #
def _inner_splits(sess_tr):
    """Leave-one-session-out; 3-fold for a subject with a single training session."""
    u = np.unique(sess_tr)
    if len(u) >= 2:
        return [(np.flatnonzero(sess_tr != s), np.flatnonzero(sess_tr == s)) for s in u]
    return list(KFold(3, shuffle=True, random_state=0).split(sess_tr))


class WithinSubject:
    """Leave-one-session-out ridge decoder for every subject, reusable across label vectors.

    X: (n_trial, n_feat); subject, session: per-trial arrays; labelled: trials that carry a
    label (the set must not change between permutations).
    """

    def __init__(self, X, subject, session, labelled, task, pca=None, select=None,
                 min_trials=12, min_train=10, min_test=4):
        self.X, self.task, self.pca, self.select = X, task, pca, select
        self.plan = []
        for sub in sorted(set(subject)):
            idx = np.flatnonzero((subject == sub) & labelled)
            sess = session[idx]
            folds = []
            if len(idx) >= min_trials and len(np.unique(sess)) >= 2:
                for held in np.unique(sess):
                    te, tr = np.flatnonzero(sess == held), np.flatnonzero(sess != held)
                    if len(tr) >= min_train and len(te) >= min_test:
                        folds.append(dict(tr=idx[tr], te=idx[te], sess_tr=sess[tr]))
            self.plan.append(dict(subject=sub, folds=folds))
        if not select:                       # cache the y-free decompositions
            for sp in self.plan:
                for f in sp["folds"]:
                    f["svd"] = _svd(*_prep(X[f["tr"]], X[f["te"]], None, pca))
                    f["inner"] = [(f["tr"][a], f["tr"][b],
                                   _svd(*_prep(X[f["tr"][a]], X[f["tr"][b]], None, pca)))
                                  for a, b in _inner_splits(f["sess_tr"])]

    def _fold(self, f, y):
        X, task = self.X, self.task
        if task == "clf" and len(np.unique(y[f["tr"]])) < 2:
            return np.nan
        acc = np.zeros(len(ALPHAS))
        inner = f.get("inner") or [(f["tr"][a], f["tr"][b], None) for a, b in _inner_splits(f["sess_tr"])]
        for i_tr, i_te, sv in inner:
            if task == "clf" and len(np.unique(y[i_tr])) < 2:
                continue
            sv = sv or _svd(*_prep(X[i_tr], X[i_te], y[i_tr], self.pca, self.select))
            acc += _alpha_scores(_ridge_path(*sv, y[i_tr]), y[i_te], task)
        a = ALPHAS[int(np.argmax(acc))]
        sv = f.get("svd") or _svd(*_prep(X[f["tr"]], X[f["te"]], y[f["tr"]], self.pca, self.select))
        return score(y[f["te"]], _ridge_path(*sv, y[f["tr"]], [a])[0], task)

    def run(self, y_raw):
        """-> (group mean score, per-subject scores)"""
        y = _encode(y_raw, self.task)
        per = np.array([_combine([self._fold(f, y) for f in sp["folds"]], self.task) if sp["folds"] else np.nan
                        for sp in self.plan])
        v = per[np.isfinite(per)]
        return (float(v.mean()) if len(v) >= 3 else np.nan), per

    @property
    def subjects(self):
        return [sp["subject"] for sp in self.plan]


# --------------------------------------------------------------------------------------- #
# scikit-learn decoders (any model; nested CV, no permutation null)
# --------------------------------------------------------------------------------------- #
def sklearn_models(task: str) -> dict:
    """name -> (estimator, hyperparameter grid). Random forests are left untuned."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
    from sklearn.svm import SVC, SVR

    if task == "reg":
        return {"ridge": (Ridge(), {"model__alpha": ALPHAS}),
                "enet": (ElasticNet(max_iter=5000, l1_ratio=0.5), {"model__alpha": np.logspace(-3, 1, 9)}),
                "pls": (PLSRegression(), {"model__n_components": [1, 2, 3, 5]}),
                "svr": (SVR(kernel="rbf", gamma="scale"), {"model__C": np.logspace(-1, 2, 4)}),
                "rf": (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1), {})}
    return {"logreg": (LogisticRegression(max_iter=5000), {"model__C": np.logspace(-3, 2, 11)}),
            "ridgeclf": (RidgeClassifier(), {"model__alpha": ALPHAS}),
            "svm": (SVC(kernel="rbf", gamma="scale"), {"model__C": np.logspace(-1, 2, 4)}),
            "rf": (RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1), {})}


def _pipeline(model, task, n_feat, pca=None, select=None):
    """scale -> [SelectKBest] -> [PCA] -> model, plus its grid."""
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    est, grid = sklearn_models(task)[model]
    steps = [("scale", StandardScaler())]
    if select:
        steps.append(("sel", SelectKBest(f_classif if task == "clf" else f_regression, k=min(select, n_feat))))
    if pca:
        steps.append(("pca", PCA(n_components=min(pca, n_feat), random_state=0)))
    return Pipeline(steps + [("model", clone(est))]), dict(grid)


def _fit_predict(Xtr, ytr, Xte, task, model, cv, groups=None, pca=None, select=None):
    from sklearn.model_selection import GridSearchCV

    if pca:     # never more components than the smallest (inner) training set supports
        n_min = min([len(a) for a, _ in cv.split(Xtr, ytr, groups)] + [len(Xtr)])
        pca = max(1, min(pca, n_min - 1))
    pipe, grid = _pipeline(model, task, Xtr.shape[1], pca, select)
    gs = GridSearchCV(pipe, grid, cv=cv, refit=True, n_jobs=1, error_score=np.nan,
                      scoring="balanced_accuracy" if task == "clf" else "r2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(Xtr, ytr, **({"groups": groups} if groups is not None else {}))
        return np.asarray(gs.predict(Xte), float).ravel()


def _score_pred(y, pred, task):
    """Classifiers predict 0/1 labels; regressors a continuous value."""
    return score(np.where(y > 0.5, 1.0, -1.0), pred - 0.5, task) if task == "clf" else score(y, pred, task)


def within_subject_sklearn(X, subject, session, y, task, model, pca=None, select=None,
                           min_trials=12, min_train=10, min_test=4):
    """Leave-one-session-out within each subject; hyperparameters from an inner
    leave-one-session-out loop (stratified/plain 3-fold with a single training session).
    -> per-subject scores (Fisher-z mean of session rho, or mean balanced accuracy)"""
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    per = []
    for sub in sorted(set(subject)):
        m = (subject == sub) & np.isfinite(y)
        Xs, ys, ss = X[m], y[m], session[m]
        if m.sum() < min_trials or len(np.unique(ss)) < 2:
            per.append(np.nan)
            continue
        if task == "clf":
            ys = ys.astype(int)
        fold = []
        for held in np.unique(ss):
            tr, te = ss != held, ss == held
            if tr.sum() < min_train or te.sum() < min_test or (task == "clf" and len(np.unique(ys[tr])) < 2):
                continue
            if len(np.unique(ss[tr])) >= 2:
                cv, g = LeaveOneGroupOut(), ss[tr]
            else:
                cv, g = (StratifiedKFold(3, shuffle=True, random_state=0) if task == "clf"
                         else KFold(3, shuffle=True, random_state=0)), None
            pred = _fit_predict(Xs[tr], ys[tr], Xs[te], task, model, cv, g, pca, select)
            fold.append(_score_pred(ys[te], pred, task))
        per.append(_combine(fold, task))
    return np.array(per)


def cross_subject_sklearn(X, subject, session, y, task, model, clip_blind=True, pca=None, select=None,
                          min_test=8, min_train=40):
    """Leave-one-subject-out; inner 3-fold grouped by subject; scored within each held-out
    session. clip_blind=True trains only on sessions (clip groups) the test session does not
    contain. -> per-subject scores"""
    per = []
    for s in sorted(set(subject)):
        sc = []
        for g in np.unique(session[(subject == s) & np.isfinite(y)]):
            te = (subject == s) & (session == g) & np.isfinite(y)
            tr = (subject != s) & np.isfinite(y) & ((session != g) if clip_blind else True)
            if te.sum() < min_test or tr.sum() < min_train or (task == "clf" and len(np.unique(y[tr])) < 2):
                continue
            ytr = y[tr].astype(int) if task == "clf" else y[tr]
            pred = _fit_predict(X[tr], ytr, X[te], task, model, GroupKFold(3), subject[tr], pca, select)
            sc.append(_score_pred(y[te], pred, task))
        per.append(_combine(sc, task))
    return np.array(per)


# --------------------------------------------------------------------------------------- #
# cross subject with the clip-identity control
# --------------------------------------------------------------------------------------- #
class CrossSubject:
    """Leave-one-subject-out decoder, one fold per (held-out subject, clip group)."""

    def __init__(self, X, subject, clip_group, labelled, conditions=("plain", "clip_blind", "matched"),
                 n_matched=3, min_test=8, min_train=40, seed=12345):
        rng = np.random.default_rng(seed)
        self.X, self.folds, store = X, [], {}

        def cached(tr, te):
            key = (tr.tobytes(), te.tobytes())
            if key not in store:
                store[key] = _svd(*_standardise(X[tr], X[te]))
            return store[key]

        for s in sorted(set(subject)):
            for g in np.unique(clip_group[(subject == s) & labelled]):
                te = (subject == s) & (clip_group == g) & labelled
                other = (subject != s) & labelled
                blind = other & (clip_group != g)
                if te.sum() < min_test or blind.sum() < min_train:
                    continue
                masks = {"plain": [other], "clip_blind": [blind], "matched": []}
                pool, n_drop = np.flatnonzero(other), int(other.sum() - blind.sum())
                for _ in range(n_matched):
                    m = other.copy()
                    m[rng.choice(pool, n_drop, replace=False)] = False
                    masks["matched"].append(m)
                te_idx = np.flatnonzero(te)
                fits = {}
                for c in conditions:
                    fits[c] = []
                    for m in masks[c]:
                        tr = np.flatnonzero(m)
                        inner = [(tr[a], tr[b], cached(tr[a], tr[b]))
                                 for a, b in GroupKFold(3).split(tr, groups=subject[tr])]
                        fits[c].append(dict(tr=tr, svd=cached(tr, te_idx), inner=inner))
                self.folds.append(dict(subject=s, group=g, te=te_idx, fits=fits))

    def run(self, y_raw, task, conditions=("plain", "clip_blind", "matched")):
        """-> DataFrame: one row per (subject, condition) with the score averaged over clip groups."""
        y = _encode(y_raw, task)
        rows = []
        for f in self.folds:
            for c in conditions:
                sc = []
                for fit in f["fits"][c]:
                    tr = fit["tr"]
                    if task == "clf" and len(np.unique(y[tr])) < 2:
                        continue
                    acc = sum(_alpha_scores(_ridge_path(*sv, y[a]), y[b], task)
                              for a, b, sv in fit["inner"])
                    alpha = ALPHAS[int(np.argmax(acc))]
                    sc.append(score(y[f["te"]], _ridge_path(*fit["svd"], y[tr], [alpha])[0], task))
                if any(np.isfinite(sc)):
                    rows.append(dict(subject=f["subject"], group=f["group"], condition=c,
                                     score=np.nanmean(sc)))
        df = pd.DataFrame(rows)
        if not len(df):
            return df
        return (df.groupby(["subject", "condition"]).score
                .apply(lambda v: _combine(v, task)).reset_index())


# --------------------------------------------------------------------------------------- #
# permutation inference
# --------------------------------------------------------------------------------------- #
def shuffle_within_session(y, session_tag, rng):
    """Permute labels among the labelled trials of each session."""
    out = y.copy()
    for s in np.unique(session_tag):
        idx = np.flatnonzero((session_tag == s) & np.isfinite(y))
        if len(idx) > 1:
            out[idx] = y[rng.permutation(idx)]
    return out


def maxstat_p(obs: np.ndarray, null: np.ndarray):
    """Per-cell p and max-statistic p over all cells (null: n_perm x n_cell, higher = better)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_cell = np.array([(1 + np.nansum(null[:, j] >= obs[j])) / (np.isfinite(null[:, j]).sum() + 1)
                           for j in range(len(obs))])
        mx = np.nanmax(null, axis=1)
        mx = mx[np.isfinite(mx)]
        p_max = np.array([(1 + np.sum(mx >= o)) / (len(mx) + 1) if np.isfinite(o) else np.nan for o in obs])
    return p_cell, p_max


def minp_corrected(obs: np.ndarray, null: np.ndarray):
    """Min-p correction across cells on different scales (null: n_cell x n_perm)."""
    n = null.shape[1]
    p_cell = np.array([(1 + np.sum(null[k] >= obs[k])) / (n + 1) for k in range(len(obs))])
    rank = (null[:, :, None] >= null[:, None, :]).sum(1) / n       # p of every null draw
    minp = rank.min(0)
    return p_cell, np.array([(1 + np.sum(minp <= p)) / (n + 1) for p in p_cell])


def group_summary(per_subject, task) -> dict:
    v = np.asarray([s for s in per_subject if np.isfinite(s)], float)
    if len(v) < 3:
        return dict(n=len(v), mean=np.nan)
    t, p = sps.ttest_1samp(v, CHANCE[task])
    return dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1), t=t, p_t=p,
                n_above_chance=int((v > CHANCE[task]).sum()))
