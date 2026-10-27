"""Step 7 — single-trial decoding of interest, with the corrections a search of this size needs.

    A  positive control   video vs rest, within subject, `control_models` (must be clearly > 0.5)
    B  within subject     every feature set (as-is, 10 PCs, 20 best features) x
                          {rating_z_session: Spearman rho, high vs low tertile: balanced accuracy};
                          labels shuffled within session and the WHOLE grid re-run per
                          permutation -> per-cell p and max-statistic p over the grid
    C  cross subject      plain / clip_blind / matched (see dotinterest/decoding.py), descriptive
    D  cross subject      clip_blind only, sets x {rating, high/low}, min-p corrected
    E  model comparison   every model in `models` x feature set, within subject (nested CV,
                          t-test of subject scores vs chance; descriptive, no permutation null)
    F  model comparison   the same models across subjects, plain and clip_blind (descriptive)

Ridge / ridge classifier carry the permutation inference (B, D) because their closed form makes
re-running the whole search thousands of times affordable. E and F show whether other
learners (elastic net, PLS, SVM/SVR, random forest, logistic regression) do any better.

Output (<derivatives>/group/decoding/):
    positive_control.csv, within_grid.csv, cross_subject_conditions.csv,
    cross_subject_corrected.csv, model_comparison_within.csv, model_comparison_cross.csv

    python scripts/07_decoding.py --config config/config.yaml [--n-perm 1000] [--stages ABCDEF]
                                  [--models ridge,enet,logreg,svm] [--sets B_mean,B_amplitude]
"""
import argparse
import time

import numpy as np
import pandas as pd

from dotinterest.config import group_dir, load_config
from dotinterest.decoding import (CHANCE, CrossSubject, WithinSubject, group_summary, maxstat_p,
                                  minp_corrected, shuffle_within_session, sklearn_models,
                                  cross_subject_sklearn, within_subject_sklearn)
from dotinterest.parcels import network_hemi

MAX_COL_NAN = 0.02      # a feature missing in more than 2 % of trials is dropped


def build_blocks(P, FC, T, min_parcels):
    """{'net:<family>:<chromo>' | 'par:...': X aligned with T}; columns present in every session."""
    key = ["session_tag", "trial"]
    tidx = pd.MultiIndex.from_frame(T[key])
    P = P.assign(network=[network_hemi(p) for p in P.parcel])
    fams = [c for c in P.columns if c not in key + ["subject", "session", "parcel", "chromo", "network"]]
    blocks = {}

    def add(name, W):
        X = W.reindex(tidx).to_numpy(float)
        present = pd.DataFrame(np.isfinite(X)).groupby(T.session_tag.to_numpy()).any().all(0).to_numpy()
        if present.any():
            blocks[name] = X[:, present]

    for fam in fams:
        for ch in ("HbO", "HbR"):
            d = P.loc[P.chromo == ch, key + ["parcel", "network", fam]].dropna(subset=[fam])
            if d.empty:
                continue
            agg = d.groupby(key + ["network"])[fam].agg(["mean", "count"])
            add(f"net:{fam}:{ch}", agg[agg["count"] >= min_parcels]["mean"].unstack("network"))
            add(f"par:{fam}:{ch}", d.pivot_table(index=key, columns="parcel", values=fam))
    for ch in ("HbO", "HbR"):
        add(f"net:fc:{ch}", FC[FC.chromo == ch].pivot_table(index=key, columns="edge", values="fc"))
    return blocks


def complete(X, sessions):
    """Drop sparse columns; fill remaining gaps with the session median (then the grand median)."""
    X = X[:, np.isnan(X).mean(0) <= MAX_COL_NAN].copy()
    for s in np.unique(sessions[np.isnan(X).any(1)]):
        m = sessions == s
        blk = X[m]
        r, c = np.where(np.isnan(blk))
        blk[r, c] = np.nanmedian(blk, axis=0)[c]
        X[m] = blk
    r, c = np.where(np.isnan(X))
    X[r, c] = np.nan_to_num(np.nanmedian(X, axis=0))[c]
    return X


def feature_sets(cfg, blocks, sessions):
    out = {}
    for name, parts in cfg["decoding"]["feature_sets"].items():
        names = sorted(k for k in blocks if k.startswith("net:")) if parts == ["net:*"] else parts
        Xs = [blocks[n] for n in names if n in blocks]
        if Xs:
            out[name] = complete(np.hstack(Xs), sessions)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-perm", type=int)
    ap.add_argument("--n-perm-cross", type=int)
    ap.add_argument("--stages", default="ABCDEF")
    ap.add_argument("--models", help="comma-separated subset of decoding.models (stages E, F)")
    ap.add_argument("--sets", help="comma-separated feature sets for stages E, F (default: all)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    dc, st = cfg["decoding"], cfg["stats"]
    n_perm, n_perm_x = a.n_perm or dc["n_perm"], a.n_perm_cross or dc["n_perm_cross"]
    fdir, out = group_dir(cfg, "features"), group_dir(cfg, "decoding")

    P = pd.read_csv(fdir / "parcel_features.csv.gz", dtype={"subject": str})
    FC = pd.read_csv(fdir / "fc_features.csv.gz", dtype={"subject": str})
    T = pd.read_csv(fdir / "trials.csv", dtype={"subject": str}).sort_values(["session_tag", "trial"],
                                                                             ignore_index=True)
    sess_tag, subject, session = T.session_tag.to_numpy(), T.subject.to_numpy(), T.session.to_numpy()

    # targets
    hi_lo = np.full(len(T), np.nan)
    for _, g in T[T.is_video].groupby("session_tag"):
        v = g.rating_z_session.dropna()
        if len(v) >= 6:
            lo, hi = v.quantile([st["tertile"], 1 - st["tertile"]])
            hi_lo[v.index[v <= lo]], hi_lo[v.index[v >= hi]] = 0.0, 1.0
    targets = {"video_vs_rest": (T.is_video.astype(float).to_numpy(), "clf"),
               "rating_cont": (T.rating_z_session.where(T.is_video).to_numpy(float), "reg"),
               "hi_lo": (hi_lo, "clf"),
               "normative": (T.rating_normative.to_numpy(float) if "rating_normative" in T else
                             np.full(len(T), np.nan), "reg")}

    FS = feature_sets(cfg, build_blocks(P, FC, T, st["min_parcels_per_network"]), sess_tag)
    del P, FC
    print(f"{T.subject.nunique()} subjects | feature sets: " + ", ".join(f"{k} ({v.shape[1]})" for k, v in FS.items()))
    rng = np.random.default_rng(dc["seed"])

    if "A" in a.stages:
        y, task = targets["video_vs_rest"]
        rows = []
        for name in [k for k in ("B_mean_HbO", "B_amplitude", "A_glm_net") if k in FS]:
            for model in dc["control_models"]:
                per = within_subject_sklearn(FS[name], subject, session, y, task, model)
                rows.append(dict(feature_set=name, model=model, **group_summary(per, task)))
                print(f"  [A] video vs rest, {name}, {model}: balanced accuracy {np.nanmean(per):.3f}")
        pd.DataFrame(rows).to_csv(out / "positive_control.csv", index=False)

    if "B" in a.stages:
        specs = []
        for name, X in FS.items():
            specs.append((name, None, None))
            if X.shape[1] > 40:
                specs += [(name, dc["wide_pca"], None), (name, None, dc["wide_select"])]
        rows = []
        for tname in ("rating_cont", "hi_lo"):
            y, task = targets[tname]
            lab = np.isfinite(y)
            models = [WithinSubject(FS[n], subject, session, lab, task, pca=pc, select=sel) for n, pc, sel in specs]
            obs = np.array([m.run(y)[0] for m in models])
            t0 = time.time()
            null = np.full((n_perm, len(models)), np.nan)
            for i in range(n_perm):
                yp = shuffle_within_session(y, sess_tag, rng)
                null[i] = [m.run(yp)[0] for m in models]
                if (i + 1) % max(1, n_perm // 10) == 0:
                    print(f"  [B] {tname}: {i + 1}/{n_perm} permutations ({(time.time() - t0) / 60:.1f} min)")
            p_cell, p_max = maxstat_p(obs, null)
            for k, (n, pc, sel) in enumerate(specs):
                rows.append(dict(target=tname, feature_set=n, pca=pc, select=sel, n_features=FS[n].shape[1],
                                 score=obs[k], chance=CHANCE[task], null_mean=np.nanmean(null[:, k]),
                                 null_p95=np.nanpercentile(null[:, k], 95), p_cell=p_cell[k], p_maxstat=p_max[k]))
            best = int(np.nanargmax(obs))
            print(f"  [B] {tname}: best {specs[best]} = {obs[best]:.3f}, max-statistic p = {p_max[best]:.3f}")
        pd.DataFrame(rows).to_csv(out / "within_grid.csv", index=False)

    xsets = [s for s in dc["cross_subject_sets"] if s in FS]
    if "C" in a.stages:
        rows = []
        for name in xsets:
            for tname, (y, task) in targets.items():
                if not np.isfinite(y).any():
                    continue
                df = CrossSubject(FS[name], subject, session, np.isfinite(y)).run(y, task)
                for c, g in df.groupby("condition"):
                    rows.append(dict(feature_set=name, target=tname, condition=c, **group_summary(g.score, task)))
                print(f"  [C] {name}/{tname}: " + ", ".join(
                    f"{c} {g.score.mean():.3f}" for c, g in df.groupby("condition")))
        pd.DataFrame(rows).to_csv(out / "cross_subject_conditions.csv", index=False)

    if "D" in a.stages:
        cells, obs, null = [], [], []
        for tname in ("hi_lo", "rating_cont"):
            y, task = targets[tname]
            perms = [shuffle_within_session(y, sess_tag, np.random.default_rng([dc["seed"], i]))
                     for i in range(n_perm_x)]
            for name in xsets:
                cs = CrossSubject(FS[name], subject, session, np.isfinite(y), conditions=("clip_blind",))
                obs.append(cs.run(y, task, ("clip_blind",)).score.mean())
                null.append([cs.run(yp, task, ("clip_blind",)).score.mean() for yp in perms])
                cells.append((name, tname, task))
                print(f"  [D] {name}/{tname}: {obs[-1]:.3f} (null mean {np.nanmean(null[-1]):.3f})")
        p_cell, p_corr = minp_corrected(np.array(obs), np.array(null))
        pd.DataFrame([dict(feature_set=n, target=t, chance=CHANCE[k], score=o, p_cell=pc, p_corrected=pk)
                      for (n, t, k), o, pc, pk in zip(cells, obs, p_cell, p_corr)]) \
          .sort_values("p_corrected").to_csv(out / "cross_subject_corrected.csv", index=False)
    # E, F: every configured model; parcel-level / very wide sets get PCA as in the grid
    want = set(a.models.split(",")) if a.models else None
    esets = a.sets.split(",") if a.sets else list(FS)
    for stage, fname in (("E", "model_comparison_within.csv"), ("F", "model_comparison_cross.csv")):
        if stage not in a.stages:
            continue
        rows = []
        for tname in ("rating_cont", "hi_lo"):
            y, task = targets[tname]
            for model in dc["models"][task]:
                if want and model not in want:
                    continue
                if model not in sklearn_models(task):
                    raise ValueError(f"unknown {task} model {model!r}; choose from {list(sklearn_models(task))}")
                for name in esets:
                    X = FS[name]
                    pca = dc["wide_pca"] if X.shape[1] > 40 else None
                    variants = ([("within", within_subject_sklearn(X, subject, session, y, task, model, pca))]
                                if stage == "E" else
                                [(c, cross_subject_sklearn(X, subject, session, y, task, model, c == "clip_blind", pca))
                                 for c in ("plain", "clip_blind")])
                    for cond, per in variants:
                        r = group_summary(per, task)
                        rows.append(dict(target=tname, model=model, feature_set=name, condition=cond,
                                         n_features=X.shape[1], pca=pca, chance=CHANCE[task], **r))
                        print(f"  [{stage}] {tname} {model:8s} {name:14s} {cond:10s} "
                              f"{r.get('mean', np.nan):+.3f} (p_t = {r.get('p_t', np.nan):.3f})")
        pd.DataFrame(rows).to_csv(out / fname, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
