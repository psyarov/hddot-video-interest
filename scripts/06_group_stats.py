"""Step 6 — group-level statistics (subject = unit of inference).

Contrasts
    video_vs_rest    manipulation check; calibrates what a real effect looks like here
    interest_cont    feature ~ rating_z_session within session (primary)
    interest_hi_lo   top vs bottom tertile of each session's ratings (secondary)
Each is run per unit (network x hemisphere, and parcel for the primary families), per family and
chromophore, on raw features and after subtracting each trial's brain-wide mean (specificity
control; magnitudes only). `ttp` is a negative control and must come out null for interest.

Output (<derivatives>/group/stats/):
    group_results.csv.gz      every unit test: effect, t, Cohen's d, p_perm, p_maxstat, q (BH)
    summary_network.csv       per contrast x family x chromo: n significant, strongest unit
    lme_results.csv           trial-level mixed model, network level (optional)
    specificity.csv           significant units before vs after removing the brain-wide mean
    reliability.csv           split-half (odd/even sessions) reliability of the group maps
    sensitivity.csv           smallest detectable Cohen's d by number of subjects

    python scripts/06_group_stats.py --config config/config.yaml [--n-perm 5000] [--no-parcels]
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from dotinterest.config import group_dir, load_config
from dotinterest.parcels import network_hemi
from dotinterest.stats import bh, group_test, lme_slope, mdes, split_half_reliability

MAGNITUDE = {"glm_auc", "mean", "slope", "std", "stdderiv", "lowfreq", "ttp",
             "mean_early", "mean_sustained", "slope_early", "slope_sustained"}
CONTRASTS = {"video_vs_rest": ("is_video01", "diff"), "interest_cont": ("rating_z_session", "corr"),
             "interest_hi_lo": ("hi_lo", "diff")}


def tertile_split(T, q):
    out = pd.Series(np.nan, index=T.index)
    for _, g in T[T.is_video].groupby("session_tag"):
        v = g.rating_z_session.dropna()
        if len(v) >= 6:
            lo, hi = v.quantile([q, 1 - q])
            out[v.index[v <= lo]] = 0.0
            out[v.index[v >= hi]] = 1.0
    return out


def parcel_mask(P, T, rule):
    """{session_tag: allowed parcels}. subject_common: sensitive in every session of the subject."""
    if rule == "all":
        return None
    has = P.groupby(["session_tag", "parcel"]).size().reset_index()[["session_tag", "parcel"]]
    sub_of = T.drop_duplicates("session_tag").set_index("session_tag").subject
    has["subject"] = has.session_tag.map(sub_of)
    sets = has.groupby("session_tag").parcel.agg(set)
    if rule == "global_common":
        keep = set.intersection(*sets)
        return {t: keep for t in sets.index}
    per_sub = {s: set.intersection(*[sets[t] for t in g.session_tag.unique()]) for s, g in has.groupby("subject")}
    return {t: per_sub[sub_of[t]] for t in sets.index}


def matrices(P, FC, T, s, levels):
    """{(level, family, chromo, corrected): (X aligned with T, unit names)}"""
    key = ["session_tag", "trial"]
    tidx = pd.MultiIndex.from_frame(T[key])
    fams = [f for f in s["families"] + s["exploratory"] + s["negative_control"] if f in P.columns]
    mask = parcel_mask(P, T, s["parcel_set"])
    if mask is not None:
        P = P[[p in mask[t] for t, p in zip(P.session_tag, P.parcel)]]
    P = P.assign(network=[network_hemi(p) for p in P.parcel])
    mag = [f for f in fams if f in MAGNITUDE]
    gm = P.groupby(["session_tag", "trial", "chromo"])[mag].transform("mean")
    for f in mag:
        P[f + "__gc"] = P[f] - gm[f]

    def place(W):
        W = W.reindex(tidx)
        return W.to_numpy(float), np.array(W.columns)

    out = {}
    for fam in fams:
        for ch in ("HbO", "HbR"):
            for corr in (False, True):
                if corr and (fam not in mag or fam not in s["families"]):
                    continue
                col = fam + ("__gc" if corr else "")
                d = P.loc[P.chromo == ch, key + ["parcel", "network", col]].dropna(subset=[col])
                if d.empty:
                    continue
                agg = d.groupby(key + ["network"])[col].agg(["mean", "count"])
                agg = agg[agg["count"] >= s["min_parcels_per_network"]]["mean"].unstack("network")
                out[("network", fam, ch, corr)] = place(agg)
                if "parcel" in levels and fam in s["families"] + s["negative_control"]:
                    out[("parcel", fam, ch, corr)] = place(d.pivot_table(index=key, columns="parcel", values=col))
    if "fc" in s["families"]:
        for ch in ("HbO", "HbR"):
            d = FC[FC.chromo == ch]
            out[("network", "fc", ch, False)] = place(d.pivot_table(index=key, columns="edge", values="fc"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-perm", type=int)
    ap.add_argument("--no-parcels", action="store_true", help="network level only (much faster)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    s = cfg["stats"]
    n_perm = a.n_perm or s["n_perm"]
    fdir, out = group_dir(cfg, "features"), group_dir(cfg, "stats")

    P = pd.read_csv(fdir / "parcel_features.csv.gz", dtype={"subject": str})
    FC = pd.read_csv(fdir / "fc_features.csv.gz", dtype={"subject": str})
    T = pd.read_csv(fdir / "trials.csv", dtype={"subject": str}).sort_values(["session_tag", "trial"],
                                                                             ignore_index=True)
    T["is_video01"] = T.is_video.astype(float)
    T["hi_lo"] = tertile_split(T, s["tertile"])
    T["rating_z_session"] = T.rating_z_session.where(T.is_video)
    print(f"{T.subject.nunique()} subjects, {T.session_tag.nunique()} sessions, {len(T)} trials")

    M = matrices(P, FC, T, s, ["network"] if a.no_parcels else ["network", "parcel"])
    del P
    print(f"{len(M)} matrices x {len(CONTRASTS)} contrasts x {n_perm} permutations")

    res, raw = [], {}
    for cname, (ycol, kind) in CONTRASTS.items():
        for (level, fam, ch, corr), (X, units) in M.items():
            df, rw = group_test(X, units, T, ycol, kind, n_perm, s["min_trials_per_session"],
                                s["min_subjects"], s["seed"])
            df.insert(0, "contrast", cname)
            df.insert(1, "level", level)
            df.insert(2, "family", fam)
            df.insert(3, "chromo", ch)
            df.insert(4, "corrected", corr)
            res.append(df)
            raw[(cname, level, fam, ch, corr)] = rw
        print(f"  {cname} done")
    R = pd.concat(res, ignore_index=True)
    R["q_perm"], R["sig_fdr"] = np.nan, False
    for _, g in R.groupby(["contrast", "level", "family", "chromo", "corrected"]):
        rej, q = bh(g.p_perm.to_numpy(), s["fdr_q"])
        R.loc[g.index, "q_perm"], R.loc[g.index, "sig_fdr"] = q, rej
    R["sig_maxstat"] = R.p_maxstat < s["fdr_q"]
    R.to_csv(out / "group_results.csv.gz", index=False, float_format="%.6g")

    rows = []
    for (c, fam, ch), g in R[(R.level == "network") & ~R.corrected].groupby(["contrast", "family", "chromo"]):
        g = g[np.isfinite(g.t)]
        if len(g):
            top = g.loc[g.t.abs().idxmax()]
            rows.append(dict(contrast=c, family=fam, chromo=ch, n_units=len(g), n_sig_fdr=int(g.sig_fdr.sum()),
                             n_sig_maxstat=int(g.sig_maxstat.sum()), top_unit=top.unit, top_t=top.t,
                             top_d=top.cohen_d, top_p_perm=top.p_perm, top_p_maxstat=top.p_maxstat,
                             top_frac_agree=top.frac_agree))
    S = pd.DataFrame(rows)
    S.to_csv(out / "summary_network.csv", index=False)
    for c in CONTRASTS:
        print(f"\n{c}\n" + S[S.contrast == c].drop(columns="contrast").round(3).to_string(index=False))

    ctrl = R[R.family.isin(s["negative_control"]) & (R.contrast != "video_vs_rest") & R.sig_fdr]
    print(f"\nnegative control ({', '.join(s['negative_control'])}): {len(ctrl)} significant interest units")

    # specificity: does an effect survive removing the brain-wide mean?
    A = R[(R.level == "network") & ~R.corrected]
    B = R[(R.level == "network") & R.corrected]
    m = A.merge(B, on=["contrast", "level", "family", "chromo", "unit"], suffixes=("_raw", "_gc"))
    m.groupby(["contrast", "family", "chromo"]).apply(
        lambda g: pd.Series(dict(n_sig_raw=g.sig_fdr_raw.sum(), n_sig_gc=g.sig_fdr_gc.sum(),
                                 n_survive=(g.sig_fdr_raw & g.sig_fdr_gc).sum())),
        include_groups=False).reset_index().to_csv(out / "specificity.csv", index=False)

    rel = [dict(zip(("contrast", "family", "chromo", "r_half", "reliability"),
                    (c, f, ch, *split_half_reliability(raw[(c, "network", f, ch, False)]))))
           for c in CONTRASTS for f in ("mean", "glm_auc") for ch in ("HbO", "HbR")
           if (c, "network", f, ch, False) in raw]
    pd.DataFrame(rel).to_csv(out / "reliability.csv", index=False)
    n_sub = int(R.n_subjects.max())
    pd.DataFrame([dict(n_subjects=n, mdes_d=mdes(n)) for n in range(4, n_sub + 20)]) \
      .to_csv(out / "sensitivity.csv", index=False)
    print(f"smallest detectable d with {n_sub} subjects (80 % power): {mdes(n_sub):.2f}")

    if s.get("lme"):
        rows = []
        for cname, (ycol, _) in CONTRASTS.items():
            use = T[ycol].notna().to_numpy()
            for (level, fam, ch, corr), (X, units) in M.items():
                if level != "network" or corr:
                    continue
                for j, u in enumerate(units):
                    v = X[use, j]
                    if np.isfinite(v).sum() < 50:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            r = lme_slope(v, T[ycol].to_numpy(float)[use], T.subject.to_numpy()[use],
                                          T.session_tag.to_numpy()[use])
                        rows.append(dict(contrast=cname, family=fam, chromo=ch, unit=u, **r))
                    except Exception:
                        continue
        L = pd.DataFrame(rows)
        if len(L):
            L["q"] = L.groupby(["contrast", "family", "chromo"]).p.transform(lambda p: bh(p.to_numpy())[1])
            L.to_csv(out / "lme_results.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
