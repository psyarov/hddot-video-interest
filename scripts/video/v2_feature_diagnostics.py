"""Video step 2 — are the stimulus features usable as GLM regressors?

1. Each feature -> DOT sampling rate (4 Hz zero-phase low-pass + interpolation); cuts -> stick
   function; first/last `edge` s of each clip (fades) set to the clip's interior mean.
2. One timeline per video block in presentation order: instruction, then per clip
   [clip | rating screen | baseline], then a tail. Features are 0 outside clips; the
   "demeaned" version removes each clip's own mean (that level belongs to the clip regressor).
3. Convolution with an HRF (canonical double gamma; sensitivity: gamma with tau = 3 s).
4. Nuisance N1 per block: clip boxcar, clip onsets, clip offsets, rating screens, instruction,
   Legendre drift (order 1 + duration / 150 s).
Reported: correlations (before/after HRF, after N1), VIF within [features + N1], share of
variance left after N1, clips where a feature is flat, PCA of the low-level features, and
(if ratings exist) clip-level feature means vs mean interest.

Output: <derivatives>/group/video_features/diagnostics/*.csv + fig_correlations.png

    python scripts/video/v2_feature_diagnostics.py --config config/config.yaml
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.stats import spearmanr

from dotinterest.config import group_dir, load_config
from dotinterest.video_features import (FEATURES, clip_regressors, convolve, hrf_canonical, hrf_gamma,
                                        legendre_drift, residualize, vif)

LOWLEVEL = ["luminance", "contrast", "flow", "loudness", "flux"]
FLAT_REL = 0.1      # a clip is flat for a feature if its SD < 10 % of the pooled within-clip SD


def build_block(clips_b, reg, dt, a):
    segs, t0 = [], a.instr
    for uid in clips_b.video_uid:
        n = len(reg[uid][0])
        segs.append((uid, int(round(t0 / dt)), n))
        t0 += n * dt + a.rating + a.baseline
    N = int(round((t0 + a.tail) / dt))
    dm = {k: np.zeros(N) for k in FEATURES}
    raw = {k: np.zeros(N) for k in FEATURES}
    ev = {k: np.zeros(N) for k in ("box", "onset", "offset", "rating", "instr")}
    ev["instr"][: int(round(a.instr / dt))] = 1
    for uid, s, n in segs:
        for k in FEATURES:
            x = reg[uid][1][k]
            raw[k][s:s + n], dm[k][s:s + n] = x, x - x.mean()
        ev["box"][s:s + n] = 1
        ev["onset"][s] = 1
        ev["offset"][min(s + n, N - 1)] = 1
        ev["rating"][s + n:s + n + int(round(a.rating / dt))] = 1
    return dict(raw=raw, dm=dm, ev=ev, N=N)


def nuisance(b, h, dt):
    cols = [convolve(b["ev"][k], h) for k in ("onset", "offset", "rating", "instr", "box")]
    return np.column_stack(cols + [legendre_drift(b["N"], 1 + int(b["N"] * dt // 150))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fs", type=float, default=8.988764, help="DOT sampling rate (Hz)")
    ap.add_argument("--edge", type=float, default=1.5, help="s of fade excluded at each clip end")
    ap.add_argument("--instr", type=float, default=10.0, help="instruction screen before a block (s)")
    ap.add_argument("--rating", type=float, default=3.5, help="rating screen after a clip (s)")
    ap.add_argument("--baseline", type=float, default=10.0, help="fixation after the rating (s)")
    ap.add_argument("--tail", type=float, default=20.0, help="time after the last clip (s)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    vdir = group_dir(cfg, "video_features")
    out = vdir / "diagnostics"
    out.mkdir(exist_ok=True)
    dt = 1 / a.fs

    clips = pd.read_csv(vdir / "clips.csv")
    have = {int(p.stem.split("_")[1]) for p in (vdir / "features").glob("clip_*.npz") if ".tmp" not in p.name}
    clips = clips[clips.video_uid.isin(have)]
    reg = {u: clip_regressors(np.load(vdir / "features" / f"clip_{u:03d}.npz"), a.fs, a.edge) for u in clips.video_uid}
    vpb = cfg["paradigm"]["videos_per_block"]
    blocks = [build_block(g.sort_values("position"), reg, dt, a)
              for _, g in clips.groupby(["session", "block"]) if len(g) == vpb]
    print(f"{len(clips)} clips, {len(blocks)} complete blocks")

    wc = pd.DataFrame({k: np.concatenate([reg[u][1][k] - reg[u][1][k].mean() for u in clips.video_uid])
                       for k in FEATURES})
    pooled = wc.std()
    cs = pd.DataFrame([{"video_uid": u, **{f"{k}_mean": reg[u][1][k].mean() for k in FEATURES},
                        **{f"{k}_sd": reg[u][1][k].std() for k in FEATURES}} for u in clips.video_uid])
    flat = pd.Series({k: (cs[f"{k}_sd"] < FLAT_REL * pooled[k]).mean() for k in FEATURES})
    cs.to_csv(out / "clip_summary.csv", index=False)

    for hname, h in (("canonical", hrf_canonical(dt)), ("gamma3", hrf_gamma(dt))):
        Fraw = np.vstack([np.column_stack([convolve(b["raw"][k], h) for k in FEATURES]) for b in blocks])
        Fdm = np.vstack([np.column_stack([convolve(b["dm"][k], h) for k in FEATURES]) for b in blocks])
        N1 = block_diag(*[nuisance(b, h, dt) for b in blocks])
        Fr = residualize(Fdm, N1)
        corr = {"within_clip": wc.corr(), "convolved_raw": pd.DataFrame(Fraw, columns=FEATURES).corr(),
                "convolved_demeaned_after_N1": pd.DataFrame(Fr, columns=FEATURES).corr()}
        for k, c in corr.items():
            c.round(3).to_csv(out / f"corr_{hname}_{k}.csv")
        pd.DataFrame({"variance_left_after_N1": Fr.var(0) / Fdm.var(0), "flat_clip_fraction": flat.values,
                      "vif": vif(Fdm, N1)}, index=FEATURES).round(3).to_csv(out / f"regressors_{hname}.csv")
        if hname == "canonical":
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for ax, (k, c) in zip(axs, corr.items()):
                ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
                ax.set_yticks(range(len(FEATURES)), FEATURES)
                for i in range(len(FEATURES)):
                    for j in range(len(FEATURES)):
                        if i != j:
                            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
                ax.set_title(k.replace("_", " "))
            fig.tight_layout()
            fig.savefig(out / "fig_correlations.png", dpi=150)
            plt.close(fig)

    Z = (wc[LOWLEVEL] - wc[LOWLEVEL].mean()) / wc[LOWLEVEL].std()
    ev_, vec = np.linalg.eigh(np.cov(Z.values.T))
    o = np.argsort(ev_)[::-1]
    pd.DataFrame(vec[:, o], index=LOWLEVEL, columns=[f"PC{i + 1}" for i in range(len(LOWLEVEL))]) \
      .assign(eigenvalue=ev_[o]).round(3).to_csv(out / "pca_lowlevel.csv")

    rpath = group_dir(cfg, "ratings") / "ratings_all.csv"
    if rpath.exists():
        mi = pd.read_csv(rpath).groupby("video_uid").rating_z_session.mean()
        c2 = cs.assign(mean_interest=cs.video_uid.map(mi))
        pd.DataFrame([dict(feature=k, rho=spearmanr(c2[f"{k}_mean"], c2.mean_interest, nan_policy="omit")[0])
                      for k in FEATURES]).round(3).to_csv(out / "clip_mean_vs_interest.csv", index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
