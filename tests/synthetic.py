"""Synthetic study in the derivatives layout of step 3, for testing steps 4-7 without data.

Every subject sees the same clips in the same order. Parcel responses to a clip are a shared
clip-specific amplitude + noise; in the `Vis` parcels HbO additionally scales with the
viewer's rating (the planted interest effect), and every video trial raises HbO over rest.

    python tests/synthetic.py --out demo/          # writes demo/config.yaml + demo/derivatives/
    python scripts/04_glm.py --config demo/config.yaml
    ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from dotinterest.io import save_timeseries

ROOT = Path(__file__).resolve().parents[1]
FS = 8.988764
PARCELS = [f"{net}_{i}_{h}" for net in ("VisCent", "SomMotA", "DorsAttnA", "DefaultA", "TempPar")
           for i in (1, 2, 3) for h in ("LH", "RH")]
CLIP_S, RATE_S, BASE_S, INSTR_S, REST_S = 60.0, 3.5, 10.0, 10.0, 120.0


def hrf(t):
    from scipy.stats import gamma
    return gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6


def make_session(rng, n_blocks, vpb, clip_amp, ratings, effect):
    """-> blocks {key: (parcel, chromo, time) array, time}, events DataFrame."""
    ev, blocks, t = [], {}, 20.0
    k = 0
    vis = np.array([p.startswith("Vis") for p in PARCELS])
    h = hrf(np.arange(0, 30, 1 / FS))
    for b in range(1, n_blocks + 1):
        t0 = t
        ev.append(dict(onset=t, duration=INSTR_S, trial_type=f"videos_instr_{b}"))
        t += INSTR_S
        for _ in range(vpb):
            k += 1
            ev += [dict(onset=t, duration=CLIP_S, trial_type=f"video_{k}"),
                   dict(onset=t + CLIP_S, duration=RATE_S, trial_type=f"slider_{k}"),
                   dict(onset=t + CLIP_S + RATE_S, duration=BASE_S, trial_type=f"post_rating_baseline_{k}")]
            t += CLIP_S + RATE_S + BASE_S
        time = np.arange(t0 - 30, t + 30, 1 / FS)
        Y = 0.3 * rng.standard_normal((len(PARCELS), 2, len(time)))
        for e in ev:
            if e["trial_type"].startswith("video_") and t0 <= e["onset"] < t:
                i = int(e["trial_type"].split("_")[1]) - 1
                box = ((time >= e["onset"]) & (time < e["onset"] + CLIP_S)).astype(float)
                resp = np.convolve(box, h)[: len(time)] / h.sum()
                amp = 1.0 + clip_amp[i] + effect * ratings[i] * vis
                Y[:, 0] += np.outer(amp * np.ones(len(PARCELS)), resp)
                Y[:, 1] -= 0.3 * np.outer(amp * np.ones(len(PARCELS)), resp)
        blocks[f"video_{b}"] = (Y, time)
        ev.append(dict(onset=t, duration=REST_S, trial_type=f"rest_{b}"))
        time = np.arange(t - 30, t + REST_S + 30, 1 / FS)
        blocks[f"rest_{b}"] = (0.3 * rng.standard_normal((len(PARCELS), 2, len(time))), time)
        t += REST_S + 30
    ev.append(dict(onset=t, duration=0.0, trial_type="end"))
    return blocks, pd.DataFrame(ev).assign(value=1.0)[["onset", "duration", "value", "trial_type"]]


def write_study(out: Path, n_sub=10, n_ses=2, n_blocks=2, vpb=8, effect=0.3, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out)
    deriv = out / "derivatives"
    n_clip = n_blocks * vpb
    rows, ratings = [], []
    for s in range(1, n_ses + 1):
        clip_amp = 0.2 * rng.standard_normal(n_clip)                 # shared clip response
        clip_mean = rng.standard_normal(n_clip)                       # shared interest
        for sub in range(1, n_sub + 1):
            sid = f"{sub:03d}"
            r = 0.6 * clip_mean + 0.8 * rng.standard_normal(n_clip)    # mostly personal interest
            rz = (r - r.mean()) / r.std(ddof=1)
            blocks, ev = make_session(rng, n_blocks, vpb, clip_amp, rz, effect)
            d = deriv / f"sub-{sid}" / f"ses-{s}"
            (d / "parcels").mkdir(parents=True, exist_ok=True)
            ev.to_csv(d / "events.csv", index=False)
            for key, (Y, time) in blocks.items():
                da = xr.DataArray(Y, dims=("parcel", "chromo", "time"),
                                  coords=dict(parcel=PARCELS, chromo=["HbO", "HbR"], time=time),
                                  attrs={"units": "micromolar"})
                save_timeseries(da, d / "parcels" / f"{key}.nc")
            rows.append(dict(subject=sid, session=s, include=1))
            for i in range(n_clip):
                ratings.append(dict(session_tag=f"sub-{sid}_ses-{s}", subject=sid, session=s,
                                    video_global=i + 1, video_uid=(s - 1) * n_clip + i + 1,
                                    rating=float(r[i]), rating_z_session=float(rz[i])))
    pd.DataFrame(rows).to_csv(out / "sessions.csv", index=False)
    R = pd.DataFrame(ratings)
    R["rating_normative"] = [R[(R.video_uid == u) & (R.subject != s)].rating_z_session.mean()
                             for u, s in zip(R.video_uid, R.subject)]
    (deriv / "group" / "ratings").mkdir(parents=True, exist_ok=True)
    R.to_csv(deriv / "group" / "ratings" / "ratings_all.csv", index=False)

    cfg = yaml.safe_load(open(ROOT / "config" / "config.example.yaml"))
    cfg["paths"].update(raw_root=str(out / "raw"), derivatives="derivatives", sessions_csv="sessions.csv")
    cfg["paradigm"].update(n_video_blocks=n_blocks, n_rest_blocks=n_blocks, videos_per_block=vpb)
    cfg["stats"].update(min_subjects=5, min_parcels_per_network=2)
    cfg["decoding"]["cross_subject_sets"] = ["B_mean_HbO", "B_amplitude"]
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out / "config.yaml"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--effect", type=float, default=0.3, help="planted interest gain in Vis HbO")
    a = ap.parse_args()
    print(f"-> {write_study(Path(a.out), n_sub=a.subjects, effect=a.effect)}")
