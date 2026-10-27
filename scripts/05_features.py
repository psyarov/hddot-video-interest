"""Step 5 — all sessions: model-free features + GLM AUC -> long and decoding-ready tables.

Per session, model-free features are computed from the cleaned parcel time series (see
dotinterest/features.py) and joined with the session's GLM AUC on (trial, parcel, chromo).
Ratings are attached to video trials on (session_tag, video_global).

Output (<derivatives>/group/features/):
    parcel_features.csv.gz    one row per trial x parcel x chromo, every feature family + glm_auc
    fc_features.csv.gz        one row per trial x network edge x chromo
    network_features.csv.gz   network x hemisphere aggregates (<family>_mean, <family>_sum, n_parcels)
    trials.csv                one row per trial: keys, video_uid, ratings
    ml_table_network.csv      one row per trial, network-level features as columns (decoding)
    sessions_qc.csv           per session: preprocessing/recon QC, trials kept, parcels

    python scripts/05_features.py --config config/config.yaml
"""
import argparse
import json

import numpy as np
import pandas as pd

from dotinterest.config import group_dir, load_config, session_dir, session_tag, sessions
from dotinterest.features import model_free_session, network_table, wide_table
from dotinterest.io import load_timeseries
from dotinterest.paradigm import trial_table, video_uid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    p, par = cfg["features"], cfg["paradigm"]
    n_per_ses = par["n_video_blocks"] * par["videos_per_block"]
    out = group_dir(cfg, "features")

    parcel_rows, fc_rows, qc_rows = [], [], []
    for sub, ses in sessions(cfg):
        d, tag = session_dir(cfg, sub, ses), session_tag(sub, ses)
        if not (d / "parcels").is_dir():
            print(f"  {tag}: no parcel time series, skipped")
            continue
        trials = trial_table(pd.read_csv(d / "events.csv"), par)
        blocks = {f.stem: load_timeseries(f) for f in sorted((d / "parcels").glob("*.nc"))
                  if not f.stem.startswith("global_")}
        try:
            long, fc, status = model_free_session(blocks, trials, p)
        except Exception as e:
            print(f"  {tag}: FAILED {type(e).__name__}: {e}")
            continue
        n_vid = int(status.trial[status.status.isin(["ok", "partial"])].str.startswith("video").sum())
        qc = json.loads((d / "qc.json").read_text()) if (d / "qc.json").exists() else {}
        qc_rows.append({"session_tag": tag, "subject": sub, "session": int(ses),
                        **qc.get("preprocess", {}), "n_video_trials": n_vid,
                        "n_parcels_used": long.parcel.nunique(),
                        "included": n_vid >= p["min_video_trials"]})
        if n_vid < p["min_video_trials"]:
            print(f"  {tag}: only {n_vid} video trials, left out")
            continue
        if (d / "glm_auc.csv.gz").exists():
            g = pd.read_csv(d / "glm_auc.csv.gz", usecols=["trial", "parcel", "chromo", "auc"])
            long = long.merge(g.rename(columns={"auc": "glm_auc"}), on=["trial", "parcel", "chromo"], how="left")
        for df in (long, fc):
            df.insert(0, "session_tag", tag)
            df.insert(1, "subject", sub)
            df.insert(2, "session", int(ses))
        parcel_rows.append(long)
        fc_rows.append(fc)
        print(f"  {tag}: {long.trial.nunique()} trials, {long.parcel.nunique()} parcels")

    P = pd.concat(parcel_rows, ignore_index=True)
    FC = pd.concat(fc_rows, ignore_index=True)
    families = [c for c in P.columns if c not in ("session_tag", "subject", "session", "trial", "parcel", "chromo")]
    N = network_table(P, families)

    T = P[["session_tag", "subject", "session", "trial"]].drop_duplicates().reset_index(drop=True)
    T["is_video"] = T.trial.str.startswith("video")
    T["video_global"] = T.trial.str.extract(r"video_(\d+)")[0].astype("Float64").astype("Int64")
    T["video_uid"] = [video_uid(s, v, n_per_ses) if pd.notna(v) else pd.NA
                      for s, v in zip(T.session, T.video_global)]
    rpath = group_dir(cfg, "ratings") / "ratings_all.csv"
    if rpath.exists():
        R = pd.read_csv(rpath, dtype={"subject": str})
        keep = ["session_tag", "video_global", "rating", "rating_z_session", "rating_z_subject",
                "rating_normative"]
        T = T.merge(R[[c for c in keep if c in R.columns]], on=["session_tag", "video_global"], how="left")
        print(f"ratings joined to {int(T.rating.notna().sum())}/{int(T.is_video.sum())} video trials")
    else:
        print("!! no ratings_all.csv (run step 0); tables written without labels")

    ff = "%.6g"
    P.to_csv(out / "parcel_features.csv.gz", index=False, float_format=ff)
    FC.to_csv(out / "fc_features.csv.gz", index=False, float_format=ff)
    N.to_csv(out / "network_features.csv.gz", index=False, float_format=ff)
    T.to_csv(out / "trials.csv", index=False)
    vals = [c for c in N.columns if c.endswith(("_mean", "_sum"))]
    W = wide_table(N.merge(T[["session_tag", "trial", "subject", "session"]], on=["session_tag", "trial"]),
                   vals, "network")
    T.merge(W, on=["session_tag", "subject", "session", "trial"], how="right") \
     .to_csv(out / "ml_table_network.csv", index=False, float_format=ff)
    pd.DataFrame(qc_rows).to_csv(out / "sessions_qc.csv", index=False)
    print(f"\n{T.session_tag.nunique()} sessions, {len(T)} trials, {len(families)} families -> {out}")


if __name__ == "__main__":
    main()
