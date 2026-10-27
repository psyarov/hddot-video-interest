"""Step 4 — per session: single-trial GLM, one fit per video clip and per rest block.

Each trial window (onset - pad .. offset + pad) is cut from its block and fitted with a
Gaussian-kernel HRF basis + drift (OLS). Feature: area under the fitted response (AUC).
Trials that cannot be fitted (aborted clip, window outside its block, window shorter than the
basis) are skipped and listed with the reason.

Output (<derivatives>/sub-<sub>/ses-<ses>/):
    glm_auc.csv.gz    trial, parcel, chromo, auc, se, t   (se/t: OLS, diagnostics only)
    glm_trials.csv    one row per trial: status and reason

    python scripts/04_glm.py --config config/config.yaml [--only SUB:SES | --index N]
"""
import argparse
import os

import pandas as pd

from dotinterest.config import add_session_args, load_config, pick_sessions, session_dir
from dotinterest.glm import check_trial, fit_trial, make_basis, trial_window
from dotinterest.io import load_timeseries
from dotinterest.paradigm import trial_table


def run(cfg, sub, ses, n_jobs):
    d = session_dir(cfg, sub, ses)
    p = cfg["glm"]
    basis = make_basis(p)
    trials = trial_table(pd.read_csv(d / "events.csv"), cfg["paradigm"])
    blocks = {f.stem: f for f in (d / "parcels").glob("*.nc") if not f.stem.startswith("global_")}
    loaded, rows, status = {}, [], []
    for r in trials.itertuples():
        key = f"{r.kind}_{r.block}"
        if key not in blocks:
            status.append(dict(trial=r.trial, status="skipped", reason="block missing"))
            continue
        if key not in loaded:
            loaded = {key: load_timeseries(blocks[key], quantify=True)}     # one block in memory
        y = loaded[key].sel(time=slice(*trial_window(r, p)))
        reason = check_trial(r, y, p)
        if reason:
            status.append(dict(trial=r.trial, status="skipped", reason=reason))
            print(f"  {r.trial}: skipped ({reason})")
            continue
        auc, se, t = fit_trial(y, r, basis, p["drift_order"], p["noise_model"], n_jobs)
        df = auc.rename("auc").to_dataframe().join(se.rename("se").to_dataframe()) \
                .join(t.rename("t").to_dataframe()).reset_index()
        df.insert(0, "trial", r.trial)
        rows.append(df[["trial", "parcel", "chromo", "auc", "se", "t"]])
        status.append(dict(trial=r.trial, status="fitted", reason=""))
    pd.concat(rows, ignore_index=True).to_csv(d / "glm_auc.csv.gz", index=False, float_format="%.6g")
    st = pd.DataFrame(status)
    st.to_csv(d / "glm_trials.csv", index=False)
    print(f"  {int((st.status == 'fitted').sum())}/{len(st)} trials fitted")


def main():
    a = add_session_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    cfg = load_config(a.config)
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    for sub, ses in pick_sessions(cfg, a.only, a.index):
        print(f"=== sub-{sub} ses-{ses}")
        if (session_dir(cfg, sub, ses) / "glm_auc.csv.gz").exists() and not a.overwrite:
            print("  done already (use --overwrite)")
            continue
        try:
            run(cfg, sub, ses, n_jobs)
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}")
            if a.only or a.index is not None:
                raise


if __name__ == "__main__":
    main()
