"""Step 2 — per session: events, channel-space preprocessing, parcel mask, reconstruction inputs.

    events      LSL markers mapped to the fNIRS clock through the hardware sync pulses
                (or `events_override` if that file exists)
    signal      see dotinterest/preprocessing.py
    parcels     Schaefer parcels sensitive to the retained channels
    alpha_meas  measurement regularisation for this session, from the noise model

Output (<derivatives>/sub-<sub>/ses-<ses>/):
    events.csv, parcel_sensitivity.csv, qc.json
    preproc.pkl     cleaned OD per block + noise model + recon parameters (input to step 3; large)

    python scripts/02_preprocess.py --config config/config.yaml [--only SUB:SES | --index N]
"""
import argparse
import json
import pickle

import pandas as pd

from dotinterest.config import (add_session_args, forward_model_path, load_config, pick_sessions,
                                raw_path, session_dir)
from dotinterest.io import events_from_markers, load_markers, load_snirf, nirs_sync_pulses
from dotinterest.parcels import hemisphere, is_cortical, network7, network17
from dotinterest.preprocessing import preprocess_session
from dotinterest.recon import estimate_alpha_meas, sensitive_parcels


def session_events(cfg, sub, ses, rec):
    override = raw_path(cfg, "events_override", sub, ses)
    if override.exists():
        print(f"  events from {override.name}")
        return pd.read_csv(override), {"events_source": "override"}
    lsl = cfg["lsl"]
    names, t_marker, t_sync = load_markers(raw_path(cfg, "xdf", sub, ses), lsl["marker_stream"], lsl["sync_stream"])
    ev, info = events_from_markers(names, t_marker, t_sync, nirs_sync_pulses(rec, lsl["nirs_sync_aux"]),
                                   lsl["dedouble_s"])
    print(f"  sync: {info['sync_n_matched']} pulses, drift {info['sync_drift_ppm']:+.1f} ppm, "
          f"residual {info['sync_resid_rms_ms']:.1f} ms")
    return ev, {"events_source": "xdf", **info}


def run(cfg, sub, ses, fm):
    out = session_dir(cfg, sub, ses)
    rec = load_snirf(raw_path(cfg, "snirf", sub, ses))
    events, qc = session_events(cfg, sub, ses, rec)
    events = events.sort_values("onset", ignore_index=True)
    events.to_csv(out / "events.csv", index=False)

    rec.geo3d = fm["montage_ras"]
    res = preprocess_session(rec, events, cfg)
    qc.update(res["qc"])
    print(f"  {qc['n_channels'] - qc['n_dropped']}/{qc['n_channels']} channels kept, "
          f"{len(res['block_limits'])} blocks")

    r = cfg["recon"]
    parcels = sensitive_parcels(fm["fwm"], fm["sensitivity"], res["drop_list"], r["dod_thresh"])
    all_p = sorted({str(x) for x in fm["brain_vertices"].parcel.values})
    pd.DataFrame({"parcel": all_p, "sensitive": [x in set(parcels) for x in all_p],
                  "is_cortical": [is_cortical(x) for x in all_p],
                  "network": [network17(x) for x in all_p], "network7": [network7(x) for x in all_p],
                  "hemi": [hemisphere(x) for x in all_p]}).to_csv(out / "parcel_sensitivity.csv", index=False)

    video_blocks = [b for b in res["block_limits"] if b.startswith("video")]
    alpha = estimate_alpha_meas(res["meas_var_od"], video_blocks, r["alpha_meas_K"])
    qc.update(n_parcels_sensitive=sum(is_cortical(x) for x in parcels), alpha_meas=alpha)
    print(f"  {qc['n_parcels_sensitive']} sensitive cortical parcels, alpha_meas = {alpha:.3g}")

    with open(out / "preproc.pkl", "wb") as f:
        pickle.dump(dict(od_clean=res["od_clean"], meas_var_od=res["meas_var_od"],
                         global_comp_channel=res["global_comp"], drop_list=res["drop_list"],
                         block_limits=res["block_limits"], sensitive_parcels=parcels, alpha_meas=alpha), f)
    (out / "qc.json").write_text(json.dumps({"preprocess": qc}, indent=2, default=float))


def main():
    a = add_session_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    cfg = load_config(a.config)
    with open(forward_model_path(cfg), "rb") as f:
        fm = pickle.load(f)
    for sub, ses in pick_sessions(cfg, a.only, a.index):
        print(f"=== sub-{sub} ses-{ses}")
        if (session_dir(cfg, sub, ses) / "preproc.pkl").exists() and not a.overwrite:
            print("  done already (use --overwrite)")
            continue
        try:
            run(cfg, sub, ses, fm)
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {e}")
            if a.only or a.index is not None:
                raise


if __name__ == "__main__":
    main()
