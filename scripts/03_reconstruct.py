"""Step 3 — per session: image reconstruction -> cleaned parcel time series.

For every block: two-compartment Tikhonov reconstruction (mua2conc, measurement-noise model)
-> brain vertices -> mean over each sensitive parcel -> parcel-space global-component
regression. Blocks are reconstructed one at a time to keep memory bounded (~16 GB peak at
567 channels; a GPU node speeds it up).

Output (<derivatives>/sub-<sub>/ses-<ses>/parcels/):
    <block>.nc          HbO/HbR per parcel (micromolar), block = video_1..3, rest_1..3
    global_<block>.nc   the removed parcel-space global component

    python scripts/03_reconstruct.py --config config/config.yaml [--only SUB:SES | --index N]
"""
import argparse
import gc
import json
import pickle

from dotinterest.config import add_session_args, forward_model_path, load_config, pick_sessions, session_dir
from dotinterest.io import save_timeseries
from dotinterest.recon import make_recon, parcel_global_removal, to_parcels


def run(cfg, sub, ses, fm):
    d = session_dir(cfg, sub, ses)
    with open(d / "preproc.pkl", "rb") as f:
        pre = pickle.load(f)
    out = d / "parcels"
    out.mkdir(exist_ok=True)
    recon = make_recon(fm["sensitivity"], cfg["recon"], pre["alpha_meas"])
    n_parcels = {}
    for b, od in pre["od_clean"].items():
        res = recon.reconstruct(od, c_meas=pre["meas_var_od"][b])
        y = to_parcels(res, fm["brain_vertices"], pre["sensitive_parcels"])
        del res
        gc.collect()
        clean, g = parcel_global_removal(y, cfg["recon"]["global_exclude"])
        save_timeseries(clean, out / f"{b}.nc")
        save_timeseries(g, out / f"global_{b}.nc")
        n_parcels[b] = int(clean.sizes["parcel"])
        print(f"  {b}: {n_parcels[b]} parcels, {clean.sizes['time']} samples")
    qc_path = d / "qc.json"
    qc = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    qc["reconstruct"] = {"n_parcels_per_block": n_parcels}
    qc_path.write_text(json.dumps(qc, indent=2, default=float))


def main():
    a = add_session_args(argparse.ArgumentParser(description=__doc__,
                                                 formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    cfg = load_config(a.config)
    with open(forward_model_path(cfg), "rb") as f:
        fm = pickle.load(f)
    for sub, ses in pick_sessions(cfg, a.only, a.index):
        print(f"=== sub-{sub} ses-{ses}")
        if (session_dir(cfg, sub, ses) / "parcels" / "rest_3.nc").exists() and not a.overwrite:
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
