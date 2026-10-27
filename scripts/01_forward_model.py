"""Step 1 — head model, montage registration and sensitivity matrix (once per montage).

All sessions share one montage and the atlas head model, so this runs once, on any session's
SNIRF file (it only reads the probe geometry and measurement list). Needs NIRFASTer (default)
or MCX; a GPU helps. Takes hours.

Output: <derivatives>/forward_model/forward_model.pkl with
    sensitivity     Adot (channel x vertex x wavelength)
    fwm             cedalion ForwardModel (used for the per-session parcel sensitivity mask)
    montage_ras     montage registered to the head model (replaces rec.geo3d downstream)
    brain_vertices  brain vertex table with parcel labels

    python scripts/01_forward_model.py --config config/config.yaml [--snirf path/to/any.snirf]
"""
import argparse
import pickle

from dotinterest.config import deriv_root, forward_model_path, load_config, raw_path, sessions
from dotinterest.headmodel import build_head_and_montage, compute_sensitivity
from dotinterest.io import load_snirf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--snirf", help="SNIRF file to take the montage from (default: first session)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = cfg["forward_model"]

    snirf = a.snirf or raw_path(cfg, "snirf", *sessions(cfg)[0])
    print(f"montage from {snirf}")
    rec = load_snirf(snirf)
    head_ijk, head_ras, montage_ijk, montage_ras = build_head_and_montage(rec.geo3d, rec["amp"], p["head_model"])

    out_dir = deriv_root(cfg) / "forward_model"
    sensitivity, fwm = compute_sensitivity(head_ijk, montage_ijk, rec._measurement_lists["amp"],
                                           out_dir, p["fluence_method"])
    with open(forward_model_path(cfg), "wb") as f:
        pickle.dump(dict(sensitivity=sensitivity, fwm=fwm, montage_ras=montage_ras,
                         brain_vertices=head_ras.brain.vertices), f)
    print(f"-> {forward_model_path(cfg)}")


if __name__ == "__main__":
    main()
