"""Atlas head model, montage registration and the forward model (sensitivity matrix)."""
from __future__ import annotations

from pathlib import Path

import cedalion
import cedalion.dot as dot
import cedalion.io

LANDMARK_NAMES = {"nz": "Nz", "lpa": "LPA", "rpa": "RPA", "iz": "Iz", "cz": "Cz"}


def build_head_and_montage(montage, ts, head_model: str = "icbm152"):
    """Load an atlas two-surface head model and register the montage to its scalp.

    The montage is aligned on shared landmarks and then spring-relaxed onto the scalp,
    which preserves source-detector distances.
    -> head_ijk, head_ras, montage_ijk, montage_ras
    """
    montage = montage.assign_coords(label=[LANDMARK_NAMES.get(str(k).lower(), str(k))
                                           for k in montage.label.values])
    head_ijk = dot.get_standard_headmodel(head_model)
    head_ras = head_ijk.apply_transform(head_ijk.t_ijk2ras)

    n_common = len(montage.points.common_labels(head_ras.landmarks))
    mode = "general" if n_common > 3 else "trans_rot_isoscale" if n_common == 3 else "identity"
    montage_ras, details = head_ras.align_and_relax_to_scalp(montage, ts, initial_align_mode=mode)
    if not details.converged:
        print("  WARNING: montage relaxation did not converge; check the landmarks")
    montage_ijk = montage_ras.points.apply_transform(head_ijk.t_ras2ijk)
    return head_ijk, head_ras, montage_ijk, montage_ras


def compute_sensitivity(head_ijk, montage_ijk, meas_list, out_dir: Path, method: str = "nirfaster"):
    """Photon fluence (NIRFASTer or MCX) -> sensitivity matrix. -> (Adot, ForwardModel)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fluence = out_dir / f"fluence_{method}.h5"
    sens = out_dir / f"sensitivity_{method}.h5"
    names = {"skin": "scalp", "gray": "gm", "white": "wm", "csf": "csf", "skull": "skull"}
    seg = head_ijk.segmentation_masks
    head_ijk.segmentation_masks = seg.assign_coords(
        segmentation_type=[names.get(s, s) for s in seg.segmentation_type.values])
    fwm = dot.ForwardModel(head_ijk, montage_ijk, meas_list)
    if method == "nirfaster":
        fwm.compute_fluence_nirfaster(fluence_fname=fluence)
    elif method == "mcx":
        fwm.compute_fluence_mcx(fluence_fname=fluence)
    else:
        raise ValueError(f"unknown fluence method {method!r}")
    fwm.compute_sensitivity(fluence_fname=fluence, sensitivity_fname=sens)
    return cedalion.io.forward_model.load_Adot(sens), fwm
