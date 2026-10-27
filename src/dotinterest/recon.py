"""Image reconstruction and parcel-space cleaning.

Two-compartment (scalp + brain) Tikhonov reconstruction with the measurement-noise model,
then brain vertices -> Schaefer parcels sensitive to the montage -> parcel means ->
a second global-component regression in parcel space.
"""
from __future__ import annotations

import cedalion.dot as dot
import numpy as np
import xarray as xr
from cedalion.sigproc import physio


def sensitive_parcels(fwm, sensitivity, drop_list, dod_thresh: float) -> list[str]:
    """Parcels whose expected OD change exceeds `dod_thresh` given the retained channels."""
    _, mask = fwm.parcel_sensitivity(Adot=sensitivity, chan_droplist=list(drop_list),
                                     dOD_thresh=dod_thresh)
    return [str(p) for p in mask.where(mask, drop=True).parcel.values]


def estimate_alpha_meas(meas_var_od: dict, blocks: list[str], K: float) -> float:
    """Measurement regularisation, averaged over blocks (cedalion's estimate_alpha_meas)."""
    return float(np.mean([dot.estimate_alpha_meas(meas_var_od[b], K) for b in blocks]))


def make_recon(sensitivity, p: dict, alpha_meas: float):
    return dot.ImageRecon(sensitivity, brain_only=False, recon_mode=p["mode"],
                          spatial_basis_functions=None, alpha_meas=alpha_meas,
                          alpha_spatial=p["alpha_spatial"], apply_c_meas=p["apply_c_meas"])


def to_parcels(recon_result, brain_vertices, parcels: list[str]) -> xr.DataArray:
    """Brain vertices of a reconstruction -> mean over the vertices of each sensitive parcel."""
    y = recon_result.where(recon_result.is_brain, drop=True)
    if y.sizes["vertex"] != len(brain_vertices):
        raise ValueError(f"reconstruction has {y.sizes['vertex']} brain vertices, head model "
                         f"{len(brain_vertices)}: parcel labels would be misassigned")
    y = y.assign_coords(vertex=brain_vertices.label.values)
    y = y.where(y.parcel.isin(parcels), drop=True)
    y = y.groupby("parcel").mean(dim="vertex")
    return y.assign_coords(time=y.time.assign_attrs(units="s"))


def parcel_global_removal(y: xr.DataArray, exclude_prefixes=("Limbic",)):
    """Regress out the inverse-variance-weighted mean over parcels (per chromophore).

    Parcels starting with `exclude_prefixes` get zero weight in the global estimate (few,
    poorly covered parcels) but are still cleaned. -> (clean, global component)
    """
    v = (y.pint.dequantify() if hasattr(y, "pint") else y).var("time")
    w = (1.0 / v).where(np.isfinite(1.0 / v), 0.0)
    excl = xr.DataArray([str(p).startswith(tuple(exclude_prefixes)) for p in w.parcel.values],
                        dims="parcel", coords={"parcel": w.parcel})
    w = w.where(~excl, 0.0)
    clean, g = physio.global_component_subtract(ts=y, ts_weights=w, k=0,
                                                spatial_dim="parcel", spectral_dim="chromo")
    return clean, g
