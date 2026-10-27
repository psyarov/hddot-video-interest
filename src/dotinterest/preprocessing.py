"""Channel-space preprocessing of one session (continuous-wave HD-DOT, two wavelengths).

    amplitude -> trim after 'end' -> repair -> OD -> TDDR + wavelet motion correction
    -> high-pass -> channel pruning (mean amplitude + SNR on the middle of rest block 1)
    -> measurement variance per block (high-passed OD; used by the image reconstruction)
    -> MBLL -> low-pass -> blocks (with buffers) -> global-component regression per block
    -> back to OD for the reconstruction.
"""
from __future__ import annotations

import cedalion
import cedalion.nirs
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils
import numpy as np
import xarray as xr
from cedalion import units
from cedalion.sigproc import physio

from .paradigm import block_limits, end_of_session


def _dpf(od, value):
    return xr.DataArray([value] * od.sizes["wavelength"], dims="wavelength",
                        coords={"wavelength": od.wavelength.values})


def od2conc(od, geo3d, dpf=6.0):
    return cedalion.nirs.cw.od2conc(od, geo3d=geo3d, dpf=_dpf(od, dpf))


def conc2od(conc, geo3d, wavelengths, dpf=6.0):
    d = xr.DataArray([dpf] * len(wavelengths), dims="wavelength", coords={"wavelength": wavelengths})
    return cedalion.nirs.cw.conc2od(conc, geo3d=geo3d, dpf=d)


def prune_channels(amp, od_hp, window, mean_amp_range, snr_thresh):
    """Channels failing the mean-amplitude range or the SNR threshold on `window` (t0, t1).

    Both metrics are computed on amplitude rebuilt from the high-passed OD, so slow drifts do
    not inflate the noise estimate. -> (drop list, dict of per-criterion lists)
    """
    t0, t1 = window
    baseline = amp.sel(time=slice(t0, t1)).mean("time")
    y = cedalion.nirs.cw.od2int(od_hp.sel(time=slice(t0, t1)), baseline)
    y = y.pint.dequantify().pint.quantify("volt")
    _, amp_mask = quality.mean_amp(y, list(mean_amp_range) * units.volt)
    _, drop_amp = xrutils.apply_mask(y, amp_mask, "drop", "channel")
    _, snr_mask = quality.snr(y, snr_thresh)
    _, drop_snr = xrutils.apply_mask(y, snr_mask, "drop", "channel")
    drop = [str(c) for c in np.union1d(drop_amp, drop_snr)]
    return drop, {"mean_amp": [str(c) for c in drop_amp], "snr": [str(c) for c in drop_snr]}


def preprocess_session(rec, events, cfg: dict) -> dict:
    """Run the channel-space pipeline. `rec.geo3d` must already hold the registered montage.

    Returns a dict with, per block key ('video_1', ..., 'rest_3'):
        od_clean[block]      OD after global-component removal (input to the reconstruction)
        meas_var_od[block]   measurement variance of the high-passed OD (reconstruction noise model)
        global_comp[block]   the removed global component (HbO/HbR)
    plus `drop_list`, `block_limits` and a `qc` dict of scalar diagnostics.
    """
    p = cfg["preprocessing"]
    lims = block_limits(events, cfg["paradigm"])

    amp = rec["amp"]
    t_end = end_of_session(events, float(amp.time.values[-1]))
    amp = quality.repair_amp(amp.sel(time=amp.time < t_end))

    od = cedalion.nirs.cw.int2od(amp)
    od = od.assign_coords(time=od.time.assign_attrs(units="s"))
    od = motion.motion_correct_wavelet(motion.tddr(od))
    od_hp = od.cd.freq_filter(fmin=p["fmin"], fmax=0.0, butter_order=p["butter_order"])

    # channel quality on rest block 1, minus `quality_trim_s` at each end
    r0, r1 = lims["rest_1"]
    window = (r0 + p["quality_trim_s"], r1 - p["quality_trim_s"])
    drop, drop_by = prune_channels(amp, od_hp, window, p["mean_amp_range_v"], p["snr_thresh"])

    def block_var(ts):
        return {b: quality.measurement_variance(ts.sel(time=slice(*lim)).pint.dequantify(),
                                                list_bad_channels=drop, bad_rel_var=p["bad_rel_var"],
                                                calc_covariance=False)
                for b, lim in lims.items()}

    meas_var_od = block_var(od_hp)
    meas_var_conc = block_var(od2conc(od_hp, rec.geo3d, p["dpf"]))

    od_bp = od_hp.cd.freq_filter(fmin=0.0, fmax=p["fmax"], butter_order=p["butter_order"])
    conc = od2conc(od_bp, rec.geo3d, p["dpf"])
    wl = od.wavelength.values

    od_clean, global_comp, gc_var = {}, {}, {}
    for b, (t0, t1) in lims.items():
        buf = p["buffer_s"][b.split("_")[0]]
        y = conc.sel(time=slice(t0 - buf, t1 + buf))
        w = 1.0 / meas_var_conc[b]                      # inverse-variance weights: bad channels ~0
        y_clean, g = physio.global_component_subtract(ts=y, ts_weights=w, k=0)
        od_clean[b] = conc2od(y_clean, rec.geo3d, wl, p["dpf"])
        global_comp[b] = g
        v0 = y.pint.dequantify().var("time")
        v1 = y_clean.pint.dequantify().var("time")
        gc_var[b] = float((1 - v1 / v0).sel(channel=[c for c in y.channel.values
                                                     if str(c) not in set(drop)]).median())

    n_ch = int(amp.sizes["channel"])
    qc = dict(n_channels=n_ch, n_dropped=len(drop), pct_dropped=100 * len(drop) / n_ch,
              n_dropped_mean_amp=len(drop_by["mean_amp"]), n_dropped_snr=len(drop_by["snr"]),
              duration_min=float(amp.time.values[-1] - amp.time.values[0]) / 60,
              n_blocks=len(lims),
              gc_var_explained_median=float(np.median(list(gc_var.values()))) if gc_var else np.nan)
    return dict(od_clean=od_clean, meas_var_od=meas_var_od, global_comp=global_comp,
                drop_list=drop, block_limits=lims, qc=qc)
