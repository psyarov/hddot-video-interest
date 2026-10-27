"""Reading SNIRF/XDF, aligning event markers to the fNIRS clock, and netCDF storage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# --------------------------------------------------------------------------------------- #
# raw data
# --------------------------------------------------------------------------------------- #
def load_snirf(path: str | Path):
    """First recording of a SNIRF file, with aux channels squeezed and time units set."""
    import cedalion.io

    rec = cedalion.io.read_snirf(path)[0]
    aux = rec.aux_ts.copy()
    for k, v in aux.items():
        t = v.time.assign_attrs({"units": rec.meta_data.get("TimeUnit", "s")})
        aux[k] = v.assign_coords(time=t).squeeze()
    rec.aux_ts = aux
    return rec


def load_markers(xdf_path: str | Path, marker_stream: str, sync_stream: str):
    """-> (marker names, marker LSL times, hardware-sync LSL times) from an XDF file."""
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    by_name = {s["info"]["name"][0]: s for s in streams}
    for name in (marker_stream, sync_stream):
        if name not in by_name:
            raise KeyError(f"stream '{name}' not in {xdf_path.name}; found {sorted(by_name)}")
    m = by_name[marker_stream]
    names = [x[0].decode() if isinstance(x[0], bytes) else str(x[0]) for x in m["time_series"]]
    return names, np.asarray(m["time_stamps"], float), np.asarray(by_name[sync_stream]["time_stamps"], float)


def nirs_sync_pulses(rec, aux_name: str = "digital") -> np.ndarray:
    """Rising edges of the digital sync input recorded by the fNIRS device (device clock)."""
    d = rec.aux_ts[aux_name]
    x, t = np.asarray(d.values).squeeze(), np.asarray(d.time.values, float)
    rise = np.where((x[:-1] == 0) & (x[1:] != 0))[0]
    return t[rise + 1]


def _dedouble(t: np.ndarray, thresh: float = 2.0) -> np.ndarray:
    """The device logs each sync pulse twice; keep the first of every close pair."""
    t = np.sort(np.asarray(t, float))
    keep = [t[0]]
    for x in t[1:]:
        if x - keep[-1] > thresh:
            keep.append(x)
    return np.array(keep)


def _align_pulses(nirs: np.ndarray, lsl: np.ndarray, max_mismatch: float = 0.05):
    """Match pulse trains by their inter-pulse intervals (robust to extra pulses at either end)."""
    best = None
    for shift in range(-(len(lsl) - 1), len(nirs)):
        pairs = [(i, i + shift) for i in range(len(nirs)) if 0 <= i + shift < len(lsl)]
        if len(pairs) < 2:
            continue
        ni, li = zip(*pairs)
        err = np.mean(np.abs(np.diff(nirs[list(ni)]) - np.diff(lsl[list(li)])))
        if best is None or err < best[0]:
            best = (err, list(ni), list(li))
    if best is None or best[0] > max_mismatch:
        raise ValueError(f"sync pulse alignment failed (interval mismatch "
                         f"{best[0] if best else np.inf:.3f} s); inspect the pulses")
    return nirs[best[1]], lsl[best[2]], best[0]


def events_from_markers(names, marker_t, sync_lsl, sync_nirs, dedouble_s: float = 2.0):
    """Map LSL marker times onto the fNIRS clock with a linear fit through matched sync pulses.

    -> (events DataFrame with onset/duration/value/trial_type, dict of sync diagnostics).
    A marker's duration is the time to the next marker.
    """
    nm, lm, err = _align_pulses(_dedouble(sync_nirs, dedouble_s), np.asarray(sync_lsl, float))
    if len(nm) >= 2:
        a, b = np.polyfit(lm, nm, 1)
    else:
        a, b = 1.0, nm[0] - lm[0]
    onsets = a * np.asarray(marker_t, float) + b
    order = np.argsort(onsets)
    onsets = onsets[order]
    names = [names[i] for i in order]
    ev = pd.DataFrame({"onset": onsets, "duration": list(np.diff(onsets)) + [0.0],
                       "value": 1.0, "trial_type": names})
    resid = nm - (a * lm + b)
    info = dict(sync_n_matched=len(nm), sync_interval_mismatch_ms=1e3 * err,
                sync_drift_ppm=(a - 1) * 1e6, sync_resid_rms_ms=1e3 * float(np.sqrt(np.mean(resid ** 2))))
    return ev, info


# --------------------------------------------------------------------------------------- #
# derivatives: parcel time series as netCDF
# --------------------------------------------------------------------------------------- #
def save_timeseries(da: xr.DataArray, path: str | Path) -> None:
    """Store a (possibly pint-quantified) DataArray as float32 netCDF, units in attrs."""
    try:
        units = str(da.pint.units)
        da = da.pint.dequantify()
    except Exception:
        units = str(da.attrs.get("units", ""))
    da = da.copy()
    da.attrs = {"units": units}
    for name, idx in list(da.indexes.items()):          # MultiIndex coords can't go to netCDF
        if isinstance(idx, pd.MultiIndex):
            da = da.reset_index(name)
    for c in list(da.coords):
        da[c].attrs = {}                                # e.g. time units would be decoded as timedelta
        if da[c].dtype == object:
            da = da.assign_coords({c: da[c].astype(str)})
    da.name = "data"
    da.astype("float32").to_dataset().to_netcdf(path, engine="h5netcdf",
                                                encoding={"data": {"zlib": True, "complevel": 4}})


def load_timeseries(path: str | Path, quantify: bool = False) -> xr.DataArray:
    """Inverse of `save_timeseries`. `quantify=True` re-attaches pint units (needed by cedalion)."""
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False, decode_timedelta=False) as ds:
        da = ds["data"].load()
    if quantify and da.attrs.get("units"):
        da = da.pint.quantify(da.attrs["units"])
        da = da.assign_coords(time=da.time.assign_attrs(units="s"))
        if "samples" not in da.coords:                  # cedalion's time-series schema needs it
            da = da.assign_coords(samples=("time", np.arange(da.sizes["time"])))
    return da
