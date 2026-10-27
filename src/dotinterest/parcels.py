"""Schaefer/Yeo-17 parcel labels.

Labels look like `ContA_IPS_2_LH` or `SomMotA_10_RH`: network first, hemisphere last. The
medial-wall label (`Background+FreeSurfer_Defined_Medial_Wall`) is not cortical.
"""
from __future__ import annotations

YEO17_TO_7 = {
    "VisCent": "Vis", "VisPeri": "Vis", "SomMotA": "SomMot", "SomMotB": "SomMot",
    "DorsAttnA": "DorsAttn", "DorsAttnB": "DorsAttn",
    "SalVentAttnA": "SalVentAttn", "SalVentAttnB": "SalVentAttn",
    "LimbicA": "Limbic", "LimbicB": "Limbic", "ContA": "Cont", "ContB": "Cont", "ContC": "Cont",
    "DefaultA": "Default", "DefaultB": "Default", "DefaultC": "Default", "TempPar": "TempPar",
}


def is_cortical(label) -> bool:
    return not any(s in str(label) for s in ("Background", "Medial_Wall", "Unknown"))


def network17(label) -> str:
    return str(label).split("_")[0]


def network7(label) -> str:
    return YEO17_TO_7.get(network17(label), network17(label))


def hemisphere(label) -> str:
    return str(label).split("_")[-1]


def network_hemi(label) -> str:
    """Grouping unit used for network-level features: Yeo-17 network x hemisphere."""
    return f"{network17(label)}_{hemisphere(label)}"
