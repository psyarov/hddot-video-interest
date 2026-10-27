"""Configuration, session lists and the derivatives folder layout."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = path.parent
    return cfg


def resolve_path(cfg: dict, p: str | Path) -> Path:
    p = Path(os.path.expandvars(str(p))).expanduser()
    return p if p.is_absolute() else (cfg["_config_dir"] / p).resolve()


def raw_path(cfg: dict, key: str, sub: str, ses: str | int) -> Path:
    """Fill a raw-data path template from `paths:` (e.g. `snirf`, `xdf`, `ratings_csv`)."""
    tpl = cfg["paths"][key]
    return resolve_path(cfg, tpl.format(raw_root=cfg["paths"]["raw_root"], sub=sub, ses=ses))


def deriv_root(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["paths"]["derivatives"])


def session_dir(cfg: dict, sub: str, ses: str | int) -> Path:
    """Per-session derivatives: <derivatives>/sub-<sub>/ses-<ses>/"""
    d = deriv_root(cfg) / f"sub-{sub}" / f"ses-{ses}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def group_dir(cfg: dict, name: str) -> Path:
    """Group-level outputs: <derivatives>/group/<name>/"""
    d = deriv_root(cfg) / "group" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def forward_model_path(cfg: dict) -> Path:
    return deriv_root(cfg) / "forward_model" / "forward_model.pkl"


def session_tag(sub: str, ses: str | int) -> str:
    return f"sub-{sub}_ses-{ses}"


def sessions(cfg: dict) -> list[tuple[str, str]]:
    """(subject, session) pairs from the sessions CSV (columns: subject, session[, include])."""
    df = pd.read_csv(resolve_path(cfg, cfg["paths"]["sessions_csv"]),
                     dtype={"subject": str, "session": str}, comment="#")
    if "include" in df:
        df = df[df["include"].astype(str).str.lower().isin(["1", "true", "yes"])]
    return list(zip(df.subject.str.strip(), df.session.str.strip()))


def pick_sessions(cfg: dict, only: str | None = None, index: int | None = None):
    """All sessions, one given as 'SUB:SES', or the index-th (for SLURM job arrays)."""
    if only:
        sub, ses = only.split(":")
        return [(sub, ses)]
    todo = sessions(cfg)
    if index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    return [todo[index]] if index is not None else todo


def add_session_args(ap):
    """Arguments shared by every per-session script."""
    ap.add_argument("--config", required=True, help="YAML config (see config/config.example.yaml)")
    ap.add_argument("--only", help="run one session, e.g. 001:2")
    ap.add_argument("--index", type=int, help="run the N-th session of the list (job arrays)")
    ap.add_argument("--overwrite", action="store_true", help="redo sessions that already have output")
    return ap


if __name__ == "__main__":
    # `python -m dotinterest.config <config>` prints the last 0-based session index (for --array)
    import sys

    print(len(sessions(load_config(sys.argv[1]))) - 1)
