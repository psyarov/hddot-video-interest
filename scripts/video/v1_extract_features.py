"""Video step 1 — stimulus features from every clip (stimuli only; no participant data).

Clips are expected as <videos>/session_<s>/block_<b>/video_<n>.mp4, where n is the position in
the block, so video_uid = (s - 1) * videos_per_session + (b - 1) * videos_per_block + n.
Needs ffmpeg on the PATH and the `video` environment (environment-video.yml). Model weights in
`paths.video_models`: face_detection_yunet_2023mar.onnx (OpenCV Zoo) and yolov8n.pt (Ultralytics).

Output (<derivatives>/group/video_features/):
    features/clip_<uid>.npz   all features with their own time axes (existing files are skipped)
    clips.csv                 video_uid, session, block, position, duration_s, fps

    python scripts/video/v1_extract_features.py --config config/config.yaml [--workers 3] [--only 1,2]
"""
import argparse
import re
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dotinterest.config import resolve_path, group_dir, load_config
from dotinterest.video_features import audio_features, duration_of, fps_of, video_features


def clip_table(videos: Path, vpb: int, vps: int) -> pd.DataFrame:
    rows = []
    for p in sorted(videos.glob("session_*/block_*/video_*.mp4")):
        if p.name.startswith("._"):
            continue
        s = int(re.search(r"session_(\d+)", str(p)).group(1))
        b = int(re.search(r"block_(\d+)", str(p)).group(1))
        n = int(re.search(r"video_(\d+)\.mp4", p.name).group(1))
        rows.append(dict(video_uid=(s - 1) * vps + (b - 1) * vpb + n, session=s, block=b, position=n, path=str(p)))
    return pd.DataFrame(rows).sort_values("video_uid", ignore_index=True)


def process(job):
    row, out, models = job
    dst = Path(out) / "features" / f"clip_{row['video_uid']:03d}.npz"
    if dst.exists():
        return row["video_uid"], "skip", 0.0
    t0 = time.time()
    try:
        feats = {**video_features(row["path"], models), **audio_features(row["path"]),
                 "duration": duration_of(row["path"])}
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **feats)
        tmp.rename(dst)
        return row["video_uid"], "ok", time.time() - t0
    except Exception:
        return row["video_uid"], "error: " + traceback.format_exc().splitlines()[-1], time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", help="comma-separated video_uids")
    a = ap.parse_args()
    cfg = load_config(a.config)
    par = cfg["paradigm"]
    out = group_dir(cfg, "video_features")
    (out / "features").mkdir(exist_ok=True)

    clips = clip_table(resolve_path(cfg, cfg["paths"]["videos"]), par["videos_per_block"],
                       par["videos_per_block"] * par["n_video_blocks"])
    clips["duration_s"] = [duration_of(p) for p in clips.path]
    clips["fps"] = [fps_of(p) for p in clips.path]
    clips.drop(columns="path").to_csv(out / "clips.csv", index=False)
    print(f"{len(clips)} clips; frame rates: {clips.fps.round(2).value_counts().to_dict()}")

    todo = clips if not a.only else clips[clips.video_uid.isin([int(x) for x in a.only.split(",")])]
    models = str(resolve_path(cfg, cfg["paths"]["video_models"]))
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (uid, status, sec) in enumerate(pool.imap_unordered(
                process, [(r, str(out), models) for r in todo.to_dict("records")]), 1):
            print(f"[{k}/{len(todo)}] clip {uid:03d}: {status} ({sec:.0f} s, total {(time.time() - t0) / 60:.1f} min)",
                  flush=True)


if __name__ == "__main__":
    main()
