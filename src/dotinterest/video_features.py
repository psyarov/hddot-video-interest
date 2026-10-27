"""Stimulus features of the video clips, and their properties as GLM regressors.

Per clip, each feature on its own time axis (clip seconds):
    luminance   Rec.709 luma of gamma-encoded RGB, frame mean                      every frame
    contrast    RMS contrast (SD of the luma image)                                every frame
    flow        Farneback optical flow, mean magnitude; frames at cuts interpolated every frame
    cut_times   peaks of 1 - HSV-histogram correlation between frames (adaptive threshold)
    faces       YuNet face boxes, union area / frame area                          every 6th frame
    bodies      YOLOv8n person boxes, union area / frame area                      every 6th frame
    speech      Silero VAD speech probability                                      31.25 Hz
    loudness    10 log10 STFT power (25 ms window, 10 ms hop)                      100 Hz
    flux        spectral flux (positive log-magnitude increments)                  100 Hz
Always use each file's own frame rate.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import gammaln

W, H = 960, 540             # decode size for detection; luminance/flow on a half-size copy
DET_EVERY = 6
CUT_MIN, CUT_K = 0.12, 6.0  # cut = local peak > max(CUT_MIN, CUT_K x local median over ±1 s)
FACE_SCORE, BODY_CONF = 0.7, 0.4
SR = 16000


# --------------------------------------------------------------------------------------- #
# extraction (needs ffmpeg, opencv, torch, ultralytics, silero-vad)
# --------------------------------------------------------------------------------------- #
def ffprobe(path, entry):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entry,
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def fps_of(path) -> float:
    num, den = ffprobe(path, "stream=r_frame_rate").split("/")
    return float(num) / float(den)


def duration_of(path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def union_area(boxes, w, h, grid=(54, 96)) -> float:
    """Fraction of the frame covered by the union of (x, y, w, h) boxes."""
    m = np.zeros(grid, bool)
    gy, gx = grid[0] / h, grid[1] / w
    for x, y, bw, bh in boxes:
        x0, y0 = max(int(np.floor(x * gx)), 0), max(int(np.floor(y * gy)), 0)
        x1, y1 = min(int(np.ceil((x + bw) * gx)), grid[1]), min(int(np.ceil((y + bh) * gy)), grid[0])
        m[y0:y1, x0:x1] = True
    return float(m.mean())


def detect_cuts(cut_dist, fps) -> np.ndarray:
    from scipy.ndimage import median_filter

    thr = np.maximum(CUT_MIN, CUT_K * median_filter(cut_dist, size=int(2 * fps) + 1, mode="nearest"))
    peak = (cut_dist >= np.r_[0, cut_dist[:-1]]) & (cut_dist >= np.r_[cut_dist[1:], 0])
    idx = np.where(peak & (cut_dist > thr))[0]
    keep = []
    for k in idx[np.argsort(cut_dist[idx])[::-1]]:        # strongest first, 0.5 s apart
        if all(abs(k - j) > fps / 2 for j in keep):
            keep.append(k)
    return np.array(sorted(keep), int)


def video_features(path, models_dir) -> dict:
    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    face = cv2.FaceDetectorYN.create(str(Path(models_dir) / "face_detection_yunet_2023mar.onnx"), "",
                                     (W, H), score_threshold=FACE_SCORE)
    yolo = YOLO(str(Path(models_dir) / "yolov8n.pt"))
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={W}:{H}:flags=area",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    lum, con, flow, cut, faces, bodies = [], [], [], [], [], []
    prev_gray = prev_hist = None
    i = 0
    while len(buf := proc.stdout.read(W * H * 3)) == W * H * 3:
        rgb = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        small = cv2.resize(rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        y = small.astype(np.float32) / 255.0 @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        lum.append(y.mean())
        con.append(y.std())
        gray = (y * 255).astype(np.uint8)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_RGB2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_gray is None:
            flow.append(np.nan)
            cut.append(0.0)
        else:
            f = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow.append(float(np.hypot(f[..., 0], f[..., 1]).mean()))
            cut.append(1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        prev_gray, prev_hist = gray, hist
        if i % DET_EVERY == 0:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, det = face.detect(bgr)
            faces.append(union_area([] if det is None else det[:, :4], W, H))
            res = yolo.predict(bgr, imgsz=640, classes=[0], conf=BODY_CONF, verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            bodies.append(union_area([(a, b, c - a, d - b) for a, b, c, d in xyxy], W, H))
        i += 1
    proc.wait()
    flow[0] = flow[1] if len(flow) > 1 else 0.0
    fps = fps_of(path)
    cut, flow = np.asarray(cut), np.asarray(flow)
    peaks = detect_cuts(cut, fps)
    bad = np.zeros(len(flow), bool)
    for k in peaks:
        bad[k:k + 2] = True                               # Farneback spikes at cuts
    flow_nocut = flow.copy()
    if bad.any() and (~bad).any():
        flow_nocut[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], flow[~bad])
    return dict(t_frame=np.arange(i) / fps, luminance=np.asarray(lum), contrast=np.asarray(con),
                flow=flow, flow_nocut=flow_nocut, cut_dist=cut, cut_times=peaks / fps,
                t_det=np.arange(len(faces)) * DET_EVERY / fps, faces=np.asarray(faces),
                bodies=np.asarray(bodies), fps=fps, n_frames=i)


def audio_features(path) -> dict:
    import torch
    from scipy.signal import stft
    from silero_vad import load_silero_vad

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    vad = load_silero_vad()
    vad.reset_states()
    n = 512
    probs = [float(vad(torch.from_numpy(x[k:k + n].copy()), SR)) for k in range(0, len(x) - n + 1, n)]
    _, t, Z = stft(x, fs=SR, nperseg=400, noverlap=240, boundary=None, padded=False)
    logmag = np.log(np.abs(Z) + 1e-6)
    return dict(t_vad=(np.arange(len(probs)) * n + n / 2) / SR, speech=np.asarray(probs), t_stft=t,
                loudness=10 * np.log10((np.abs(Z) ** 2).sum(0) + 1e-12),
                flux=np.r_[0.0, np.maximum(np.diff(logmag, axis=1), 0).sum(0)], audio_dur=len(x) / SR)


# --------------------------------------------------------------------------------------- #
# features as regressors at the DOT sampling rate
# --------------------------------------------------------------------------------------- #
FEATURES = ["luminance", "contrast", "flow", "cuts", "faces", "bodies", "speech", "loudness", "flux"]


def _gamma_pdf(t, shape, scale=1.0):
    t = np.maximum(t, 1e-12)
    return np.exp((shape - 1) * np.log(t) - t / scale - gammaln(shape) - shape * np.log(scale))


def hrf_canonical(dt, length=32.0):
    t = np.arange(0, length, dt)
    h = _gamma_pdf(t, 6) - _gamma_pdf(t, 16) / 6.0
    return h / h.sum()


def hrf_gamma(dt, length=32.0, tau=3.0, n=3):
    t = np.arange(0, length, dt)
    h = _gamma_pdf(t, n, tau)
    return h / h.sum()


def convolve(x, h):
    return np.convolve(x, h)[: len(x)]


def _resample(t, x, t_new, native_fs, cutoff=4.0):
    x = np.asarray(x, float)
    if native_fs > 2 * cutoff:
        x = sosfiltfilt(butter(4, cutoff, fs=native_fs, output="sos"), x)
    return np.interp(t_new, t, x)


def clip_regressors(d: dict, fs: float, edge_s: float = 1.5):
    """Feature file of one clip -> time vector and {feature: array} at the DOT rate.

    Loudness is expressed in dB above a -70 dB floor; cuts become a stick function. The first and
    last `edge_s` seconds (edited fades) are set to the clip's interior mean; model them with
    clip onset/offset regressors instead.
    """
    dur = float(d["duration"])
    t = np.arange(0, dur, 1 / fs)
    loud = np.maximum(d["loudness"], -70.0) + 70.0
    f = dict(luminance=_resample(d["t_frame"], d["luminance"], t, float(d["fps"])),
             contrast=_resample(d["t_frame"], d["contrast"], t, float(d["fps"])),
             flow=_resample(d["t_frame"], d["flow_nocut"], t, float(d["fps"])),
             faces=np.interp(t, d["t_det"], d["faces"]), bodies=np.interp(t, d["t_det"], d["bodies"]),
             speech=_resample(d["t_vad"], d["speech"], t, 31.25),
             loudness=_resample(d["t_stft"], loud, t, 100.0),
             flux=_resample(d["t_stft"], d["flux"], t, 100.0))
    f["cuts"] = np.histogram(d["cut_times"], bins=np.r_[t, t[-1] + 1 / fs])[0].astype(float)
    interior = (t >= edge_s) & (t < dur - edge_s)
    for k in f:
        f[k] = np.where(interior, f[k], f[k][interior].mean())
    return t, f


def legendre_drift(n, order):
    x = np.linspace(-1, 1, n)
    return np.column_stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(order + 1)])


def residualize(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - X @ beta


def vif(F, nuisance):
    """Variance inflation of each column of F within the design [F, nuisance]."""
    Fr = residualize(F, nuisance)
    tss = ((F - F.mean(0)) ** 2).sum(0)
    out = []
    for j in range(F.shape[1]):
        others = np.delete(Fr, j, axis=1)
        r = residualize(Fr[:, j], others) if others.shape[1] else Fr[:, j]
        out.append(tss[j] / max((r ** 2).sum(), 1e-12))
    return np.array(out)
