"""
Milestone 1 - fingerprints for reference-based source tracing.

video_hashes(path)  -> (n, 64) uint8 array: one pHash per frame at FPS frames/sec
audio_hashes(path)  -> (m, 16) uint8 array: one band-energy-difference hash per 0.25 s

Both use only ffmpeg + numpy + OpenCV + scipy (no Chromaprint/Whisper yet).
"""
import subprocess
import numpy as np
import cv2
from scipy.signal import stft

FPS = 2                 # video hashes per second
AUDIO_SR = 8000
HOP = 100              # STFT hop in samples: 80 frames/s, so 0.25 s = exactly 20 frames


def _run_ffmpeg(args):
    cmd = ["ffmpeg", "-v", "error"] + args
    return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout


def video_hashes(path):
    """Perceptual hash per frame: 64-bit pHash at 2 fps.

    Extracts frames at FPS fps, resizes to 8x8 DCT low-frequency block,
    thresholds against median of AC coefficients to produce 64 bits.
    """
    raw = _run_ffmpeg(["-i", path, "-vf", f"fps={FPS},scale=64:64,format=gray",
                       "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 64, 64)
    out = np.zeros((len(frames), 64), np.uint8)
    for i, f in enumerate(frames):
        small = cv2.resize(f, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        d = cv2.dct(small)[:8, :8].flatten()          # low-frequency block
        med = np.median(d[1:])                          # ignore DC term
        out[i] = (d > med).astype(np.uint8)
    return out


AUDIO_WIN = 0.5         # seconds per audio window
AUDIO_HOP = 0.25        # seconds between audio windows (4 hashes per second)
AUDIO_BANDS = 16
# Fixed per-band thresholds, calibrated ONCE on reference clips that were NOT
# the evaluation target. Fixed, not per-clip, so that a trimmed copy hashes
# exactly like the original at the same moment.
AUDIO_THRESHOLDS = np.array([1.9447, 7.6646, 7.2850, 3.2516, 0.9524, -0.0367,
                             -0.6025, -1.0319, -1.3979, -1.6819, -1.9147,
                             -2.1260, -2.3079, -2.4934, -2.8866, -3.8763])


def audio_features(path):
    """Loudness-free band log-energies per window, shape (bands, windows)."""
    raw = _run_ffmpeg(["-i", path, "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                       "-f", "s16le", "-"])
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if len(x) < 512:
        return None
    _, _, Z = stft(x, fs=AUDIO_SR, nperseg=256, noverlap=256 - HOP)
    E = np.abs(Z) ** 2
    edges = np.linspace(0, E.shape[0], AUDIO_BANDS + 1, dtype=int)
    bands = np.stack([E[edges[i]:edges[i + 1]].sum(0) for i in range(AUDIO_BANDS)])
    frames_per_s = AUDIO_SR / HOP
    w, h = int(round(AUDIO_WIN * frames_per_s)), int(round(AUDIO_HOP * frames_per_s))
    cols = [bands[:, s:s + w].mean(1) for s in range(0, bands.shape[1] - w + 1, h)]
    if not cols:
        return None
    W = np.log(np.stack(cols, 1) + 1e-9)
    return W - W.mean(0, keepdims=True)


def audio_hashes(path):
    """Spectral-shape hash: 16 bits per 0.25 s window (0.5 s long), 4 per second."""
    W = audio_features(path)
    if W is None:
        return np.zeros((0, AUDIO_BANDS), np.uint8)
    return (W > AUDIO_THRESHOLDS[:, None]).T.astype(np.uint8)
