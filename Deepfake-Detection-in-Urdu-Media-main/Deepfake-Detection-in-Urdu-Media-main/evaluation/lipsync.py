"""
Lip-sync vs. speech-audio consistency (step 4, first version).

Idea: when a person speaks, mouth opening rises and falls with the speech
energy. Real speech shows a consistent lag (usually small). Lip-synced fakes
often show a weak correlation or an unusual lag.

Inputs:
  mouth_openness_series(landmarks) -> per-frame openness from FaceMesh-style
                                      landmarks (needs a landmark model; see notes)
  audio_envelope(pcm, sr)          -> per-frame RMS energy of the speech track
  av_sync(...)                     -> best lag and correlation

NOTE: this is the logic only. It is validated on synthetic signals here.
Real-face accuracy is NOT measured yet. Thresholds need labelled data.
Transcript consistency (Whisper word timing vs mouth motion) is a later step.
"""
import numpy as np

COMMON_FPS = 25.0
MAX_LAG_S = 0.5

# FaceMesh indices (MediaPipe): inner lips 13 (upper), 14 (lower); corners 61, 291
UPPER_LIP, LOWER_LIP, LEFT_CORNER, RIGHT_CORNER = 13, 14, 61, 291


def mouth_openness_series(frames_landmarks, width_px_per_frame=None):
    """frames_landmarks: list of arrays (468, 2 or 3) in normalised image coords,
    or None for frames with no face. Openness = inner-lip gap / mouth width.
    Missing frames are interpolated later."""
    out = []
    for lm in frames_landmarks:
        if lm is None:
            out.append(np.nan)
            continue
        lm = np.asarray(lm)
        gap = abs(lm[UPPER_LIP, 1] - lm[LOWER_LIP, 1])
        width = np.hypot(
            lm[LEFT_CORNER, 0] - lm[RIGHT_CORNER, 0],
            lm[LEFT_CORNER, 1] - lm[RIGHT_CORNER, 1],
        ) + 1e-9
        out.append(gap / width)
    return np.array(out, dtype=float)


def audio_envelope(pcm, sr, fps=COMMON_FPS):
    """RMS energy per 1/fps second."""
    hop = int(round(sr / fps))
    n = len(pcm) // hop
    x = np.asarray(pcm[: n * hop], dtype=float).reshape(n, hop)
    return np.sqrt((x**2).mean(1))


def _resample(x, fps_in, fps_out=COMMON_FPS):
    x = np.asarray(x, dtype=float)
    t_in = np.arange(len(x)) / fps_in
    t_out = (
        np.arange(0, t_in[-1], 1.0 / fps_out) if len(x) > 1 else np.array([0.0])
    )
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return np.zeros_like(t_out)
    return np.interp(t_out, t_in[ok], x[ok])


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def av_sync(openness, fps_video, envelope, fps_audio, max_lag_s=MAX_LAG_S):
    """Normalised cross-correlation between mouth openness and audio energy.

    Returns the lag (seconds, positive = mouth moves AFTER the sound) that gives the
    highest correlation within +/- max_lag_s, the peak correlation, and the
    correlation at zero lag.
    """
    a = _z(_resample(openness, fps_video))
    b = _z(_resample(envelope, fps_audio))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_k = int(round(max_lag_s * COMMON_FPS))
    best = (-2.0, 0)
    corr_zero = None
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            x, y = a[k:], b[: n - k]
        else:
            x, y = a[: n + k], b[-k:]
        m = min(len(x), len(y))
        if m < int(COMMON_FPS):  # need at least one second of overlap
            continue
        c = float(np.dot(_z(x[:m]), _z(y[:m])) / m)
        if k == 0:
            corr_zero = c
        if c > best[0]:
            best = (c, k)
    return {
        "lag_seconds": round(best[1] / COMMON_FPS, 3),
        "peak_correlation": round(best[0], 3),
        "correlation_at_zero_lag": round(
            corr_zero if corr_zero is not None else float("nan"), 3
        ),
    }
