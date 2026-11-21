"""
Milestone 1 - align a suspect video against a reference and label each second.

Video and audio are aligned SEPARATELY, because an editor can shift the audio
track relative to the picture. Labels per 1-second segment:
  MATCH          video and audio both agree with the reference
  VIDEO_ALTERED  video disagrees, audio agrees
  AUDIO_ALTERED  audio disagrees, video agrees
  BOTH_ALTERED   both disagree
"""
import numpy as np
from .fingerprint import FPS, AUDIO_HOP

VIDEO_T = 0.25   # normalised Hamming distance: above => video differs
AUDIO_T = 0.25   # normalised Hamming distance: above => audio differs
AUDIO_PER_S = int(round(1 / AUDIO_HOP))   # audio hashes per second (4)


def ham(a, b):
    """Normalised Hamming distance over the last axis."""
    return (a != b).mean(axis=-1)


def best_offset(sv, rv):
    """Slide the suspect's hashes over the reference; return (offset_index, dist).

    Uses median distance so a few edge windows (codec priming, cut points)
    cannot decide the offset.
    """
    m, n = len(sv), len(rv)
    if m == 0 or n < m:
        return None, 1.0
    best = (None, 1.0)
    for k in range(n - m + 1):
        d = np.median(ham(sv, rv[k:k + m]))
        if d < best[1]:
            best = (k, d)
    return best


def label_segments(sv, sa, rv, ra, v_off, a_off):
    """Label each 1-second segment by comparing suspect vs reference hashes.

    sv, sa: suspect video/audio hash arrays
    rv, ra: reference video/audio hash arrays
    v_off, a_off: best alignment offsets (in hash indices)
    """
    seg_v, seg_a = FPS, AUDIO_PER_S
    labels = []
    for s in range(len(sv) // seg_v):
        v_s = sv[s * seg_v:(s + 1) * seg_v]
        v_r = rv[v_off + s * seg_v: v_off + (s + 1) * seg_v]
        dv = ham(v_s, v_r).mean() if len(v_r) == len(v_s) else 1.0
        a_s = sa[s * seg_a:(s + 1) * seg_a]
        a_r = ra[a_off + s * seg_a: a_off + (s + 1) * seg_a]
        da = ham(a_s, a_r).mean() if (len(a_r) == len(a_s) and len(a_s) > 0) else 1.0
        v_ok, a_ok = dv <= VIDEO_T, da <= AUDIO_T
        if v_ok and a_ok:
            lab = "MATCH"
        elif a_ok:
            lab = "VIDEO_ALTERED"
        elif v_ok:
            lab = "AUDIO_ALTERED"
        else:
            lab = "BOTH_ALTERED"
        labels.append({"t": s, "video_dist": round(float(dv), 3),
                       "audio_dist": round(float(da), 3), "label": lab})
    return labels


def trace(suspect_v, suspect_a, ref_v, ref_a):
    """Run full source tracing: align suspect to reference, label each second.

    Returns a dict with found flag, offsets, distances, and per-second labels.
    """
    v_off, v_dist = best_offset(suspect_v, ref_v)
    if v_off is None:
        return {"found": False, "reason": "reference too short"}
    a_off, a_dist = best_offset(suspect_a, ref_a)
    if a_off is None:
        a_off = v_off * AUDIO_PER_S // FPS
    labels = label_segments(suspect_v, suspect_a, ref_v, ref_a, v_off, a_off)
    return {
        "found": bool(v_dist <= VIDEO_T),
        "video_offset_seconds": v_off / FPS,
        "audio_offset_seconds": a_off / AUDIO_PER_S,
        "video_distance": round(float(v_dist), 3),
        "audio_distance": round(float(a_dist), 3),
        "segments": labels,
    }
