"""
TRACE Source Tracing Package
-----------------------------
Reference-based source tracing for video and audio.

Components:
  - fingerprint.py  : Video pHash per frame (2 fps) + audio spectral-shape hash (4/sec)
  - align.py        : Offset search + per-second labels (MATCH / VIDEO_ALTERED /
                      AUDIO_ALTERED / BOTH_ALTERED)
  - evaluate.py     : Retrieval + segment accuracy evaluation harness

This complements deepfake detection: deepfake detection asks "is this AI-generated?"
while source tracing asks "is this the same video as a known reference, and which
parts were altered?"

Requirements: ffmpeg, numpy, opencv-python, scipy
"""

from .fingerprint import video_hashes, audio_hashes, FPS, AUDIO_HOP
from .align import ham, best_offset, trace, VIDEO_T, AUDIO_T

__all__ = [
    "video_hashes",
    "audio_hashes",
    "FPS",
    "AUDIO_HOP",
    "ham",
    "best_offset",
    "trace",
    "VIDEO_T",
    "AUDIO_T",
]
