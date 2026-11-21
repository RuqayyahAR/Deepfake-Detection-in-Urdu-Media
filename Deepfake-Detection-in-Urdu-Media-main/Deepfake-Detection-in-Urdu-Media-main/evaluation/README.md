# Evaluation Module

This directory contains the evaluation toolkit for TRACE: manifest validation,
detector interface, accuracy metrics (AUC / EER with bootstrap CIs), a runner,
and a first-pass lip-sync consistency check.

## Files

| File | Purpose |
|------|---------|
| `manifest.py` | Validates a labelled CSV manifest before any metric is computed |
| `detectors.py` | Detector interface (`score(path) -> float`) + heuristic baseline |
| `metrics.py` | AUC, EER (with threshold), bootstrap 95% CIs over videos |
| `run_eval.py` | Validate manifest -> score test split -> AUC / EER / CIs |
| `lipsync.py` | Lip-sync vs audio energy cross-correlation (first version) |

## Commands

```bash
# 1. Validate a labelled manifest
python -m evaluation.manifest set.csv /path/to/videos

# 2. Run evaluation on the test split using the heuristic detector
python -m evaluation.run_eval \
    --manifest set.csv \
    --base /path/to/videos \
    --detector heuristic \
    --split test

# 3. Evaluate with a custom pretrained detector
#    (implement detectors.PretrainedDetector first)
python -m evaluation.run_eval \
    --manifest set.csv \
    --base /path/to/videos \
    --detector mymodule:MyDetector \
    --split test
```

## Manifest CSV format

Required columns:

| Column | Allowed values |
|--------|---------------|
| path | relative path to video file |
| label | `real` \| `fake` |
| split | `train` \| `val` \| `test` |
| source | dataset name (text) |
| language | `en` \| `hi` \| `ur` \| `te` \| `none` |
| manipulation | `none` \| `faceswap` \| `lipsync` \| `synthesis` \| `edit` \| `other` |
| licence | text |
| sha256 | optional - checked against file if present |

## Lip-sync validation (synthetic signals)

```
aligned 0.00 s        -> lag 0.00, peak corr 0.99
mouth 0.2 s after sound -> lag +0.20, 0.99
mouth 0.2 s before sound -> lag -0.20, 0.98
unrelated motion      -> peak corr 0.06
flat mouth            -> peak corr 0.00
```

## Not done (needs your machine or data)

- Real labelled videos (datasets need access requests and licence checks;
  do not redistribute)
- Pretrained model: no model selected; huggingface.co and
  storage.googleapis.com are blocked here; torch and whisper are not installed
- Real-face lip landmarks: MediaPipe FaceLandmarker needs its .task model file
  (download blocked here)
- Whisper transcript timing vs mouth motion (transcript step)

## Related work (for citation)

- HAV-DF (Hindi audio-video deepfake dataset, arXiv 2411.15457)
- MAVOS-DD (multilingual audio-video open-set benchmark, arXiv 2505.11109)
- PolyGlotFake (multilingual multimodal dataset, arXiv 2405.08838)
- IndieFake (Indian-language audio deepfake dataset, arXiv 2506.19014)
- Indic-CodecFake / SATYAM (arXiv 2604.19949)
- Urdu deepfake audio dataset (ACL Findings 2024, 2024.findings-acl.861)

## Dependencies

Add to your environment:

```bash
pip install scikit-learn numpy
```

The core TRACE app (`requirements.txt`) does not need scikit-learn; it is only
required for the evaluation module.
