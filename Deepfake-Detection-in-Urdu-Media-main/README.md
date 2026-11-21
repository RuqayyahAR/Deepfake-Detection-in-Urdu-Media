# TRACE - Explainable Deepfake Detection & Forensic Reliability

TRACE: An investigator-oriented deepfake detection system that combines
frame-level AI classification, spatial and temporal evidence, technical
media analysis, and robustness evaluation under common video compression
and resizing transformations.

Built by: 2 intermediate + 1 beginner programming student
Designed to be: easy to understand, easy to present, easy to explain

---

## What is TRACE?

TRACE is a Flask web application that investigates whether a video looks
authentic or manipulated. It works with media in four languages: Hindi,
Urdu, Telugu and English. Instead of just predicting "real" or "fake",
TRACE produces a forensic investigation report with:

- A **suspicion score** (not a claimed probability)
- A **three-way classification**: Likely Authentic / Likely Manipulated / Inconclusive
- **Explainable indicators** showing *why* the system flagged the video
- **Video metadata** and **SHA-256 hash** for chain of custody
- **Suspicious frame timestamps**
- **Compression robustness testing** (how results change under re-encoding)
- **Limitations and investigation notes**

---

## What does it do?

1. Lets you pick a language (Hindi, Urdu, Telugu or English).
2. Lets you upload a video file OR paste a website link
   (YouTube, Instagram, Facebook, TikTok, Twitter/X, Vimeo, Reddit, etc.).
3. Downloads the video (with audio) and runs forensic checks:
   - Blur / smoothness analysis
   - Edge density and colour balance checks
   - Frequency-domain block artifact detection
   - Noise residue and uniformity analysis (AI smoothing patterns)
   - Temporal coherence / jitter detection (frame-to-frame morphing)
4. Produces a three-way classification based on transparent thresholds:
   - Pretrained model output takes priority when available
   - Forensic heuristics are supporting evidence only
   - "Likely Authentic" requires BOTH spatial AND temporal face evidence
5. Lists the specific forensic indicators found.
6. Searches the internet for the real/original version (only when the verdict is LIKELY MANIPULATED). Results are possible references, NOT proof of authenticity.
7. Tests compression robustness (original vs compressed/resized/re-encoded).

---

## How does it help?

TRACE focuses on Hindi, Urdu, Telugu and English regional-language media,
where many existing deepfake tools perform poorly. It gives students,
journalists and everyday users a way to investigate suspicious videos and
understand *why* the system reached its conclusion, rather than blindly
trusting a black-box "real/fake" label.

**Language support note**: Only Urdu has a spoken-language check (Whisper +
FastText, models not bundled). Hindi and Telugu are listed as UI language
options but do not yet have dedicated checks. The selected language does not
change the detection analysis.

---

## ⚠ Important Limitation

TRACE is an **intermediate student-level detector** that uses visual
heuristics and forensic image analysis. It does **not** use a trained
deep-learning model (no Xception/EfficientNet weights are bundled), so it
may miss very high-quality AI-generated videos. A result of "likely
authentic" does **not** guarantee the video is real.

The suspicion score is **not a genuine probability**. It is a transparent,
explainable composite of forensic indicators. Always verify with the AI
internet search results or the original source on the creator's official
channel.

---

## Colour Theme

The app uses a brownish / earthy colour theme (creams, light browns and
dark browns) throughout.

---

## Running the App

### 1. Install Python
Python 3.10 or higher. Tested with Python 3.15.

### 2. Install the required Python packages
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (needed for compression robustness + audio language check)
- Windows: `choco install ffmpeg`
- Linux: `sudo apt install ffmpeg`

### 4. (Optional) Install the pretrained deepfake detection model
TRACE works without a pretrained model (using forensic heuristics only), but for
AI-powered detection, install CodeRafay's `deepfake-detect`:

```bash
# Create a separate virtual environment (recommended)
py -m venv .venv-df
.\.venv-df\Scripts\Activate.ps1

# Install the model package (downloads ~1.7 GB CLIP weights on first run)
pip install --upgrade pip
pip install git+https://github.com/CodeRafay/deepfake-detect
```

The model is loaded ONCE when the app starts. If it is not installed, TRACE
falls back to forensic heuristic analysis and clearly labels the result as
"forensic_heuristic_only".

### 5. Configure environment
Create a `.env` file from the example:
```bash
copy .env.example .env
```
Then edit `.env` and set `FLASK_SECRET_KEY` to a random string (required by Flask).

### 5. Run the app
```bash
py app.py
# or
python app.py
```

Open your browser: **http://127.0.0.1:5000**

---

## Project Structure

```
Deepfake-Detection-in-Urdu-Media-main/
├── app.py                 # Flask web application (main entry point)
├── deepfake_engine.py     # Detection engine (spatial + temporal forensics)
├── language_router.py     # Optional spoken-language check (Whisper + FastText)
├── hash_check.py          # SHA-256 file hashing utility
├── ai_search.py           # Internet search for real/original video version
├── deepfake_model_adapter.py  # Pretrained model adapter (CodeRafay deepfake-detect)
├── requirements.txt       # Python dependencies
├── requirements-optional.txt  # Optional dependencies (Whisper, FastText)
├── README.md
├── static/
│   └── style.css          # Brownish theme stylesheet
├── templates/
│   ├── base.html          # Shared page layout
│   ├── index.html         # Home dashboard with description
│   ├── language.html      # Language selection page
│   ├── upload.html        # Upload video (file or website link)
│   ├── trace_upload.html  # Source tracing upload page
│   ├── trace_result.html  # Source tracing result page
│   └── result.html        # TRACE forensic report page
└── tracing/
    ├── __init__.py        # Package init
    ├── fingerprint.py     # Video pHash + audio spectral hash
    ├── align.py           # Offset search + per-second labels
    ├── evaluate.py        # Evaluation harness
    └── build_testset.py   # Synthetic test set builder
```

---

## How the Scoring Works (transparent math)

TRACE does not use a black-box model by default. It combines explainable components:

### 1. Pretrained Model (Primary Evidence, optional)
When CodeRafay's `deepfake-detect` is installed:
- Extracts faces from video frames using YuNet
- Runs frame-level inference through the pretrained classifier
- Returns a video-level label (`real` / `fake`) and a model score (0 to 1)
- The model score is **not** a calibrated probability

### 2. Forensic Heuristics (Supporting Evidence)
When the model is unavailable or as supporting evidence:
- **Spatial analysis** (per face crop): blur (Laplacian variance), high-frequency
  ratio (FFT), edge density (Canny), colour balance, chrominance dispersion,
  block-boundary artifacts, image uniformity.
- **Temporal analysis** (contiguous 16-frame clip): frame-to-frame motion
  difference, jitter (standard deviation), near-static frames, static frame ratio.
  Uses contiguous frames (not frames sampled seconds apart) so temporal checks
  measure actual manipulation, not scene change.
- **Combined heuristic score** = 0.60 × spatial + 0.40 × temporal (clipped to 0–1).

### Classification Logic
- If the pretrained model returns `real`/`fake` with a score → classification
  is based on the model output (**pretrained_model** basis)
- If the model is unavailable but BOTH spatial AND temporal face evidence exist →
  classification uses heuristic thresholds (**forensic_heuristic_only** basis)
  - Below 20% = Likely Authentic
  - 20–40% = Inconclusive
  - 40%+ = Likely Manipulated
- If only partial forensic evidence exists → **INCONCLUSIVE**
- If no face detected (whole-frame only) → **INCONCLUSIVE** (can never be Likely Authentic)
- If neither has sufficient evidence → **INSUFFICIENT_EVIDENCE**

These thresholds are starting points and should be calibrated with labelled
validation data in a real deployment.

### Face Detection
TRACE uses OpenCV's built-in YuNet face detector
(`cv2.FaceDetectorYN_create`) as the primary detector, with Haar cascade as a
fallback. This replaces the silently-failing Haar cascade used in earlier versions.

---

## Compression Robustness (research component)

TRACE tests how stable its result is under:
- Low-quality compression (CRF 28)
- Medium-quality compression (CRF 20)
- High-quality compression (CRF 15)
- Resizing to 720p
- Re-encoding to MPEG-4

This investigates whether social-media compression changes the detection
outcome — a key research question for real-world deepfake detection.

---

## Source Tracing (reference-based verification)

TRACE includes a source tracing module that answers: **Is this video the same
as a known reference, and which parts were altered?** This complements deepfake
detection (which asks "is this AI-generated?") by asking "where does this video
come from?"

### How it works

1. Upload a **suspect video** and a **reference video** (the known original)
2. The app extracts per-frame video pHashes (2 fps) and per-second audio
   spectral-shape hashes (4/sec)
3. It slides the suspect over the reference to find the best temporal offset
4. Each 1-second segment is labelled:
   - **MATCH** - both video and audio agree with the reference
   - **VIDEO_ALTERED** - video differs, audio matches
   - **AUDIO_ALTERED** - audio differs, video matches
   - **BOTH_ALTERED** - both differ

### Source tracing vs deepfake detection

| Source Tracing | Deepfake Detection |
|---|---|
| Compares against a known reference | Analyses the video in isolation |
| Tells you which segments are altered | Tells you if the video looks AI-generated |
| Good for detecting edits, splices, dubs | Good for detecting synthetic faces |
| Does not prove authenticity | Does not prove provenance |

Use both together for a complete investigation.

### Files

- `tracing/fingerprint.py` - Video pHash + audio spectral hash
- `tracing/align.py` - Offset search + per-second labels
- `tracing/evaluate.py` - Evaluation harness (retrieval + segment accuracy)
- `tracing/build_testset.py` - Synthetic test set builder (no real people)

---

## Evaluation Module

TRACE includes an evaluation toolkit for measuring detector accuracy on labelled
data. This is separate from the web app and runs as a CLI.

### Files

- `evaluation/manifest.py` - Validates a labelled CSV manifest before any metric
  is computed
- `evaluation/detectors.py` - Detector interface (`score(path) -> float`) +
  heuristic baseline that wraps the TRACE engine
- `evaluation/metrics.py` - AUC, EER (with threshold), bootstrap 95% CIs over
  videos
- `evaluation/run_eval.py` - Validate manifest -> score test split -> AUC /
  EER / CIs, per language
- `evaluation/lipsync.py` - Lip-sync vs audio energy cross-correlation
  (first version)

### Commands

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
python -m evaluation.run_eval \
    --manifest set.csv \
    --base /path/to/videos \
    --detector mymodule:MyDetector \
    --split test
```

Install optional dependencies:

```bash
pip install -r requirements-optional.txt
```

---

## Bug Fixes Applied (2026-11-21 Code Review)

The following issues were identified and fixed during the code review:

| # | Issue | Fix |
|---|-------|-----|
| 1 | Temporal checks used 24 frames spread seconds apart (measured scene change) | Now uses a 16-frame contiguous clip for temporal analysis |
| 2 | Haar face cascade silently missing on some OpenCV builds | Uses bundled YuNet (`cv2.FaceDetectorYN_create`) with Haar cascade fallback |
| 3 | "LIKELY AUTHENTIC" needed only spatial OR temporal evidence | Now requires BOTH spatial AND temporal face evidence. Whole-frame and flat content are always INCONCLUSIVE |
| 4 | `main.py` crashed (dict multiplied by 100) | CLI reads `suspicion_score` from result dict |
| 5 | Language check returned "matched" when it never ran | Explicit statuses: VERIFIED_URDU, VERIFIED_NON_URDU, SKIPPED_UNAVAILABLE, NO_AUDIO, ERROR |
| 6 | yt-dlp downloads were video-only (audio lost) | Downloads `bestvideo+bestaudio` merged with ffmpeg so audio reaches language check |
| 7 | AI search ran for every video with random filename query | AI search only runs when verdict is LIKELY MANIPULATED |
| 8 | Hard-coded Flask secret key + debug=True | `FLASK_SECRET_KEY` from environment variable; `debug=False` |
| 9 | URL downloader SSRF vulnerability | Blocks localhost/private/link-local/multicast/reserved IP addresses |
| 10 | Temporary file conflicts | Unique `tempfile.mkdtemp()` directory per compression test |
| 11 | requirements.txt listed unnecessary heavy packages | Core requirements separated; optional packages in `requirements-optional.txt` |

---

## Known Limitations

- **Heuristic-only detection**: Without the pretrained model, TRACE uses visual heuristics (blur, edges, colour, temporal jitter). These cannot reliably separate real from fake. A real face photo and a smoothed copy may score in the same zone.
- **Pretrained model not bundled**: CodeRafay's `deepfake-detect` (~1.7 GB CLIP weights) must be installed separately. If unavailable, TRACE falls back to forensic heuristics and clearly labels the result.
- **Language support**: Only Urdu has a spoken-language check (Whisper + FastText, models not bundled). Hindi and Telugu are listed as supported UI languages but do not yet have dedicated checks. The selected language does not change the detection analysis.
- **AI search**: Uses DuckDuckGo keyword scraping. Finding a matching video online does NOT prove the uploaded video is authentic.
- **No fake-news detection for news links**: Not yet implemented.
- **No camera-vs-AI real-time mode**: Not yet implemented.
- **Suspicion score is not a probability**: It is an explainable composite of forensic indicators and/or model output, not a statistically calibrated confidence.

## Novelty & Related Work

Multilingual deepfake detection already exists. Do not claim TRACE is the first. Related work includes:
- HAV-DF (Hindi audio-video deepfake dataset, arXiv 2411.15457)
- MAVOS-DD (multilingual audio-video open-set benchmark, arXiv 2505.11109)
- PolyGlotFake (multilingual multimodal dataset, arXiv 2405.08838)
- IndieFake (Indian-language audio deepfake dataset, arXiv 2506.19014)
- Indic-CodecFake / SATYAM (arXiv 2604.19949)
- Urdu deepfake audio dataset (ACL Findings 2024, 2024.findings-acl.861)

Open angles for student research:
- Lip-sync vs. transcript consistency in Urdu/Hindi/Telugu speech (Whisper transcript + mouth-landmark timing), evaluated on your own labelled set.
- Evasion resistance under platform re-encoding (cybersecurity framing).
- Your own labelled evaluation set with AUC and EER, plus a documented protocol.

## Plagiarism Notice

No tool can guarantee a plagiarism check will pass. Cite the datasets above, disclose that the starting code comes from this repository, and write any new modules yourself.

