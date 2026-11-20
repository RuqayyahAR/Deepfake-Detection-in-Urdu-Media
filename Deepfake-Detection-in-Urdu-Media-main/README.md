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
3. Downloads the video and runs forensic checks:
   - Blur / smoothness analysis
   - Edge density and colour balance checks
   - Frequency-domain block artifact detection
   - Noise residue and uniformity analysis (AI smoothing patterns)
   - Temporal coherence / jitter detection (frame-to-frame morphing)
4. Produces a three-way classification based on transparent thresholds.
5. Lists the specific indicators found.
6. Searches the internet for the real/original version for comparison.
7. Tests compression robustness (original vs compressed/resized/re-encoded).

---

## How does it help?

TRACE focuses on Hindi, Urdu, Telugu and English regional-language media,
where many existing deepfake tools perform poorly. It gives students,
journalists and everyday users a way to investigate suspicious videos and
understand *why* the system reached its conclusion, rather than blindly
trusting a black-box "real/fake" label.

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

### 4. Run the app
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
├── requirements.txt       # Python dependencies
├── README.md
├── static/
│   └── style.css          # Brownish theme stylesheet
└── templates/
    ├── base.html          # Shared page layout
    ├── index.html         # Home dashboard with description
    ├── language.html      # Language selection page
    ├── upload.html        # Upload video (file or website link)
    └── result.html        # TRACE forensic report page
```

---

## How the Scoring Works (transparent math)

TRACE does not use a black-box model. It combines explainable components:

- **Spatial score** (weighted 60%): blur (Laplacian variance), high-frequency
  ratio (FFT), edge density (Canny), colour balance, chrominance dispersion,
  block-boundary artifacts, image uniformity.
- **Temporal score** (weighted 40%): frame-to-frame motion difference, jitter
  (standard deviation), near-static frames, static frame ratio.
- **Combined score** = 0.60 × spatial + 0.40 × temporal (clipped to 0–1).

**Classification thresholds:**
- **40%+** = Likely Manipulated
- **20–40%** = Inconclusive
- **Below 20%** = Likely Authentic

These thresholds are starting points and should be calibrated with labelled
validation data in a real deployment.

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
