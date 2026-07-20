# Automated Deepfake Detection in Urdu Media

An ongoing research-driven AI project focused on combating digital misinformation in regional broadcasting. This system leverages deep learning models to analyze video and audio streams, specifically optimized to detect synthetic manipulations and deepfakes within Urdu media contexts where traditional western-centric datasets lack representative coverage.

---

## Problem Statement & Context

Most state-of-the-art deepfake detection models are trained heavily on high-definition datasets consisting primarily of English speakers (e.g., FaceForensics++, DFDC). When deployed on regional broadcasts, these models experience significant performance degradation due to:

1. **Low-resolution feeds & broadcast compression artifacts** typical in regional television.
2. **Unique linguistic patterns, lip sync mechanics, and phonetic traits** inherent to the Urdu language.
3. **Cultural variations** in lighting, studio environments, and traditional attire affecting facial landmarking.

This project aims to bridge that gap by building an optimized preprocessing and classification pipeline tailored to Urdu news anchors, political broadcasts, and digital media.

---

## System Pipeline Architecture

The system processes video files through a multi-stage sequential execution pipeline:

1. **Video Ingestion & Frame Extraction:** Video streams are decoded using OpenCV and broken down into temporal frame sequences.
2. **Facial Detection & ROI Extraction:** MTCNN (Multi-task Cascaded Convolutional Networks) crops and aligns faces, isolating the Region of Interest (ROI) while filtering out background broadcast graphics.
3. **Sequence Chunking:** Continuous frames are batched into fixed temporal segments to preserve motion vectors and audio-visual sync data.
4. **Dual-Stream Deep Learning Inference Engine:**
   - **Spatial Stream:** Analyzes per-frame facial anomalies -- Laplacian variance, FFT energy ratios, edge density, and chrominance dispersion.
   - **Temporal Stream:** Analyzes inter-frame coherence -- jitter, static-inpainting detection, and optical-flow irregularities.
5. **Probability Output:** Generates a localized confidence score indicating whether the media asset is "Authentic" or "Manipulated".

---

## Tech Stack

- **Language:** Python 3.10+
- **Deep Learning Frameworks:** PyTorch, Whisper
- **Computer Vision:** OpenCV, MTCNN
- **Language Detection:** FastText (lid.176.bin)
- **Audio Processing:** FFmpeg

---

## Key Features

- **Multi-Modal Visual Analysis:** Inspects spatial anomalies (facial blending artifacts, unnatural smoothness, frequency-domain irregularities) and temporal inconsistencies across consecutive video frames.
- **Regional Optimization:** Tailored features to account for varying broadcast quality, stream-compression ratios, and specific low-light conditions unique to mainstream regional television packages.
- **Urdu Language Verification:** Automatic audio transcription via Whisper and language classification via FastText to verify Urdu-origin media before visual analysis.
- **Forensic Chain of Custody:** SHA-256 hash computation for evidentiary integrity.

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed and available on PATH
- A CUDA-capable GPU is recommended for faster Whisper transcription and MTCNN face detection

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/urdu-deepfake-detection.git
   cd urdu-deepfake-detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the FastText language identification model:**
   ```bash
   curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```

5. **Install FFmpeg (if not already present):**
   ```bash
   # Windows (Chocolatey)
   choco install ffmpeg
   # Ubuntu / Kali
   sudo apt update && sudo apt install ffmpeg -y
   ```

### Usage

```bash
# Full pipeline: hash -> language check -> deepfake inference -> report
python main.py --input path/to/video.mp4

# Specify a custom FastText model path
python main.py --input path/to/video.mp4 --fasttext /path/to/lid.176.bin

# Standalone hash utility
python hash_check.py path/to/video.mp4
```

### Example Output

```
==================================================
   URDU MEDIA FORENSIC PIPELINE (REMOTE ENGINE)
==================================================
[*] Environment Diagnostic Log:
    [-] Hostname: kali-vm
    [-] Execution Context: REMOTE-SSH (Headless VM)
    [-] Base Directory: /home/user/urdu-deepfake-detection

[+] Forensic Chain of Custody Initialized.
[+] Target Hash: a1b2c3d4...
[*] Loading Language Verification Models...
[+] Language Classification: ur (Confidence: 0.97)
[+] Media validated. Running language-optimized pipeline weights.
[*] Running Dual-Stream Spatial/Temporal AI inference engine...
[*] Extracted 30 sampled frames.
[*] Tracking 30 face crops across frames.
[*] Spatial anomaly score:  0.2100
[*] Temporal anomaly score: 0.1500
[*] Combined manipulation probability: 0.1830

================ ANALYSIS RESULTS ================
Asset File: video.mp4
SHA-256 Identifier: a1b2c3d4...
Calculated Manipulation Probability: 18.30%
STATUS: AUTHENTIC MEDIA ASSET
==================================================
```

---

## Project Structure

```
.
├── main.py                 # Pipeline entry point and CLI orchestrator
├── deepfake_engine.py      # Dual-stream spatial/temporal inference engine
├── language_router.py      # Urdu language verification (Whisper + FastText)
├── hash_check.py           # SHA-256 file hashing utility
├── requirements.txt        # Python dependencies
└── README.md
```
