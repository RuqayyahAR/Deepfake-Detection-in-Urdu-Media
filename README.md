# Deepfake Detector — Multilingual AI Media & Forensic Analysis

**Deepfake Detector** is a Flask + Celery + Redis forensic investigation platform supporting **English, Hindi, Urdu, and Telugu** video and audio media.

## Directory Structure

```text
Deepfake-Detection_mini-project/
├── README.md
└── Deepfake-Detection/
    ├── app.py                   Flask app factory & web UI routes
    ├── config.py                Environment-based configuration & limits
    ├── extensions.py            SQLAlchemy & Celery initialization
    ├── tasks.py                 Celery background pipeline + thread-pool fallback
    ├── requirements.txt
    ├── .env.example
    ├── api/
    │   └── routes.py            REST API (/api/analyze, /api/jobs/<id>, /api/reports/<id>)
    ├── database/
    │   └── models.py            AnalysisJob, AnalysisResult, ForensicEvidence, ModelExecution, AnalysisReport
    ├── services/
    │   ├── storage.py           Magic-byte validation, SHA-256, SSRF-safe URL validation & yt-dlp download
    │   ├── preprocessing.py     FFprobe inspection, frame extraction, 16kHz mono WAV extraction
    │   ├── visual.py            YuNet face detection + ViT deepfake classifier + visual forensics
    │   ├── audio.py             Wav2Vec2 audio deepfake classifier + Librosa acoustic analysis
    │   ├── language.py          Whisper spoken-language ID, transcription & speech/prosody metrics
    │   ├── forensic.py          SHA-256, FFprobe metadata, video pHash, audio spectral fingerprint
    │   ├── aggregation.py       Multimodal evidence aggregation (4-way verdict)
    │   └── reports.py           Traceable forensic report builder
    ├── models/
    │   └── face_detection_yunet_2023mar.onnx
    ├── static/
    │   └── style.css            Preserved earthy brown theme (#f5efe3, #faf6f0, #5d4037, #8d6e63, #d7ccc8)
    ├── templates/
    │   ├── base.html
    │   ├── index.html           Dashboard (introduction + Video/Audio buttons)
    │   ├── upload.html          Combined language selection + media upload / video URL input
    │   ├── job_status.html      Real-time background processing tracker
    │   └── report.html          Evidence-backed forensic report
    ├── tests/                   Pytest suite
    ├── evaluation/              Benchmark manifest & model evaluation scripts
    └── tracing/                 Reference-based source tracing package
```

## Setup & Execution

### 1. System Dependencies
- **FFmpeg + FFprobe** on `PATH`
- **Redis** (`redis-server` on `localhost:6379`) for Celery background tasks (when Redis is offline, the backend automatically uses an in-process background `ThreadPoolExecutor` fallback)

### 2. Python Environment
```powershell
cd "C:\Users\amatu\OneDrive\Documents\Deepfake-Detection_mini-project\Deepfake-Detection"
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### 3. Pretrained Models
Cached Hugging Face weights used by default:
- Visual: `dsabljic/deepfake_faces_vit` (plus `models/face_detection_yunet_2023mar.onnx`)
- Audio: `mo-thecreator/Deepfake-audio-detection`
- Speech & Language: `openai/whisper-tiny`

### 4. Start Services
**Terminal 1 — Redis:**
```powershell
redis-server
```

**Terminal 2 — Celery Worker (Windows `--pool=solo`):**
```powershell
..\.venv\Scripts\Activate.ps1
celery -A tasks:get_celery_app worker --loglevel=info --pool=solo
```

**Terminal 3 — Flask Web App:**
```powershell
..\.venv\Scripts\Activate.ps1
python app.py
```

Open `http://127.0.0.1:5000`.

### 5. Run Tests
```powershell
..\.venv\Scripts\Activate.ps1
pytest tests/ -v
```

## Known Limitations
- Model scores (`0.000` to `1.000`) are uncalibrated classifier confidences, not true posterior probabilities.
- `Likely Authentic` does not prove a file's complete provenance or recording history.
- SHA-256 verifies exact file byte integrity only.
- Whisper performs speech transcription and language identification; it is not an audio deepfake classifier.
- Bundled `evaluation/data/` clips are synthetic smoke-test placeholders; multilingual accuracy benchmarking requires external labelled datasets for English, Hindi, Urdu, and Telugu.
