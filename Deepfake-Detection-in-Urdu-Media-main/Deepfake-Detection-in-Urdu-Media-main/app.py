"""
Deepfake Detection Web App - TRACE
-------------------------------------------------------------
TRACE: Explainable Deepfake Detection and Forensic Reliability
Assessment System.

Flask web interface (student-level, easy to understand and present).

Flow:
1. Language selection (Hindi, Urdu, Telugu, English)
2. Upload video file OR paste website link (YouTube, Instagram, etc.)
3. Forensic analysis -> suspicion score + three-way classification
4. Forensic report with metadata, SHA-256, frame-level findings,
   compression robustness, and limitations

Made by: 2 intermediate + 1 beginner programming students
-------------------------------------------------------------
"""

import os
import sys
import uuid
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, redirect, url_for, session, flash

from hash_check import generate_file_hash
from deepfake_engine import DeepfakeEngine
from language_router import LanguageRouter

try:
    from ai_search import search_real_video
    SEARCH_AVAILABLE = True
except Exception:
    SEARCH_AVAILABLE = False

app = Flask(__name__)
app.secret_key = "student-deepfake-project-2026"

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "deepfake_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

SUPPORTED_LANGUAGES = {
    "hindi": "Hindi",
    "urdu": "Urdu",
    "telugu": "Telugu",
    "english": "English",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", languages=SUPPORTED_LANGUAGES)


@app.route("/select-language", methods=["GET", "POST"])
def select_language():
    if request.method == "POST":
        lang = request.form.get("language")
        if lang not in SUPPORTED_LANGUAGES:
            flash("Please pick a valid language.", "error")
            return redirect(url_for("select_language"))
        session["language"] = lang
        flash("Selected " + SUPPORTED_LANGUAGES[lang] + ". Now upload your video.", "info")
        return redirect(url_for("upload"))
    return render_template("language.html", languages=SUPPORTED_LANGUAGES)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "language" not in session:
        flash("Please select a language first.", "warning")
        return redirect(url_for("select_language"))

    if request.method == "POST":
        video_path = None
        source_type = None

        if "video_file" in request.files:
            file = request.files["video_file"]
            if file and file.filename != "" and allowed_file(file.filename):
                filename = str(uuid.uuid4()) + "_" + file.filename
                video_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(video_path)
                source_type = "file"

        link = request.form.get("website_link", "").strip()
        if not video_path and link:
            try:
                video_path = download_video_from_link(link)
                source_type = "link"
            except Exception as e:
                flash("Could not download video from link: " + str(e), "error")
                return redirect(url_for("upload"))

        if not video_path:
            flash("Please upload a video file OR paste a website link.", "error")
            return redirect(url_for("upload"))

        session["video_path"] = video_path
        session["source_type"] = source_type
        session["source_link"] = link if source_type == "link" else ""
        return redirect(url_for("result"))

    return render_template(
        "upload.html",
        language=SUPPORTED_LANGUAGES.get(session.get("language", ""), ""),
    )


def download_video_from_link(link):
    import requests

    social_domains = [
        "youtube.com", "youtu.be", "instagram.com", "facebook.com",
        "fb.watch", "tiktok.com", "twitter.com", "x.com",
        "vimeo.com", "reddit.com",
    ]
    is_social = any(domain in link.lower() for domain in social_domains)

    if is_social:
        return _download_with_ytdlp(link)
    return _download_direct(link)


def _download_with_ytdlp(link):
    import yt_dlp
    filename = str(uuid.uuid4()) + ".%(ext)s"
    path_template = os.path.join(UPLOAD_FOLDER, filename)
    ydl_opts = {
        "outtmpl": path_template,
        "format": "bestvideo[ext=mp4]/bestvideo",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(link, download=True)
    except Exception as e:
        raise ValueError(
            "Could not download video from this link using yt-dlp. "
            "Make sure the link is public and valid. Error: " + str(e)
        )
    downloaded_file = ydl.prepare_filename(info)
    if not os.path.exists(downloaded_file):
        prefix = os.path.basename(path_template).replace(".%(ext)s", "")
        for f in os.listdir(UPLOAD_FOLDER):
            if f.startswith(prefix):
                downloaded_file = os.path.join(UPLOAD_FOLDER, f)
                break
    if not os.path.exists(downloaded_file) or os.path.getsize(downloaded_file) < 4096:
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)
        raise ValueError(
            "Downloaded file is too small to be a video. "
            "The link may be private or unavailable."
        )
    return downloaded_file


def _download_direct(link):
    import requests
    resp = requests.get(link, stream=True, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "The link returned a webpage (HTML), not a video file. "
            "For YouTube/Instagram/Facebook, paste the share link directly."
        )
    filename = str(uuid.uuid4()) + ".mp4"
    path = os.path.join(UPLOAD_FOLDER, filename)
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    if os.path.getsize(path) < 4096:
        os.remove(path)
        raise ValueError(
            "Downloaded file is too small to be a video. "
            "Please paste a direct link to a video file."
        )
    return path


# ---------------------------------------------------------------------------
# Compression robustness testing (TRACE research component)
# ---------------------------------------------------------------------------
def compression_robustness_test(video_path):
    """
    Create compressed/resized copies of the video and run detection on each.
    Returns a list of dicts: {transformation, score, classification, indicators_count}
    """
    results = []

    engine = DeepfakeEngine()
    base = engine.analyze(video_path)
    results.append({
        "transformation": "Original (no modification)",
        "score": base["combined_score"],
        "classification": base["classification"],
        "indicators": len(base.get("indicators", [])),
    })

    import subprocess

    def run_ffmpeg(input_path, output_path, args):
        cmd = ["ffmpeg", "-y", "-i", input_path] + args + [output_path]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 4096
        except Exception:
            return False

    base_dir = tempfile.gettempdir()

    p1 = os.path.join(base_dir, "trace_lowq.mp4")
    if run_ffmpeg(video_path, p1, ["-vcodec", "libx264", "-crf", "28", "-preset", "fast"]):
        try:
            r1 = engine.analyze(p1)
            results.append({
                "transformation": "Low-quality compression (CRF 28)",
                "score": r1["combined_score"],
                "classification": r1["classification"],
                "indicators": len(r1.get("indicators", [])),
            })
        except Exception:
            results.append({
                "transformation": "Low-quality compression (CRF 28)",
                "score": None, "classification": "Error", "indicators": 0,
            })
        finally:
            if os.path.exists(p1):
                os.remove(p1)

    p2 = os.path.join(base_dir, "trace_medq.mp4")
    if run_ffmpeg(video_path, p2, ["-vcodec", "libx264", "-crf", "20", "-preset", "fast"]):
        try:
            r2 = engine.analyze(p2)
            results.append({
                "transformation": "Medium-quality compression (CRF 20)",
                "score": r2["combined_score"],
                "classification": r2["classification"],
                "indicators": len(r2.get("indicators", [])),
            })
        except Exception:
            results.append({
                "transformation": "Medium-quality compression (CRF 20)",
                "score": None, "classification": "Error", "indicators": 0,
            })
        finally:
            if os.path.exists(p2):
                os.remove(p2)

    p3 = os.path.join(base_dir, "trace_highq.mp4")
    if run_ffmpeg(video_path, p3, ["-vcodec", "libx264", "-crf", "15", "-preset", "fast"]):
        try:
            r3 = engine.analyze(p3)
            results.append({
                "transformation": "High-quality compression (CRF 15)",
                "score": r3["combined_score"],
                "classification": r3["classification"],
                "indicators": len(r3.get("indicators", [])),
            })
        except Exception:
            results.append({
                "transformation": "High-quality compression (CRF 15)",
                "score": None, "classification": "Error", "indicators": 0,
            })
        finally:
            if os.path.exists(p3):
                os.remove(p3)

    p4 = os.path.join(base_dir, "trace_resized.mp4")
    if run_ffmpeg(video_path, p4, ["-vf", "scale=1280:720"]):
        try:
            r4 = engine.analyze(p4)
            results.append({
                "transformation": "Resized to 720p",
                "score": r4["combined_score"],
                "classification": r4["classification"],
                "indicators": len(r4.get("indicators", [])),
            })
        except Exception:
            results.append({
                "transformation": "Resized to 720p",
                "score": None, "classification": "Error", "indicators": 0,
            })
        finally:
            if os.path.exists(p4):
                os.remove(p4)

    p5 = os.path.join(base_dir, "trace_reencoded.mp4")
    if run_ffmpeg(video_path, p5, ["-vcodec", "mpeg4", "-qscale:v", "5"]):
        try:
            r5 = engine.analyze(p5)
            results.append({
                "transformation": "Re-encoded (MPEG-4)",
                "score": r5["combined_score"],
                "classification": r5["classification"],
                "indicators": len(r5.get("indicators", [])),
            })
        except Exception:
            results.append({
                "transformation": "Re-encoded (MPEG-4)",
                "score": None, "classification": "Error", "indicators": 0,
            })
        finally:
            if os.path.exists(p5):
                os.remove(p5)

    return results


# ---------------------------------------------------------------------------
# Result route
# ---------------------------------------------------------------------------
@app.route("/result")
def result():
    video_path = session.get("video_path")
    language = session.get("language", "english")
    source_type = session.get("source_type", "file")
    source_link = session.get("source_link", "")

    if not video_path or not os.path.exists(video_path):
        flash("No video found. Please upload again.", "error")
        return redirect(url_for("upload"))

    engine = DeepfakeEngine()
    try:
        analysis = engine.analyze(video_path)
    except Exception as e:
        flash("Could not analyze the video: " + str(e), "error")
        return redirect(url_for("upload"))

    score = analysis.get("combined_score", 0.0)
    classification = analysis.get("classification", "INCONCLUSIVE")
    indicators = analysis.get("indicators", [])
    details = analysis.get("details", "")
    metadata = analysis.get("metadata", {})
    file_hash = analysis.get("file_hash")
    suspicious_ts = analysis.get("suspicious_timestamps", [])

    lang_info = None
    try:
        router = LanguageRouter("lid.176.bin")
        detected = router.verify_urdu_media(video_path)
        lang_info = {
            "matched": detected,
            "selected": language,
            "note": "Language check completed.",
        }
    except Exception:
        lang_info = {
            "matched": None,
            "selected": language,
            "note": "Language models not available - visual analysis only.",
        }

    search_results = []
    if SEARCH_AVAILABLE:
        try:
            query = source_link if source_link else os.path.basename(video_path)
            search_results = search_real_video(query)
        except Exception as e:
            search_results = [{"title": "Search unavailable",
                               "url": "#",
                               "note": "Could not search internet: " + str(e)}]

    compression_results = compression_robustness_test(video_path)

    return render_template(
        "result.html",
        language=SUPPORTED_LANGUAGES.get(language, language),
        score=score,
        classification=classification,
        indicators=indicators,
        details=details,
        file_hash=file_hash,
        video_name=os.path.basename(video_path),
        source_type=source_type,
        source_link=source_link,
        lang_info=lang_info,
        metadata=metadata,
        frames_analyzed=analysis.get("frames_analyzed"),
        suspicious_timestamps=suspicious_ts,
        analysis_mode=analysis.get("analysis_mode"),
        search_results=search_results,
        compression_results=compression_results,
    )


if __name__ == "__main__":
    app.run(debug=True)
