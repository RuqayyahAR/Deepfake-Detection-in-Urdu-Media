"""
Deepfake Detection Web App - TRACE
-------------------------------------------------------------
TRACE: Explainable Deepfake Detection and Forensic Reliability
Assessment System.

Flask web interface (student-level, easy to understand and present).

BUG FIXES APPLIED:
  #12: Flask secret key from environment variable, debug=False
  #13: Upload validation (size, format, cleanup)
  #14: SSRF protection (block localhost/private IPs)
  #15: Unique temp directories for compression testing

Flow:
1. Language selection (Hindi, Urdu, Telugu, English)
2. Upload video file OR paste website link
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
import ipaddress
import subprocess

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

try:
    from tracing.fingerprint import video_hashes, audio_hashes, FPS, AUDIO_HOP
    from tracing.align import trace as trace_video, VIDEO_T, AUDIO_T
    TRACING_AVAILABLE = True
except Exception:
    TRACING_AVAILABLE = False

# BUG FIX #12: Secret key from environment variable
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is not configured")

# Security limits
MAX_UPLOAD_SIZE_MB = 500
MAX_DOWNLOAD_SIZE_MB = 500
MAX_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

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


def is_safe_url(hostname):
    """
    BUG FIX #14: SSRF protection.
    Block localhost, private IP ranges, and internal network addresses.
    """
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # It's a hostname, not an IP. Allow but could add DNS allowlisting.
        return True
    # Block private, loopback, link-local, multicast, reserved
    if (ip.is_private or ip.is_loopback or ip.is_link_local or
            ip.is_multicast or ip.is_reserved):
        return False
    return True


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
        flash("Selected " + SUPPORTED_LANGUAGES[lang] +
              ". Now upload your video.", "info")
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

        # BUG FIX #13: Upload validation (size, format, cleanup)
        if "video_file" in request.files:
            file = request.files["video_file"]
            if file and file.filename != "" and allowed_file(file.filename):
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                if file_size > MAX_BYTES:
                    flash("File too large (max %d MB)." % MAX_UPLOAD_SIZE_MB, "error")
                    return redirect(url_for("upload"))
                if file_size == 0:
                    flash("Empty file uploaded.", "error")
                    return redirect(url_for("upload"))

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
        # Download best video + best audio and merge (needs ffmpeg).
        # This keeps the audio track so the language check can run.
        "format": "bestvideo+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
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
    # BUG FIX #14: SSRF - parse hostname and block private/loopback IPs
    from urllib.parse import urlparse
    try:
        parsed = urlparse(link)
        hostname = parsed.hostname
        if hostname and not is_safe_url(hostname):
            raise ValueError(
                "Unsafe URL destination: blocked localhost/private IP address."
            )
    except Exception:
        # If we can't parse it, don't download
        raise ValueError("Invalid URL format.")

    import requests
    try:
        resp = requests.get(link, stream=True, timeout=30,
                            allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ValueError("Failed to download from URL: " + str(e))

    # Check content type
    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "The link returned a webpage (HTML), not a video file. "
            "For YouTube/Instagram/Facebook, paste the share link directly."
        )

    # Check size limit while downloading
    filename = str(uuid.uuid4()) + ".mp4"
    path = os.path.join(UPLOAD_FOLDER, filename)
    total = 0
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            total += len(chunk)
            if total > MAX_BYTES:
                f.close()
                os.remove(path)
                raise ValueError(
                    "Download too large (max %d MB)." % MAX_DOWNLOAD_SIZE_MB
                )
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
# BUG FIX #15: unique temp directory per analysis
# ---------------------------------------------------------------------------
def compression_robustness_test(video_path):
    """
    Create compressed/resized copies of the video and run detection on each.
    Uses a unique temporary directory per analysis to avoid conflicts.
    """
    results = []

    engine = DeepfakeEngine()
    base = engine.analyze(video_path)
    results.append({
        "transformation": "Original (no modification)",
        "score": base["suspicion_score"],
        "classification": base["classification"],
        "indicators": len(base.get("indicators", [])),
    })

    def run_ffmpeg(input_path, output_path, args):
        cmd = ["ffmpeg", "-y", "-i", input_path] + args + [output_path]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 4096
        except Exception:
            return False

    # BUG FIX #15: unique temp directory
    work_dir = tempfile.mkdtemp(prefix="trace_robust_")
    try:
        p1 = os.path.join(work_dir, "trace_lowq.mp4")
        if run_ffmpeg(video_path, p1, ["-vcodec", "libx264", "-crf", "28", "-preset", "fast"]):
            try:
                r1 = engine.analyze(p1)
                results.append({
                    "transformation": "Low-quality compression (CRF 28)",
                    "score": r1["suspicion_score"],
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

        p2 = os.path.join(work_dir, "trace_medq.mp4")
        if run_ffmpeg(video_path, p2, ["-vcodec", "libx264", "-crf", "20", "-preset", "fast"]):
            try:
                r2 = engine.analyze(p2)
                results.append({
                    "transformation": "Medium-quality compression (CRF 20)",
                    "score": r2["suspicion_score"],
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

        p3 = os.path.join(work_dir, "trace_highq.mp4")
        if run_ffmpeg(video_path, p3, ["-vcodec", "libx264", "-crf", "15", "-preset", "fast"]):
            try:
                r3 = engine.analyze(p3)
                results.append({
                    "transformation": "High-quality compression (CRF 15)",
                    "score": r3["suspicion_score"],
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

        p4 = os.path.join(work_dir, "trace_resized.mp4")
        if run_ffmpeg(video_path, p4, ["-vf", "scale=1280:720"]):
            try:
                r4 = engine.analyze(p4)
                results.append({
                    "transformation": "Resized to 720p",
                    "score": r4["suspicion_score"],
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

        p5 = os.path.join(work_dir, "trace_reencoded.mp4")
        if run_ffmpeg(video_path, p5, ["-vcodec", "mpeg4", "-qscale:v", "5"]):
            try:
                r5 = engine.analyze(p5)
                results.append({
                    "transformation": "Re-encoded (MPEG-4)",
                    "score": r5["suspicion_score"],
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
    finally:
        # Clean up the unique work directory
        try:
            os.rmdir(work_dir)
        except Exception:
            pass

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

    score = analysis.get("suspicion_score")
    classification = analysis.get("classification", "INSUFFICIENT_EVIDENCE")
    indicators = analysis.get("indicators", [])
    details = analysis.get("details", "")
    metadata = analysis.get("metadata", {})
    file_hash = analysis.get("file_hash")
    suspicious_ts = analysis.get("suspicious_timestamps", [])
    frames_analyzed = analysis.get("frames_analyzed")
    analysis_mode = analysis.get("analysis_mode")

    # Pretrained model results (kept separate from forensic indicators)
    model_label = analysis.get("model_label")
    model_score = analysis.get("model_score")
    model_frames = analysis.get("model_frames", [])
    model_error = analysis.get("model_error")
    model_available = analysis.get("model_available", False)
    classification_basis = analysis.get("classification_basis", "unknown")
    suspicious_frame_details = analysis.get("suspicious_frame_details", [])

    # Language check (optional)
    lang_info = None
    try:
        router = LanguageRouter("lid.176.bin")
        detected = router.verify_urdu_media(video_path)
        lang_info = {
            "status": detected.get("language_status", "unknown"),
            "detected_language": detected.get("detected_language"),
            "confidence": detected.get("detected_confidence"),
            "note": detected.get("reason", ""),
        }
    except Exception:
        lang_info = {
            "status": "ERROR",
            "detected_language": None,
            "confidence": None,
            "note": "Language models not available - visual analysis only.",
        }

    # AI internet search (BUG FIX #11: use source_context, not filename)
    # BUG FIX: Only run AI search for LIKELY MANIPULATED videos, not every upload.
    search_results = []
    if SEARCH_AVAILABLE and classification == "LIKELY MANIPULATED":
        try:
            query = source_link if source_link else os.path.basename(video_path)
            search_results = search_real_video(query=query,
                                               source_context=source_link)
        except Exception as e:
            search_results = [{"title": "Search unavailable",
                               "url": "#",
                               "note": "Could not search internet: " + str(e)}]
    elif SEARCH_AVAILABLE:
        search_results = [{"title": "Search skipped",
                           "url": "#",
                           "note": "AI search only runs when the verdict is "
                                   "LIKELY MANIPULATED. Current verdict: " +
                                   classification}]

    # Compression robustness (BUG FIX #15: unique temp dir)
    compression_results = compression_robustness_test(video_path)

    # Clean up the analyzed video after processing
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception:
        pass

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
        frames_analyzed=frames_analyzed,
        suspicious_timestamps=suspicious_ts,
        analysis_mode=analysis_mode,
        # Pretrained model data (kept separate from forensic indicators)
        model_label=model_label,
        model_score=model_score,
        model_frames=model_frames,
        model_error=model_error,
        model_available=model_available,
        classification_basis=classification_basis,
        suspicious_frame_details=suspicious_frame_details,
        search_results=search_results,
        compression_results=compression_results,
    )


# ---------------------------------------------------------------------------
# Source tracing route
# ---------------------------------------------------------------------------
@app.route("/trace", methods=["GET", "POST"])
def trace_source():
    """Source tracing: compare a suspect video against a reference video.

    Uses per-frame video pHash and per-second audio spectral hashes to
    detect which segments match the reference and which are altered.
    """
    if not TRACING_AVAILABLE:
        flash("Source tracing is not available (tracing module missing).",
              "error")
        return redirect(url_for("index"))

    if request.method == "POST":
        suspect_path = None
        reference_path = None

        # Suspect video upload
        if "suspect_file" in request.files:
            file = request.files["suspect_file"]
            if file and file.filename != "" and allowed_file(file.filename):
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                if file_size > MAX_BYTES:
                    flash("Suspect file too large (max %d MB)." %
                          MAX_UPLOAD_SIZE_MB, "error")
                    return redirect(url_for("trace_source"))
                if file_size == 0:
                    flash("Empty suspect file uploaded.", "error")
                    return redirect(url_for("trace_source"))
                filename = str(uuid.uuid4()) + "_" + file.filename
                suspect_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(suspect_path)

        # Reference video upload
        if "reference_file" in request.files:
            file = request.files["reference_file"]
            if file and file.filename != "" and allowed_file(file.filename):
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                if file_size > MAX_BYTES:
                    flash("Reference file too large (max %d MB)." %
                          MAX_UPLOAD_SIZE_MB, "error")
                    return redirect(url_for("trace_source"))
                if file_size == 0:
                    flash("Empty reference file uploaded.", "error")
                    return redirect(url_for("trace_source"))
                filename = str(uuid.uuid4()) + "_" + file.filename
                reference_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(reference_path)

        if not suspect_path or not reference_path:
            flash("Please upload both a suspect video and a reference video.",
                  "error")
            return redirect(url_for("trace_source"))

        try:
            sv, sa = video_hashes(suspect_path), audio_hashes(suspect_path)
            rv, ra = video_hashes(reference_path), audio_hashes(reference_path)
            result = trace_video(sv, sa, rv, ra)
        except Exception as e:
            flash("Tracing failed: " + str(e), "error")
            # Clean up
            for p in (suspect_path, reference_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            return redirect(url_for("trace_source"))

        # Clean up uploaded files after processing
        for p in (suspect_path, reference_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        return render_template(
            "trace_result.html",
            suspect_name=os.path.basename(suspect_path),
            reference_name=os.path.basename(reference_path),
            result=result,
        )

    return render_template("trace_upload.html")


if __name__ == "__main__":
    # BUG FIX #12: debug=False for deployment
    app.run(debug=False)
