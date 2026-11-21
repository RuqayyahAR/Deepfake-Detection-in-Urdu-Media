"""
Language Router - TRACE
-------------------------------------------------------------
Spoken language identification using Whisper + FastText.

BUG FIX #10: Returns explicit statuses instead of a misleading Boolean.
Never returns True just because models are missing.

Statuses:
  - VERIFIED_URDU       - Audio transcribed and classified as Urdu
  - VERIFIED_NON_URDU   - Audio transcribed, classified as non-Urdu
  - SKIPPED_UNAVAILABLE - Whisper/FastText models not installed
  - NO_AUDIO            - No audio stream detected in video
  - ERROR               - Processing failed

This module is separate from deepfake detection. Language identification
does not vouch for authenticity.
-------------------------------------------------------------
"""

import os
import subprocess
import tempfile


class LanguageRouter:
    def __init__(self, fasttext_model_path):
        self.fasttext_model_path = fasttext_model_path
        self.whisper_model = None
        self.ft_model = None
        self.available = False
        self.load_error = None

        try:
            import whisper  # noqa: F401
            import fasttext  # noqa: F401
            self.whisper_model = whisper.load_model("base")
            self.ft_model = fasttext.load_model(fasttext_model_path)
            self.available = True
            print("[*] Language Verification Models loaded.")
        except Exception as e:
            self.available = False
            self.load_error = str(e)
            print("[!] Language models not available (" + str(e) + ").")
            print("[!] Language check will be SKIPPED_UNAVAILABLE.")

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------
    def extract_audio(self, video_path, output_audio_path=None):
        if output_audio_path is None:
            output_audio_path = os.path.join(
                tempfile.gettempdir(), "deepfake_temp_audio.wav"
            )

        print("[*] Extracting audio from " + video_path + "...")
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ab", "160k",
            "-ac", "2",
            "-ar", "44100",
            "-vn",
            output_audio_path,
        ]
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace").strip()
            print("[!] FFmpeg exited with code " +
                  str(result.returncode) + ": " + stderr_msg)
        return output_audio_path

    # ------------------------------------------------------------------
    # Language verification
    # ------------------------------------------------------------------
    def verify_media(self, video_path):
        """
        Returns a dict with explicit language_status:
          - VERIFIED_URDU
          - VERIFIED_NON_URDU
          - SKIPPED_UNAVAILABLE
          - NO_AUDIO
          - ERROR
        """
        # BUG FIX #10: if models missing, return SKIPPED_UNAVAILABLE, not True
        if not self.available:
            return {
                "language_status": "SKIPPED_UNAVAILABLE",
                "detected_language": None,
                "detected_confidence": None,
                "reason": "Language models unavailable: " + str(self.load_error),
            }

        audio_file = self.extract_audio(video_path)

        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
            return {
                "language_status": "NO_AUDIO",
                "detected_language": None,
                "detected_confidence": None,
                "reason": "No audio stream detected in video.",
            }

        try:
            result = self.whisper_model.transcribe(audio_file)
            text = result.get("text", "").replace("\n", " ")

            if not text or len(text.strip()) < 3:
                return {
                    "language_status": "NO_AUDIO",
                    "detected_language": None,
                    "detected_confidence": None,
                    "reason": "Whisper transcription returned no usable text.",
                }

            predictions = self.ft_model.predict(text, k=1)
            language_label = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]

            print("[+] Detected text: " + text[:50] + "...")
            print("[+] Language Classification: " +
                  language_label + " (Confidence: {:.2f})".format(confidence))

            is_urdu = language_label == "ur"
            return {
                "language_status": "VERIFIED_URDU" if is_urdu else "VERIFIED_NON_URDU",
                "detected_language": language_label,
                "detected_confidence": float(confidence),
                "reason": "Transcription and FastText classification completed.",
            }
        except Exception as e:
            return {
                "language_status": "ERROR",
                "detected_language": None,
                "detected_confidence": None,
                "reason": "Language processing failed: " + str(e),
            }
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

    # Backward-compatible alias
    def verify_urdu_media(self, video_path):
        """Deprecated alias for verify_media. Returns explicit status dict."""
        return self.verify_media(video_path)
