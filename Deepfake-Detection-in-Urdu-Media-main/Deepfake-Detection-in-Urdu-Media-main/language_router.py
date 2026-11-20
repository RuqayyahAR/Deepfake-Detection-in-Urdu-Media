"""
Language Router
-------------------------------------------------------------
Detects the spoken language of a video using Whisper (transcription)
+ FastText (language identification).

Designed to be optional: if Whisper/FastText are not installed or the
FastText model file is missing, the class still works and simply reports
that the language check could not be run.
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

        # Try to load heavy models only if they are installed
        try:
            import whisper  # noqa: F401
            import fasttext  # noqa: F401
            self.whisper_model = whisper.load_model("base")
            self.ft_model = fasttext.load_model(fasttext_model_path)
            self.available = True
            print("[*] Language Verification Models loaded.")
        except Exception as e:
            # Models missing - language check will be skipped gracefully
            print("[!] Language models not available (" + str(e) + ").")
            print("[!] Visual analysis will run without language check.")

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------
    def extract_audio(self, video_path, output_audio_path=None):
        """Extracts audio channel using FFmpeg."""
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
    def verify_urdu_media(self, video_path):
        """Transcribes audio and checks if language is Urdu.

        Returns True if Urdu, False otherwise.
        If models are not available, returns True (skip language gate).
        """
        # If models could not be loaded, do not block the pipeline
        if not self.available:
            print("[!] Skipping language check - models not available.")
            return True

        audio_file = self.extract_audio(video_path)

        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
            print("[!] No audio stream detected. Defaulting to visual pipeline.")
            return True

        try:
            result = self.whisper_model.transcribe(audio_file)
            text = result.get("text", "").replace("\n", " ")

            predictions = self.ft_model.predict(text, k=1)
            language_label = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]

            print("[+] Detected text: " + text[:50] + "...")
            print("[+] Language Classification: " +
                  language_label + " (Confidence: {:.2f})".format(confidence))

            return language_label == "ur"
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)
