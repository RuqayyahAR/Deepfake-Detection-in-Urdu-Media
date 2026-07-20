import os
import subprocess
import tempfile
import whisper
import fasttext


class LanguageRouter:
    def __init__(self, fasttext_model_path):
        print("[*] Loading Language Verification Models...")
        self.whisper_model = whisper.load_model("base")
        self.ft_model = fasttext.load_model(fasttext_model_path)

    def extract_audio(self, video_path, output_audio_path=None):
        """Extracts audio channel using FFmpeg standard demuxing."""
        if output_audio_path is None:
            output_audio_path = os.path.join(tempfile.gettempdir(), "deepfake_temp_audio.wav")

        print(f"[*] Extracting audio from {video_path}...")
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ab", "160k",
            "-ac", "2",
            "-ar", "44100",
            "-vn",
            output_audio_path,
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr_msg = result.stderr.decode(errors="replace").strip()
            print(f"[!] FFmpeg exited with code {result.returncode}: {stderr_msg}")
        return output_audio_path

    def verify_urdu_media(self, video_path):
        """Transcribes audio and verifies if language is Urdu."""
        audio_file = self.extract_audio(video_path)

        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
            print("[!] No audio stream detected. Defaulting video to spatial visual pipeline.")
            return True

        try:
            result = self.whisper_model.transcribe(audio_file)
            text = result.get("text", "").replace("\n", " ")

            predictions = self.ft_model.predict(text, k=1)
            language_label = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]

            print(f"[+] Detected text: {text[:50]}...")
            print(f"[+] Language Classification: {language_label} (Confidence: {confidence:.2f})")

            return language_label == "ur"
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)
