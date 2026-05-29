import os
import cv2
import whisper
import fasttext
import subprocess

class LanguageRouter:
    def __init__(self, fasttext_model_path):
        print("[*] Loading Language Verification Models...")
        self.whisper_model = whisper.load_model("base")
        self.ft_model = fasttext.load_model(fasttext_model_path)

    def extract_audio(self, video_path, output_audio_path="temp_audio.wav"):
        """Extracts audio channel using FFmpeg standard demuxing."""
        print(f"[*] Extracting audio from {video_path}...")
        command = f"ffmpeg -y -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_audio_path}"
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_audio_path

    def verify_urdu_media(self, video_path):
        """Transcribes audio and verifies if language is Urdu."""
        audio_file = self.extract_audio(video_path)
        
        if not os.path.exists(audio_file) or os.path.getsize(audio_file) < 1000:
            print("[!] No audio stream detected. Defaulting video to spatial visual pipeline.")
            return True 
            
        # Transcribe audio track
        result = self.whisper_model.transcribe(audio_file)
        text = result.get("text", "").replace("\n", " ")
        
        # Predict language using FastText
        predictions = self.ft_model.predict(text, k=1)
        language_label = predictions[0][0].replace("__label__", "")
        confidence = predictions[1][0]
        
        # Clean up temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        print(f"[+] Detected text: {text[:50]}...")
        print(f"[+] Language Classification: {language_label} (Confidence: {confidence:.2f})")
        
        return language_label == "ur"
