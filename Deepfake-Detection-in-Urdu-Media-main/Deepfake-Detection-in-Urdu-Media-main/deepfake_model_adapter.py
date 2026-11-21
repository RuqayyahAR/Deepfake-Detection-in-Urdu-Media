"""
Deepfake Model Adapter - TRACE
Integrates a pretrained deepfake detection model (CodeRafay's deepfake-detect v0.2)
with TRACE. AI predictions are kept SEPARATE from forensic heuristic indicators.
"""

import os


class DeepfakeModelAdapter:
    """Adapter for CodeRafay's pretrained deepfake detector (singleton)."""

    _instance = None
    _detector = None
    _available = False
    _error = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            from deepfake_detect import Detector
            self._detector = Detector()
            self._available = True
            print("[*] Pretrained deepfake model loaded successfully.")
        except ImportError as e:
            self._available = False
            self._error = (
                "deepfake-detect package not installed. "
                "Install with: pip install git+https://github.com/CodeRafay/deepfake-detect"
            )
            print("[!] " + self._error)
        except Exception as e:
            self._available = False
            self._error = "Failed to load pretrained model: " + str(e)
            print("[!] " + self._error)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_available(self):
        return self._available

    def get_error(self):
        return self._error

    def predict(self, video_path):
        if not self._available:
            return {
                "available": False,
                "label": None,
                "score": None,
                "frames": [],
                "error": self._error or "Model not available",
            }

        if not os.path.exists(video_path):
            return {
                "available": True,
                "label": "error",
                "score": None,
                "frames": [],
                "error": "Video file not found: " + video_path,
            }

        try:
            result = self._detector.predict(video_path)
            normalized = {
                "available": True,
                "label": result.get("label"),
                "score": result.get("score"),
                "frames": result.get("frames", []),
                "error": None,
            }
            if normalized["score"] is not None:
                try:
                    normalized["score"] = float(normalized["score"])
                except (ValueError, TypeError):
                    normalized["score"] = None
            return normalized
        except Exception as e:
            return {
                "available": True,
                "label": "error",
                "score": None,
                "frames": [],
                "error": "Inference failed: " + str(e),
            }


def get_adapter():
    return DeepfakeModelAdapter.get_instance()
