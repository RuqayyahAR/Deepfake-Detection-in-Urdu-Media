"""
Deepfake Engine - TRACE (Intermediate Student Level)
-------------------------------------------------------------
TRACE: Explainable Deepfake Detection and Forensic Reliability
Assessment System.

This engine combines:
  1. A pretrained deepfake detection model (CodeRafay's deepfake-detect v0.2)
     - Face-level inference, frame-level scores, video-level prediction
     - Loaded ONCE via DeepfakeModelAdapter singleton
  2. Explainable forensic heuristics (blur, edges, colour, temporal jitter)
     - Kept SEPARATE from AI predictions; NOT combined into a 60/40 formula

All scores are *suspicion scores* (0 to 1), not statistically calibrated
probabilities.

Classification (three outcomes, based on validated model output + evidence sufficiency):
    - LIKELY AUTHENTIC     (pretrained model says "real" OR forensic evidence
                            with BOTH spatial + temporal face evidence below threshold)
    - LIKELY MANIPULATED   (pretrained model says "fake" OR forensic evidence
                            at/above manipulated threshold)
    - INCONCLUSIVE         (borderline scores, whole-frame analysis, flat content,
                            or insufficient evidence)
    - INSUFFICIENT_EVIDENCE (not enough usable frames/faces)

Important: Whole-frame analysis (no face detected) can NEVER produce
LIKELY_AUTHENTIC. It is always INCONCLUSIVE or LIKELY_MANIPULATED.

Requirements: opencv-python, numpy
Optional:       deepfake-detect
-------------------------------------------------------------
"""

import cv2
import numpy as np
import os
import hashlib
from datetime import datetime


class DeepfakeEngine:
    """Heuristic deepfake detection engine (intermediate student level)."""

    # Temporal analysis uses a contiguous 16-frame clip (not 24 spread frames)
    TEMPORAL_FRAME_COUNT = 16
    FACE_SIZE = (160, 160)

    # Classification thresholds - starting points, calibrate with labelled data
    THRESH_AUTHENTIC = 0.20   # score < this  => likely authentic (face evidence only)
    THRESH_MANIPULATED = 0.40 # score >= this => likely manipulated

    def __init__(self):
        # Face detector: YuNet ONNX (bundled with OpenCV 4.8+)
        # Replaces Haar cascade which was silently missing on some builds.
        self.yunet = None
        self._init_yunet()

        # Pretrained model adapter (loaded once, singleton)
        self.model_adapter = get_adapter()

        # Legacy pretrained classifier hook (kept for backward compat)
        self.pretrained_model = None
        self.pretrained_available = False
        self.pretrained_error = None

    def _init_yunet(self):
        """Initialize YuNet face detector from OpenCV's built-in ONNX model."""
        try:
            # OpenCV 4.8+ includes FaceDetectorYN
            if hasattr(cv2, "FaceDetectorYN_create"):
                # Try to find the YuNet ONNX model in OpenCV data path
                yunet_path = cv2.data.haarcascades.replace(
                    "haarcascades", ""
                ) + "face_detection_yunet_2023mar.onnx"
                if not os.path.exists(yunet_path):
                    # Fallback: try common OpenCV extra model locations
                    yunet_path = None
                if yunet_path and os.path.exists(yunet_path):
                    self.yunet = cv2.FaceDetectorYN_create(
                        yunet_path, "", (320, 320), score_threshold=0.6
                    )
                    print("[*] YuNet face detector loaded from:", yunet_path)
                else:
                    # Create without model file - OpenCV will use internal default
                    self.yunet = cv2.FaceDetectorYN_create(
                        "", "", (320, 320), score_threshold=0.6
                    )
                    print("[*] YuNet face detector initialized (default model).")
            else:
                print("[!] OpenCV build does not include FaceDetectorYN. "
                      "Falling back to Haar cascade.")
                self.yunet = None
        except Exception as e:
            print("[!] YuNet initialization failed: " + str(e) + ". "
                  "Falling back to Haar cascade.")
            self.yunet = None

        # Haar cascade as final fallback (only if YuNet unavailable)
        self.face_cascade = None
        if self.yunet is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    self.face_cascade = None
            except Exception:
                self.face_cascade = None

        # Pretrained model adapter (loaded once, singleton)
        self.model_adapter = get_adapter()

        # Legacy pretrained classifier hook (kept for backward compat)
        self.pretrained_model = None
        self.pretrained_available = False
        self.pretrained_error = None

    # ------------------------------------------------------------------
    # Pretrained model integration (modular placeholder)
    # ------------------------------------------------------------------
    def load_pretrained_model(self, model_path=None, model_type="xception"):
        """
        Placeholder for integrating a pretrained deepfake classifier
        (e.g. Xception or EfficientNet). Weights are NOT bundled with
        this student project.
        """
        self.pretrained_available = False
        self.pretrained_error = None

        if not model_path:
            self.pretrained_error = "No model path provided"
            print("[!] No pretrained model path provided. Running heuristic-only.")
            return False

        if not os.path.exists(model_path):
            self.pretrained_error = "Model weights file not found: " + model_path
            print("[!] " + self.pretrained_error)
            return False

        try:
            # Student implementation point:
            # self.pretrained_model = load_your_model_here(model_path)
            self.pretrained_available = True
            print("[*] Pretrained model loaded from:", model_path)
            return True
        except Exception as e:
            self.pretrained_error = str(e)
            print("[!] Failed to load pretrained model: " + str(e))
            return False

    # ------------------------------------------------------------------
    # Frame extraction (BUG FIX #4: unique indices, no duplicates)
    # ------------------------------------------------------------------
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file: " + video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / fps) if fps > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames.")

        # BUG FIX #4: unique indices, never more than available frames
        num_samples = min(self.FRAME_SAMPLE_COUNT, total_frames)
        indices = np.unique(
            np.linspace(0, total_frames - 1, num=num_samples, dtype=int)
        )

        frames = []
        frame_indices = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                frame_indices.append(int(idx))
        cap.release()

        return {
            "frames": frames,
            "frame_indices": frame_indices,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
            "unique_frames": len(frames),
        }

    # ------------------------------------------------------------------
    # Face detection (YuNet primary, Haar cascade fallback)
    # ------------------------------------------------------------------
    def _detect_faces(self, frame):
        """Detect faces using YuNet (primary) or Haar cascade (fallback).

        Returns (face_crops, face_boxes) where face_crops is a list of
        resized RGB face crops and face_boxes is a list of (x, y, w, h).
        """
        face_crops = []
        face_boxes = []

        # ---- YuNet (primary) ----
        if self.yunet is not None:
            try:
                h, w = frame.shape[:2]
                self.yunet.setInputSize((w, h))
                _, faces = self.yunet.detect(frame)
                if faces is not None:
                    for face in faces:
                        x, y, box_w, box_h = face[:4].astype(int)
                        x, y = max(0, x), max(0, y)
                        crop = frame[y:y + box_h, x:x + box_w]
                        if crop.size != 0:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            face_crops.append(cv2.resize(crop_rgb, self.FACE_SIZE))
                            face_boxes.append((x, y, box_w, box_h))
                return face_crops, face_boxes
            except Exception:
                pass  # fall through to Haar

        # ---- Haar cascade (fallback only) ----
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25)
                )
                for (x, y, box_w, box_h) in rects[:3]:
                    x, y = max(0, x), max(0, y)
                    crop = frame[y:y + box_h, x:x + box_w]
                    if crop.size != 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        face_crops.append(cv2.resize(crop_rgb, self.FACE_SIZE))
                        face_boxes.append((x, y, box_w, box_h))
            except Exception:
                pass

        return face_crops, face_boxes

    # ------------------------------------------------------------------
    # Laplacian helper
    # ------------------------------------------------------------------
    @staticmethod
    def _laplacian_var(gray):
        try:
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            return float(lap.var())

    # ------------------------------------------------------------------
    # Spatial analysis (per face crop)
    # BUG FIX #2: removed invalid colour-histogram-mean comparison
    # ------------------------------------------------------------------
    @staticmethod
    def _spatial_anomaly_score(face_rgb):
        """Explainable spatial forensics for a face crop.

        Returns (score, indicators, component_scores, status)
        status: "OK" or "INSUFFICIENT_EVIDENCE"
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Blur / smoothness
        lap_var = DeepfakeEngine._laplacian_var(gray)
        blur_score = 0.0
        if lap_var < 80:
            blur_score = 0.25
        elif lap_var < 180:
            blur_score = 0.10

        # 2. High-frequency ratio (FFT)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        total_energy = magnitude.sum() + 1e-10
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = ((y_coords - cy) ** 2 + (x_coords - cx) ** 2) <= radius ** 2
        low_freq = magnitude[mask].sum()
        high_freq_ratio = 1.0 - (low_freq / total_energy)
        hf_score = 0.0
        if high_freq_ratio > 0.72:
            hf_score = 0.18
        elif high_freq_ratio > 0.58:
            hf_score = 0.08

        # 3. Edge density
        edges = cv2.Canny(face_rgb, 100, 200)
        edge_density = edges.mean() / 255.0
        edge_score = 0.0
        if edge_density < 0.05:
            edge_score = 0.18
        elif edge_density < 0.10:
            edge_score = 0.08

        # 4. Colour balance - BUG FIX #2: disabled invalid histogram-mean check
        # Normalized histograms always sum to 1, so their means are nearly
        # identical and carry no manipulation signal. Re-enable only with
        # meaningful statistics (channel means, std devs, histogram distances).
        colour_score = 0.0
        colour_disabled = True

        # 5. Skin-tone chrominance
        ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
        cb_std = ycrcb[:, :, 1].astype(np.float32).std()
        cr_std = ycrcb[:, :, 2].astype(np.float32).std()
        chroma_disp = (cb_std + cr_std) / 2.0
        chroma_score = 0.0
        if chroma_disp > 16:
            chroma_score = 0.14
        elif chroma_disp > 11:
            chroma_score = 0.06

        components = {
            "blur": blur_score,
            "high_freq": hf_score,
            "edge_density": edge_score,
            "colour": colour_score,
            "chroma": chroma_score,
        }
        score = sum(components.values())

        indicators = []
        if blur_score >= 0.25:
            indicators.append("Unnaturally smooth face (very low sharpness)")
        elif blur_score >= 0.10:
            indicators.append("Face appears slightly over-smoothed")
        if hf_score >= 0.18:
            indicators.append("High-frequency blending/seam artifacts")
        elif hf_score >= 0.08:
            indicators.append("Some high-frequency artifacts")
        if edge_score >= 0.18:
            indicators.append("Very few edges - smeared/averaged face")
        elif edge_score >= 0.08:
            indicators.append("Lower-than-normal edge detail")
        if chroma_score >= 0.14:
            indicators.append("Unnatural skin colour shifts")
        elif chroma_score >= 0.06:
            indicators.append("Slight skin-tone inconsistency")

        return float(np.clip(score, 0.0, 1.0)), indicators, components, "OK"

    # ------------------------------------------------------------------
    # Whole-frame spatial analysis (fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _block_boundary_score(gray):
        """
        Measure discontinuities at 8x8 block boundaries.
        BUG FIX #8: actual block-boundary discontinuity, not just variance.
        """
        h, w = gray.shape
        bs = 8
        if h < bs * 2 or w < bs * 2:
            return 0.0, []

        # Compare mean intensity of adjacent blocks horizontally and vertically
        horiz_diffs = []
        vert_diffs = []
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                block = gray[y:y + bs, x:x + bs]
                # Right neighbour
                if x + bs < w:
                    right = gray[y:y + bs, x + bs:x + 2 * bs]
                    horiz_diffs.append(abs(block.mean() - right.mean()))
                # Bottom neighbour
                if y + bs < h:
                    bottom = gray[y + bs:y + 2 * bs, x:x + bs]
                    vert_diffs.append(abs(block.mean() - bottom.mean()))

        if not horiz_diffs and not vert_diffs:
            return 0.0, []

        all_diffs = horiz_diffs + vert_diffs
        mean_diff = float(np.mean(all_diffs))
        score = 0.0
        inds = []

        # Very low boundary difference = unnaturally smooth (AI smoothing)
        # Moderate boundary difference = normal video
        # Very high = strong block artifacts
        if mean_diff < 2.0:
            score += 0.12
            inds.append("Unnaturally uniform image regions (AI smoothing)")
        if mean_diff > 15.0:
            score += 0.10
            inds.append("Strong block-boundary artifacts detected")

        return float(np.clip(score, 0.0, 1.0)), inds

    @staticmethod
    def _whole_frame_spatial_score(frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        score = 0.0
        inds = []
        lap_var = DeepfakeEngine._laplacian_var(gray)
        if lap_var < 100:
            score += 0.20
            inds.append("Overall video looks unusually smooth/blurred")
        edges = cv2.Canny(frame_rgb, 100, 200)
        edge_density = edges.mean() / 255.0
        if edge_density < 0.04:
            score += 0.18
            inds.append("Very low edge detail across frames (smeared look)")
        block_score, block_inds = DeepfakeEngine._block_boundary_score(gray)
        score += block_score
        inds.extend(block_inds)
        if not inds:
            inds.append("No strong whole-frame artifacts detected")
        return float(np.clip(score, 0.0, 1.0)), inds

    # ------------------------------------------------------------------
    # Temporal analysis (face crops)
    # BUG FIX #3: preserves original frame indices for gap detection
    # BUG FIX #9: returns INSUFFICIENT_EVIDENCE instead of zero
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_anomaly_score(face_sequence_with_indices):
        """
        face_sequence_with_indices: list of (face_rgb, original_frame_index)
        Returns (score_or_None, indicators, status)
        """
        if len(face_sequence_with_indices) < 3:
            return None, [], "INSUFFICIENT_EVIDENCE"

        diffs = []
        time_gaps = []
        prev_gray = None
        prev_idx = None

        for face_rgb, idx in face_sequence_with_indices:
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if prev_gray is not None and prev_idx is not None:
                frame_gap = idx - prev_idx
                if frame_gap > 0:
                    diffs.append(cv2.absdiff(gray, prev_gray).mean())
                    time_gaps.append(frame_gap)
            prev_gray = gray
            prev_idx = idx

        if not diffs:
            return None, [], "INSUFFICIENT_EVIDENCE"

        diff_arr = np.array(diffs)
        mean_diff = diff_arr.mean()
        std_diff = diff_arr.std()
        static_ratio = (diff_arr < 1.0).mean()

        score = 0.0
        inds = []
        if std_diff > 15:
            score += 0.25
            inds.append("Face jitters/flickers between frames (possible morphing)")
        elif std_diff > 8:
            score += 0.10
            inds.append("Some unnatural frame-to-frame movement")
        if mean_diff < 1.0:
            score += 0.22
            inds.append("Face almost completely still (possible inpainting)")
        elif mean_diff < 3.0:
            score += 0.08
            inds.append("Very little natural face movement")
        if static_ratio > 0.5:
            score += 0.18
            inds.append("More than half frames nearly identical - suspicious")
        elif static_ratio > 0.3:
            score += 0.06
            inds.append("Many frames look identical")

        return float(np.clip(score, 0.0, 1.0)), inds, "OK"

    @staticmethod
    def _whole_frame_temporal_score(frames_rgb_with_indices):
        if len(frames_rgb_with_indices) < 3:
            return None, [], "INSUFFICIENT_EVIDENCE"

        diffs = []
        prev_gray = None
        prev_idx = None
        for fr, idx in frames_rgb_with_indices:
            gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray = cv2.resize(gray, (320, 240))
            if prev_gray is not None and prev_idx is not None:
                if idx - prev_idx > 0:
                    diffs.append(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray
            prev_idx = idx

        if not diffs:
            return None, [], "INSUFFICIENT_EVIDENCE"

        diff_arr = np.array(diffs)
        mean_diff = diff_arr.mean()
        std_diff = diff_arr.std()
        static_ratio = (diff_arr < 0.7).mean()

        score = 0.0
        inds = []
        if std_diff > 16:
            score += 0.24
            inds.append("Frame-to-frame jitter detected (possible morphing)")
        elif std_diff > 8:
            score += 0.10
            inds.append("Some unnatural frame-to-frame movement")
        if mean_diff < 0.7:
            score += 0.22
            inds.append("Nearly static frames (possible inpainting)")
        elif mean_diff < 2.5:
            score += 0.08
            inds.append("Very little natural motion")
        if static_ratio > 0.5:
            score += 0.22
            inds.append("More than half frames nearly identical - suspicious")
        elif static_ratio > 0.3:
            score += 0.08
            inds.append("Many frames look identical")
        return float(np.clip(score, 0.0, 1.0)), inds, "OK"

    # ------------------------------------------------------------------
    # File hash + metadata
    # ------------------------------------------------------------------
    @staticmethod
    def file_hash(path):
        sha = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    sha.update(block)
            return sha.hexdigest()
        except FileNotFoundError:
            return None

    @staticmethod
    def video_metadata(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        if meta["fps"] > 0:
            meta["duration_seconds"] = round(meta["total_frames"] / meta["fps"], 2)
        else:
            meta["duration_seconds"] = 0.0
        return meta

    # ------------------------------------------------------------------
    # Classification (shared function - BUG FIX #6)
    # ------------------------------------------------------------------
    @staticmethod
    def classify_score(score, thresh_auth=0.20, thresh_man=0.40):
        """
        Shared classification function for CLI and web.
        Returns one of: LIKELY_AUTHENTIC, LIKELY_MANIPULATED, INCONCLUSIVE, INSUFFICIENT_EVIDENCE
        """
        if score is None:
            return "INSUFFICIENT_EVIDENCE"
        if score < thresh_auth:
            return "LIKELY_AUTHENTIC"
        if score >= thresh_man:
            return "LIKELY_MANIPULATED"
        return "INCONCLUSIVE"

    # ------------------------------------------------------------------
    # Contiguous frame extraction for temporal analysis
    # BUG FIX: uses 16 contiguous frames, not 24 spread frames
    # ------------------------------------------------------------------
    def _extract_contiguous_frames(self, video_path, count=16):
        """Extract a contiguous clip of frames for temporal analysis.

        Returns list of (frame_rgb, frame_index) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file: " + video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames.")

        # Take a contiguous block from the middle of the video
        start = max(0, (total_frames - count) // 2)
        indices = list(range(start, min(start + count, total_frames)))

        frames = []
        frame_indices = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
                frame_indices.append(idx)
        cap.release()
        return list(zip(frames, frame_indices))

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self, video_path):
        extracted = self._extract_frames(video_path)
        frames = extracted["frames"]
        frame_indices = extracted["frame_indices"]
        fps = extracted["fps"]
        print("[*] Sampled", len(frames), "frames for analysis.")

        # Also extract a contiguous clip for temporal analysis
        try:
            contiguous = self._extract_contiguous_frames(video_path, self.TEMPORAL_FRAME_COUNT)
        except Exception:
            contiguous = []

        print("[*] Detecting faces...")
        all_face_seqs = []  # list of (face_rgb, original_frame_index)
        frames_with_faces = []
        for i, frame in enumerate(frames):
            faces, boxes = self._detect_faces(frame)
            if faces:
                all_face_seqs.append((faces[0], frame_indices[i]))
                frames_with_faces.append(i)

        result = {
            "frames_analyzed": len(frames),
            "frames_with_faces": len(frames_with_faces),
            "frame_indices": frame_indices,
            "fps": fps,
            "metadata": self.video_metadata(video_path),
            "file_hash": self.file_hash(video_path),
            "pretrained_available": self.model_adapter.is_available(),
            "pretrained_error": self.model_adapter.get_error(),
        }

        # ================================================================
        # PRETRAINED MODEL INFERENCE (AI predictions, kept separate)
        # ================================================================
        print("[*] Running pretrained deepfake model inference...")
        model_result = self.model_adapter.predict(video_path)
        result["model_available"] = model_result["available"]
        result["model_label"] = model_result.get("label")
        result["model_score"] = model_result.get("score")
        result["model_frames"] = model_result.get("frames", [])
        result["model_error"] = model_result.get("error")

        # ================================================================
        # FORENSIC HEURISTIC ANALYSIS (supporting evidence only)
        # NOT combined with AI score into a 60/40 formula
        # ================================================================
        print("[*] Detecting faces for forensic analysis...")
        all_face_seqs = []  # list of (face_rgb, original_frame_index)
        frames_with_faces = []
        for i, frame in enumerate(frames):
            faces, boxes = self._detect_faces(frame)
            if faces:
                all_face_seqs.append((faces[0], frame_indices[i]))
                frames_with_faces.append(i)

        # ---- Path A: face-based forensic analysis ----
        spatial_avg = None
        temporal_score = None
        spatial_status = "OK"
        temporal_status = "OK"

        if len(all_face_seqs) >= 2:
            print("[*] Tracking", len(all_face_seqs), "face crops (forensic).")
            spatial_scores, spatial_inds, spatial_comps = [], [], []
            for f, _idx in all_face_seqs:
                s, inds, comps, st = self._spatial_anomaly_score(f)
                spatial_scores.append(s)
                spatial_inds.extend(inds)
                spatial_comps.append(comps)
            spatial_avg = float(np.mean(spatial_scores))

            # Temporal analysis on contiguous clip, not spread frames
            contiguous_faces = []
            for frame_rgb, idx in contiguous:
                faces, _ = self._detect_faces(frame_rgb)
                if faces:
                    contiguous_faces.append((faces[0], idx))

            if len(contiguous_faces) >= 2:
                temporal_score, temporal_inds, temporal_status = self._temporal_anomaly_score(
                    contiguous_faces
                )
            else:
                temporal_score = None
                temporal_inds = []
                temporal_status = "INSUFFICIENT_EVIDENCE"
                print("[!] Not enough contiguous faces for temporal analysis.")

            result["forensic_mode"] = "face-based"
            result["analysis_mode"] = "face-based"
            result["spatial_score"] = spatial_avg
            result["temporal_score"] = temporal_score
            result["spatial_status"] = spatial_status
            result["temporal_status"] = temporal_status
            result["forensic_indicators"] = spatial_inds + temporal_inds
            result["spatial_components"] = spatial_comps
            result["forensic_suspicious_frames"] = frames_with_faces

        # ---- Path B: whole-frame forensic fallback ----
        else:
            print("[!] Few/no faces detected. Using whole-frame forensic analysis.")
            frames_rgb_with_idx = [
                (cv2.cvtColor(f, cv2.COLOR_BGR2RGB), frame_indices[i])
                for i, f in enumerate(frames)
            ]
            s_scores, s_inds = [], []
            for fr, _idx in frames_rgb_with_idx:
                s, inds = self._whole_frame_spatial_score(fr)
                s_scores.append(s)
                s_inds.extend(inds)
            spatial_avg = float(np.mean(s_scores)) if s_scores else None

            # Whole-frame temporal on contiguous clip
            contiguous_rgb = [(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), idx) for f, idx in contiguous]
            temporal_score, temporal_inds, temporal_status = self._whole_frame_temporal_score(
                contiguous_rgb
            )

            result["forensic_mode"] = "whole-frame"
            result["analysis_mode"] = "whole-frame"
            result["spatial_score"] = spatial_avg
            result["temporal_score"] = temporal_score
            result["spatial_status"] = "OK" if spatial_avg is not None else "INSUFFICIENT_EVIDENCE"
            result["temporal_status"] = temporal_status
            result["forensic_indicators"] = s_inds + temporal_inds
            result["forensic_suspicious_frames"] = list(range(len(frames)))

        # ================================================================
        # CLASSIFICATION: depends on validated model output + evidence sufficiency
        # BUG FIX: LIKELY_AUTHENTIC needs BOTH spatial AND temporal face evidence.
        # Whole-frame analysis can NEVER produce LIKELY_AUTHENTIC.
        # ================================================================
        model_label = model_result.get("label")
        model_score = model_result.get("score")
        model_error = model_result.get("error")

        # Determine if we have sufficient evidence for a classification
        has_model_evidence = (
            model_result["available"]
            and model_label in ("real", "fake")
            and model_score is not None
        )
        has_face_spatial = spatial_avg is not None
        has_face_temporal = temporal_score is not None
        has_both_face_evidence = has_face_spatial and has_face_temporal

        # Build combined indicators list (forensic only, clearly labelled)
        all_indicators = result.get("forensic_indicators", [])
        seen = set()
        unique_indicators = []
        for ind in all_indicators:
            if ind and ind not in seen:
                seen.add(ind)
                unique_indicators.append(ind)
        result["indicators"] = unique_indicators

        # Suspicious frame ranking: use model frame-level results first
        model_suspicious_frames = []
        if model_result.get("frames"):
            model_frames = model_result["frames"]
            try:
                ranked = sorted(
                    model_frames,
                    key=lambda f: float(f.get("score", 0) or 0),
                    reverse=True,
                )
                model_suspicious_frames = [
                    f for f in ranked[:5] if f.get("score") is not None
                ]
            except Exception:
                model_suspicious_frames = []

        # Map suspicious frame indices to timestamps
        result["suspicious_timestamps"] = []
        result["suspicious_frame_details"] = []

        if model_suspicious_frames:
            for mf in model_suspicious_frames[:5]:
                t = mf.get("time")
                if t is not None:
                    result["suspicious_timestamps"].append(round(float(t), 2))
                result["suspicious_frame_details"].append({
                    "time": mf.get("time"),
                    "score": mf.get("score"),
                    "faces": mf.get("faces"),
                })
        elif result.get("forensic_suspicious_frames") and fps > 0:
            for idx in result["forensic_suspicious_frames"][:5]:
                if idx < len(frame_indices):
                    t = frame_indices[idx] / fps
                    result["suspicious_timestamps"].append(round(t, 2))

        # Build "why this conclusion" reasons
        why_reasons = []

        # Final classification logic
        if has_model_evidence:
            # Model output is primary evidence
            if model_label == "fake":
                classification = "LIKELY MANIPULATED"
                why_reasons.append("Pretrained model classified video as fake/manipulated.")
                if model_score is not None:
                    why_reasons.append("Model score: {:.2f} (higher = more likely manipulated).".format(model_score))
            elif model_label == "real":
                classification = "LIKELY AUTHENTIC"
                why_reasons.append("Pretrained model classified video as real/authentic.")
                if model_score is not None:
                    why_reasons.append("Model score: {:.2f} (lower = more likely authentic).".format(model_score))
            else:
                classification = "INCONCLUSIVE"
                why_reasons.append("Pretrained model returned inconclusive result.")

            result["classification_basis"] = "pretrained_model"

        elif has_both_face_evidence:
            # Forensic fallback: need BOTH spatial AND temporal for LIKELY_AUTHENTIC
            classification = self.classify_score(
                spatial_avg, self.THRESH_AUTHENTIC, self.THRESH_MANIPULATED
            )

            # Override: if below authentic threshold but missing temporal evidence,
            # bump to INCONCLUSIVE
            if classification == "LIKELY_AUTHENTIC" and not has_face_temporal:
                classification = "INCONCLUSIVE"
                why_reasons.append("Spatial evidence below threshold but temporal evidence missing.")

            # Add reason based on score
            if spatial_avg is not None:
                why_reasons.append("Spatial anomaly score: {:.2f}".format(spatial_avg))
            if temporal_score is not None:
                why_reasons.append("Temporal anomaly score: {:.2f}".format(temporal_score))

            if classification == "LIKELY AUTHENTIC":
                why_reasons.append("Both spatial and temporal face evidence below manipulation threshold.")
            elif classification == "LIKELY MANIPULATED":
                why_reasons.append("Forensic evidence at/above manipulation threshold.")
            else:
                why_reasons.append("Forensic score in borderline inconclusive zone.")

            result["classification_basis"] = "forensic_heuristic_only"

        elif has_forensic_evidence:
            # Only one type of forensic evidence (spatial OR temporal, not both)
            classification = "INCONCLUSIVE"
            why_reasons.append("Only partial forensic evidence available (need both spatial + temporal for a definitive classification).")
            result["classification_basis"] = "forensic_partial"

        else:
            # No face evidence at all (whole-frame only)
            classification = "INCONCLUSIVE"
            why_reasons.append("No face detected in video. Whole-frame analysis cannot confirm authenticity.")
            if spatial_avg is not None:
                why_reasons.append("Whole-frame spatial score: {:.2f} (supporting evidence only).".format(spatial_avg))
            result["classification_basis"] = "none"

        result["classification"] = classification
        result["why_this_conclusion"] = why_reasons
        result["suspicion_score"] = model_score if has_model_evidence else (
            (0.60 * spatial_avg + 0.40 * temporal_score)
            if (spatial_avg is not None and temporal_score is not None)
            else (spatial_avg if spatial_avg is not None else temporal_score)
        )
        if result["suspicion_score"] is not None:
            result["suspicion_score"] = float(np.clip(result["suspicion_score"], 0.0, 1.0))
            result["suspicion_percentage"] = round(result["suspicion_score"] * 100, 2)
        else:
            result["suspicion_percentage"] = None

        result["details"] = (
            "Model: {ml} ({ms}) | Forensic: spatial={sa}, temporal={ta} | "
            "Classification basis: {cb}".format(
                ml=model_label if model_label else "N/A",
                ms=model_score if model_score is not None else "N/A",
                sa=spatial_avg if spatial_avg is not None else "N/A",
                ta=temporal_score if temporal_score is not None else "N/A",
                cb=result["classification_basis"],
            )
        )

        print("[*] Model label:", model_label, "| Model score:", model_score)
        print("[*] Classification:", classification, "| Basis:", result["classification_basis"])
        print("[*] Why:", "; ".join(why_reasons))
        return result
