"""
Deepfake Engine - TRACE (Intermediate Student Level)
-------------------------------------------------------------
TRACE: Explainable Deepfake Detection and Forensic Reliability
Assessment System.

This engine does NOT claim to output a genuine probability. It outputs
a transparent *suspicion score* (0 to 1) built from explainable forensic
indicators, plus a three-way classification:
    - LIKELY AUTHENTIC
    - LIKELY MANIPULATED
    - INCONCLUSIVE

All heuristics are documented so students can present and explain them.

Modules:
  1. Spatial analysis        - per-frame visual forensics (blur, edges,
                               colour, block artifacts, uniformity)
  2. Temporal analysis       - cross-frame motion / jitter / static checks
  3. Pretrained classifier   - modular placeholder for Xception/EfficientNet
                               integration (weights not bundled)
  4. Forensic report         - SHA-256, metadata, frame-level findings,
                               suspicious timestamps, limitations

Requirements: opencv-python, numpy
Optional:       torch, torchvision, mtcnn, whisper, fasttext
-------------------------------------------------------------
"""

import cv2
import numpy as np
import os
import hashlib
from datetime import datetime


class DeepfakeEngine:
    """Heuristic deepfake detection engine (intermediate student level)."""

    FRAME_SAMPLE_COUNT = 24
    FACE_SIZE = (160, 160)

    # Classification thresholds (calibration placeholders - adjust with
    # validation data in a real deployment)
    THRESH_MANIPULATED = 0.40   # >= this => likely manipulated
    THRESH_AUTHENTIC   = 0.20   # <  this  => likely authentic
    # between the two => inconclusive

    def __init__(self):
        # Face detectors
        self.use_mtcnn = False
        self.face_cascade = None

        try:
            from mtcnn import MTCNN
            import torch  # noqa: F401
            self.detector = MTCNN()
            self.use_mtcnn = True
        except Exception:
            self.use_mtcnn = False

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

        # Pretrained classifier hook (not loaded by default)
        self.pretrained_model = None
        self.pretrained_available = False

    # ------------------------------------------------------------------
    # Pretrained model integration (modular placeholder)
    # ------------------------------------------------------------------
    def load_pretrained_model(self, model_path=None, model_type="xception"):
        """
        Placeholder for integrating a pretrained deepfake classifier
        (e.g. Xception or EfficientNet). Weights are NOT bundled with
        this student project. To enable, provide a path to saved weights
        and implement the forward pass here.
        """
        self.pretrained_available = False
        if model_path and os.path.exists(model_path):
            # Student implementation point:
            # self.pretrained_model = load_your_model_here(model_path)
            self.pretrained_available = True
            print("[*] Pretrained model loaded from:", model_path)
        else:
            print("[!] No pretrained model weights found. "
                  "Running heuristic-only analysis.")

    # ------------------------------------------------------------------
    # Frame extraction
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

        indices = np.linspace(
            0, total_frames - 1, self.FRAME_SAMPLE_COUNT, dtype=int
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
        }

    # ------------------------------------------------------------------
    # Face detection (multi-scale)
    # ------------------------------------------------------------------
    def _detect_faces(self, frame):
        face_crops = []
        face_boxes = []

        if self.use_mtcnn:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                detections = self.detector.detect_faces(rgb)
                for det in detections:
                    if det.get("confidence", 0) < 0.85:
                        continue
                    x, y, w, h = det["box"]
                    x, y = max(0, x), max(0, y)
                    crop = rgb[y:y + h, x:x + w]
                    if crop.size != 0:
                        face_crops.append(cv2.resize(crop, self.FACE_SIZE))
                        face_boxes.append((x, y, w, h))
            except Exception:
                pass

        if self.face_cascade is not None and len(face_crops) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scales = [1.0, 0.75, 0.5]
            for scale in scales:
                if scale != 1.0:
                    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                rects = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25)
                )
                if len(rects) > 0:
                    sx = frame.shape[1] / gray.shape[1]
                    sy = frame.shape[0] / gray.shape[0]
                    for (x, y, w, h) in rects[:3]:
                        x, y = int(x * sx), int(y * sy)
                        w, h = int(w * sx), int(h * sy)
                        x, y = max(0, x), max(0, y)
                        crop = frame[y:y + h, x:x + w]
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        if crop.size != 0:
                            face_crops.append(cv2.resize(crop, self.FACE_SIZE))
                            face_boxes.append((x, y, w, h))
                    break

        return face_crops, face_boxes

    # ------------------------------------------------------------------
    # Spatial analysis (per face crop)
    # ------------------------------------------------------------------
    @staticmethod
    def _laplacian_var(gray):
        try:
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            return float(lap.var())

    @staticmethod
    def _spatial_anomaly_score(face_rgb):
        """Explainable spatial forensics for a face crop.

        Returns (score, indicators, component_scores)
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

        # 4. Colour balance
        hist_r = cv2.calcHist([face_rgb], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([face_rgb], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([face_rgb], [2], None, [256], [0, 256]).flatten()
        hist_r = hist_r / (hist_r.sum() + 1e-10)
        hist_g = hist_g / (hist_g.sum() + 1e-10)
        hist_b = hist_b / (hist_b.sum() + 1e-10)
        colour_imbalance = (
            abs(hist_r.mean() - hist_g.mean())
            + abs(hist_g.mean() - hist_b.mean())
            + abs(hist_b.mean() - hist_r.mean())
        ) / 3.0
        colour_score = 0.0
        if colour_imbalance > 0.07:
            colour_score = 0.10
        elif colour_imbalance > 0.04:
            colour_score = 0.04

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
        if colour_score >= 0.10:
            indicators.append("Unnatural colour channel balance")
        elif colour_score >= 0.04:
            indicators.append("Slight colour channel imbalance")
        if chroma_score >= 0.14:
            indicators.append("Unnatural skin colour shifts")
        elif chroma_score >= 0.06:
            indicators.append("Slight skin-tone inconsistency")

        return float(np.clip(score, 0.0, 1.0)), indicators, components

    # ------------------------------------------------------------------
    # Whole-frame spatial analysis (fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _block_artifact_score(gray):
        h, w = gray.shape
        bs = 8
        block_vars = []
        for y in range(0, h - bs, bs):
            for x in range(0, w - bs, bs):
                block_vars.append(gray[y:y + bs, x:x + bs].var())
        if not block_vars:
            return 0.0, []
        block_vars = np.array(block_vars)
        mean_var = block_vars.mean()
        std_var = block_vars.std()
        score = 0.0
        inds = []
        if mean_var < 35:
            score += 0.16
            inds.append("Unnaturally uniform image regions (AI smoothing)")
        if std_var > 500:
            score += 0.14
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
        hist_r = cv2.calcHist([frame_rgb], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([frame_rgb], [1], None, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([frame_rgb], [2], None, [256], [0, 256]).flatten()
        hist_r = hist_r / (hist_r.sum() + 1e-10)
        hist_g = hist_g / (hist_g.sum() + 1e-10)
        hist_b = hist_b / (hist_b.sum() + 1e-10)
        colour_imbalance = (
            abs(hist_r.mean() - hist_g.mean())
            + abs(hist_g.mean() - hist_b.mean())
            + abs(hist_b.mean() - hist_r.mean())
        ) / 3.0
        if colour_imbalance > 0.07:
            score += 0.12
            inds.append("Unnatural overall colour balance")
        block_score, block_inds = DeepfakeEngine._block_artifact_score(gray)
        score += block_score
        inds.extend(block_inds)
        if not inds:
            inds.append("No strong whole-frame artifacts detected")
        return float(np.clip(score, 0.0, 1.0)), inds

    # ------------------------------------------------------------------
    # Temporal analysis (face crops)
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_anomaly_score(face_sequence):
        """Cross-frame coherence on face crops.

        Returns (score, indicators)
        """
        if len(face_sequence) < 3:
            return 0.0, []

        diffs = []
        prev_gray = None
        for face_rgb in face_sequence:
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if prev_gray is not None:
                diffs.append(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray

        if not diffs:
            return 0.0, []

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
        return float(np.clip(score, 0.0, 1.0)), inds

    @staticmethod
    def _whole_frame_temporal_score(frames_rgb):
        if len(frames_rgb) < 3:
            return 0.0, []
        diffs = []
        prev_gray = None
        for fr in frames_rgb:
            gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gray = cv2.resize(gray, (320, 240))
            if prev_gray is not None:
                diffs.append(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray
        if not diffs:
            return 0.0, []
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
        return float(np.clip(score, 0.0, 1.0)), inds

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
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self, video_path):
        extracted = self._extract_frames(video_path)
        frames = extracted["frames"]
        frame_indices = extracted["frame_indices"]
        fps = extracted["fps"]
        print("[*] Sampled", len(frames), "frames for analysis.")

        print("[*] Detecting faces...")
        all_face_seqs = []
        face_boxes_per_frame = []
        frames_with_faces = []
        for i, frame in enumerate(frames):
            faces, boxes = self._detect_faces(frame)
            if faces:
                all_face_seqs.append(faces[0])
                face_boxes_per_frame.append(boxes[0] if boxes else None)
                frames_with_faces.append(i)

        result = {
            "frames_analyzed": len(frames),
            "frames_with_faces": len(frames_with_faces),
            "frame_indices": frame_indices,
            "fps": fps,
            "metadata": self.video_metadata(video_path),
            "file_hash": self.file_hash(video_path),
        }

        # ---- Path A: face-based ----
        if len(all_face_seqs) >= 2:
            print("[*] Tracking", len(all_face_seqs), "face crops.")
            spatial_scores, spatial_inds, spatial_comps = [], [], []
            for f in all_face_seqs:
                s, inds, comps = self._spatial_anomaly_score(f)
                spatial_scores.append(s)
                spatial_inds.extend(inds)
                spatial_comps.append(comps)
            spatial_avg = float(np.mean(spatial_scores))
            temporal_score, temporal_inds = self._temporal_anomaly_score(
                all_face_seqs
            )
            combined = 0.60 * spatial_avg + 0.40 * temporal_score
            combined = float(np.clip(combined, 0.0, 1.0))
            result["analysis_mode"] = "face-based"
            result["spatial_score"] = spatial_avg
            result["temporal_score"] = temporal_score
            result["combined_score"] = combined
            result["indicators"] = spatial_inds + temporal_inds
            result["spatial_components"] = spatial_comps
            result["suspicious_frames"] = frames_with_faces

        # ---- Path B: whole-frame fallback ----
        else:
            print("[!] Few/no faces detected. Whole-frame forensic analysis.")
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            s_scores, s_inds = [], []
            for fr in frames_rgb:
                s, inds = self._whole_frame_spatial_score(fr)
                s_scores.append(s)
                s_inds.extend(inds)
            spatial_avg = float(np.mean(s_scores)) if s_scores else 0.0
            temporal_score, temporal_inds = self._whole_frame_temporal_score(
                frames_rgb
            )
            combined = 0.60 * spatial_avg + 0.40 * temporal_score
            combined = float(np.clip(combined, 0.0, 1.0))
            result["analysis_mode"] = "whole-frame"
            result["spatial_score"] = spatial_avg
            result["temporal_score"] = temporal_score
            result["combined_score"] = combined
            result["indicators"] = s_inds + temporal_inds
            result["suspicious_frames"] = list(range(len(frames)))

        # ---- Classification (three outcomes) ----
        if combined >= self.THRESH_MANIPULATED:
            classification = "LIKELY MANIPULATED"
        elif combined < self.THRESH_AUTHENTIC:
            classification = "LIKELY AUTHENTIC"
        else:
            classification = "INCONCLUSIVE"

        result["combined_score"] = float(np.clip(combined, 0.0, 1.0))
        result["classification"] = classification
        result["suspicion_percentage"] = round(combined * 100, 2)
        result["details"] = (
            "Spatial score: {:.2f} | Temporal score: {:.2f} | "
            "Combined: {:.2f} | Mode: {}".format(
                spatial_avg, temporal_score, combined, result.get("analysis_mode", "unknown")
            )
        )

        # Deduplicate indicators
        seen = set()
        unique = []
        for ind in result["indicators"]:
            if ind and ind not in seen:
                seen.add(ind)
                unique.append(ind)
        result["indicators"] = unique

        # Timestamp mapping for suspicious frames
        result["suspicious_timestamps"] = []
        for idx in result.get("suspicious_frames", [])[:5]:
            if idx < len(frame_indices):
                t = frame_indices[idx] / fps if fps > 0 else 0.0
                result["suspicious_timestamps"].append(round(t, 2))

        print("[*] Combined suspicion score: {:.4f}".format(combined))
        print("[*] Classification:", classification)
        return result
