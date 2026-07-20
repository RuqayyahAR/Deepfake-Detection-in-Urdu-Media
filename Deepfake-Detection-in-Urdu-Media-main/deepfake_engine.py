import cv2
import numpy as np
import torch
from mtcnn import MTCNN


class DeepfakeEngine:
    """Dual-stream spatial/temporal deepfake detection engine.

    Spatial stream: analyzes individual frames for facial inconsistencies.
    Temporal stream: analyzes frame-to-frame temporal coherence.
    Both streams produce anomaly scores that are combined into a final
    manipulation probability.
    """

    FRAME_SAMPLE_COUNT = 30
    FACE_SIZE = (160, 160)

    def __init__(self):
        self.detector = MTCNN(
            keep_all=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            min_face_size=40,
        )

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------
    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames.")

        indices = np.linspace(0, total_frames - 1, self.FRAME_SAMPLE_COUNT, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    # ------------------------------------------------------------------
    # Face detection & alignment
    # ------------------------------------------------------------------
    def _detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)
        faces = []
        for det in detections:
            if det["confidence"] < 0.90:
                continue
            x, y, w, h = det["box"]
            x, y = max(0, x), max(0, y)
            face_crop = rgb[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue
            face_resized = cv2.resize(face_crop, self.FACE_SIZE)
            faces.append(face_resized)
        return faces

    # ------------------------------------------------------------------
    # Spatial analysis  (per-frame)
    # ------------------------------------------------------------------
    @staticmethod
    def _spatial_anomaly_score(face_rgb):
        """Compute anomaly indicators for a single face crop.

        Metrics:
        - Laplacian variance (blur / unnatural smoothness)
        - Frequency-domain energy ratio (high-freq vs low-freq)
        - Edge density (Canny)
        - Colour-space statistics (skin-tone consistency proxy)
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Laplacian variance -- deepfakes often have abnormally low or
        #    high variance due to blending artifacts.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. FFT energy ratio -- unnatural frequency patterns.
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        total_energy = magnitude.sum() + 1e-10
        y_coords, x_coords = np.ogrid[:h, :w]
        mask = ((y_coords - cy) ** 2 + (x_coords - cx) ** 2) <= radius ** 2
        low_freq_energy = magnitude[mask].sum()
        high_freq_ratio = 1.0 - (low_freq_energy / total_energy)

        # 3. Edge density via Canny.
        edges = cv2.Canny(face_rgb, 100, 200)
        edge_density = edges.mean() / 255.0

        # 4. Skin-tone chrominance consistency (YCbCr).
        ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
        cb_std = ycrcb[:, :, 1].astype(np.float32).std()
        cr_std = ycrcb[:, :, 2].astype(np.float32).std()
        chroma_dispersion = (cb_std + cr_std) / 2.0

        # Combine into a single anomaly score [0, 1].
        # Thresholds are heuristic baselines calibrated on general face data.
        score = 0.0
        # Unnaturally smooth faces (low laplacian) can indicate synthesis.
        if laplacian_var < 80:
            score += 0.3
        elif laplacian_var < 200:
            score += 0.15
        # Excessive high-frequency content can indicate blending seams.
        if high_freq_ratio > 0.75:
            score += 0.25
        elif high_freq_ratio > 0.60:
            score += 0.10
        # Low edge density suggests smearing / averaging artifacts.
        if edge_density < 0.05:
            score += 0.25
        elif edge_density < 0.10:
            score += 0.10
        # High chroma dispersion may indicate unnatural colour shifts.
        if chroma_dispersion > 18:
            score += 0.20
        elif chroma_dispersion > 12:
            score += 0.10

        return np.clip(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Temporal analysis  (across frames)
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_anomaly_score(face_sequence):
        """Analyse temporal coherence across consecutive face crops.

        Deepfakes often exhibit:
        - Excessive jitter (face warping flicker)
        - Unnaturally stable regions (static inpainting)
        - Optical-flow irregularities
        """
        if len(face_sequence) < 3:
            return 0.0

        diffs = []
        prev_gray = None
        for face_rgb in face_sequence:
            gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if prev_gray is not None:
                frame_diff = cv2.absdiff(gray, prev_gray)
                diffs.append(frame_diff.mean())
            prev_gray = gray

        if not diffs:
            return 0.0

        diff_arr = np.array(diffs)
        mean_diff = diff_arr.mean()
        std_diff = diff_arr.std()

        score = 0.0
        # High inter-frame jitter is suspicious.
        if std_diff > 15:
            score += 0.35
        elif std_diff > 8:
            score += 0.15
        # Extremely low mean difference suggests static inpainting.
        if mean_diff < 1.0:
            score += 0.30
        elif mean_diff < 3.0:
            score += 0.15
        # Ratio of near-zero frames to moving frames.
        static_ratio = (diff_arr < 1.0).mean()
        if static_ratio > 0.5:
            score += 0.25
        elif static_ratio > 0.3:
            score += 0.10

        return np.clip(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, video_path):
        """Run the full dual-stream pipeline on *video_path*.

        Returns a float in [0, 1] representing the manipulation probability.
        """
        print("[*] Extracting sampled frames from video...")
        frames = self._extract_frames(video_path)
        print(f"[*] Sampled {len(frames)} frames for analysis.")

        print("[*] Detecting faces with MTCNN...")
        all_face_seqs = []
        for frame in frames:
            faces = self._detect_faces(frame)
            if faces:
                all_face_seqs.append(faces[0])  # primary face

        if not all_face_seqs:
            print("[!] No faces detected across sampled frames. Returning neutral score 0.50")
            return 0.50

        print(f"[*] Tracking {len(all_face_seqs)} face crops across frames.")

        # Spatial stream
        spatial_scores = [self._spatial_anomaly_score(f) for f in all_face_seqs]
        spatial_avg = float(np.mean(spatial_scores))

        # Temporal stream
        temporal_score = self._temporal_anomaly_score(all_face_seqs)

        # Weighted combination
        final = 0.55 * spatial_avg + 0.45 * temporal_score
        final = np.clip(final, 0.0, 1.0)

        print(f"[*] Spatial anomaly score:  {spatial_avg:.4f}")
        print(f"[*] Temporal anomaly score: {temporal_score:.4f}")
        print(f"[*] Combined manipulation probability: {final:.4f}")

        return float(final)
