# Automated Deepfake Detection in Urdu Media

An ongoing research-driven AI project focused on combating digital misinformation in regional broadcasting. This system leverages deep learning models to analyze video and audio streams, specifically optimized to detect synthetic manipulations and deepfakes within Urdu media contexts where traditional western-centric datasets lack representative coverage.

---

## 🧐 Problem Statement & Context

Most state-of-the-art deepfake detection models are trained heavily on high-definition datasets consisting primarily of English speakers (e.g., FaceForensics++, DFDC). When deployed on regional broadcasts, these models experience significant performance degradation due to:
1. **Low-resolution feeds & broadcast compression artifacts** typical in regional television.
2. **Unique linguistic patterns, lip sync mechanics, and phonetic traits** inherent to the Urdu language.
3. **Cultural variations** in lighting, studio environments, and traditional attire affecting facial landmarking.

This project aims to bridge that gap by building an optimized preprocessing and classification pipeline tailored to Urdu news anchors, political broadcasts, and digital media.

---

## 🏗️ System Pipeline Architecture

The system processes video files through a multi-stage sequential execution pipeline:

1. **Video Ingestion & Frame Extraction:** Video streams are decoded using OpenCV and broken down into temporal frame sequences.
2. **Facial Detection & ROI Extraction:** MTCNN (Multi-task Cascaded Convolutional Networks) crops and aligns faces, isolating the Region of Interest (ROI) while filtering out background broadcast graphics.
3. **Sequence Sequence Chunking:** Continuous frames are batched into fixed temporal segments to preserve motion vectors and audio-visual sync data.
4. **Deep Learning Inference Engine:** Dual-stream architecture analyzing spatial visual inconsistencies (frame-by-frame) and temporal artifacts (fluidity across time).
5. **Probability Output:** Generates a localized confidence score indicating whether the media asset is "Authentic" or "Manipulated".

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch / TensorFlow, Keras
* **Computer Vision:** OpenCV, MTCNN, Facenet
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Performance Tools:** FFmpeg (for rapid video demuxing)

---

## 🌟 Key Features

* **Multi-Modal Visual Analysis:** Inspects spatial anomalies (such as facial blending artifacts, asymmetric blending, or unnatural blinking) and temporal inconsistencies across consecutive video frames.
* **Regional Optimization:** Tailored features to account for varying broadcast quality, stream-compression ratios, and specific low-light conditions unique to mainstream regional television packages.
* **Custom Preprocessing Pipeline:** Robust face-tracking and alignment that disregards ticker-tape text, station logos, and picture-in-picture broadcasts common in news streams.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have a GPU environment configured with CUDA support for viable inference and training times.

### Installation

1. **Clone the repository:**
```bash
   git clone [https://github.com/YOUR_USERNAME/urdu-deepfake-detection.git](https://github.com/YOUR_USERNAME/urdu-deepfake-detection.git)
   cd urdu-deepfake-detection
