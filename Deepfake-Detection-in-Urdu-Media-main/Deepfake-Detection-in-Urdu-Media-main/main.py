"""
TRACE CLI - Deepfake Detection Command Line Interface
-------------------------------------------------------------
Student-level CLI for forensic video analysis.

Usage:
    python main.py --input path/to/video.mp4
    python main.py --input path/to/video.mp4 --fasttext /path/to/lid.176.bin

Outputs a suspicion score (not a probability) and a three-way classification.
-------------------------------------------------------------
"""

import os
import sys
import argparse
import socket
from hash_check import generate_file_hash
from language_router import LanguageRouter
from deepfake_engine import DeepfakeEngine


def check_execution_environment():
    """Logs environment metadata."""
    hostname = socket.gethostname()
    is_ssh = "SSH_CONNECTION" in os.environ or "VSCODE_CLI" in os.environ

    print("[*] Environment Diagnostic Log:")
    print("    [-] Hostname: " + hostname)
    print("    [-] Execution Context: " +
          ("REMOTE-SSH (Headless VM)" if is_ssh else "Local Machine"))
    print("    [-] Base Directory: " +
          os.path.abspath(os.path.dirname(__file__)) + "\n")


def run_deepfake_inference(video_path):
    """Runs the forensic analysis engine and returns the result dict."""
    print("[*] Running TRACE spatial/temporal forensic analysis...")
    engine = DeepfakeEngine()
    return engine.analyze(video_path)


def main():
    parser = argparse.ArgumentParser(
        description="TRACE Deepfake Forensic Pipeline (Student Edition)"
    )
    parser.add_argument("--input", required=True, help="Path to input media file (.mp4)")
    parser.add_argument("--fasttext", default="lid.176.bin",
                        help="Path to pre-trained FastText model")
    args = parser.parse_args()

    target_path = os.path.abspath(args.input)

    print("==================================================")
    print("   TRACE DEEPFAKE FORENSIC PIPELINE (STUDENT)   ")
    print("==================================================")

    check_execution_environment()

    if not os.path.exists(target_path):
        print("[!] Error: Target file at '" + target_path + "' cannot be resolved.")
        sys.exit(1)

    # Step 1: File Integrity Validation
    file_hash = generate_file_hash(target_path)
    print("[+] Forensic Chain of Custody Initialized.")
    print("[+] Target Hash: " + str(file_hash))

    # Step 2: Language Routing (optional)
    try:
        router = LanguageRouter(args.fasttext)
        lang_result = router.verify_media(target_path)
        print("[+] Language status: " + str(lang_result.get("language_status", "unknown")))
    except Exception as e:
        print("[!] Language routing bypassed: " + str(e) +
              ". Defaulting to visual processing.")

    # Step 3: Deepfake forensic analysis
    result = run_deepfake_inference(target_path)

    # BUG FIX #5: result is a dict, not a number
    # BUG FIX #6: use shared classification function
    score = result.get("suspicion_score")
    classification = result.get("classification", "INSUFFICIENT_EVIDENCE")

    print("\n================ ANALYSIS RESULTS ================")
    print("Asset File: " + os.path.basename(target_path))
    print("SHA-256 Identifier: " + str(file_hash))
    # BUG FIX #7: report as suspicion score, not probability
    if score is not None:
        print("Suspicion Score: {:.2f}%".format(score * 100))
    else:
        print("Suspicion Score: N/A (insufficient evidence)")
    print("Classification: " + classification)
    print("Analysis Mode: " + result.get("analysis_mode", "unknown"))
    print("Frames Analyzed: " + str(result.get("frames_analyzed", 0)))
    print("Frames with Faces: " + str(result.get("frames_with_faces", 0)))

    indicators = result.get("indicators", [])
    if indicators:
        print("\nForensic Indicators Found:")
        for ind in indicators:
            print("  - " + ind)
    else:
        print("\nNo specific forensic indicators flagged.")

    print("==================================================")


if __name__ == "__main__":
    main()
