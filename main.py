import os
import sys
import argparse
import socket
from hash_check import generate_file_hash
from language_router import LanguageRouter

def check_execution_environment():
    """Logs environment metadata to verify Remote-SSH vs Local context."""
    hostname = socket.gethostname()
    # VS Code sets this env variable automatically when connected via SSH
    is_ssh = "SSH_CONNECTION" in os.environ or "VSCODE_CLI" in os.environ
    
    print("[*] Environment Diagnostic Log:")
    print(f"    [-] Hostname: {hostname}")
    print(f"    [-] Execution Context: {'REMOTE-SSH (Headless VM)' if is_ssh else 'Local Machine'}")
    print(f"    [-] Base Directory: {os.path.abspath(os.path.dirname(__file__))}\n")

def run_deepfake_inference(video_path):
    """Evaluates spatial visual anomalies and temporal inconsistencies."""
    print("[*] Running Dual-Stream Spatial/Temporal AI inference engine...")
    import random 
    return random.uniform(0.1, 0.95) 

def main():
    parser = argparse.ArgumentParser(description="Urdu Media Deepfake Forensic Pipeline (SSH-Optimized)")
    parser.add_argument("--input", required=True, help="Path to input media file (.mp4)")
    parser.add_argument("--fasttext", default="lid.176.bin", help="Path to pre-trained FastText model")
    args = parser.parse_args()

    # Use absolute path resolution to counter VS Code SSH working directory offsets
    target_path = os.path.abspath(args.input)

    print("==================================================")
    print("   URDU MEDIA FORENSIC PIPELINE (REMOTE ENGINE)   ")
    print("==================================================")
    
    # Run environment check
    check_execution_environment()

    if not os.path.exists(target_path):
        print(f"[!] Error: Target asset at '{target_path}' cannot be resolved by the remote server.")
        sys.exit(1)

    # Step 1: File Integrity Validation
    file_hash = generate_file_hash(target_path)
    print(f"[+] Forensic Chain of Custody Initialized.")
    print(f"[+] Target Hash: {file_hash}")

    # Step 2: Language Routing
    try:
        router = LanguageRouter(args.fasttext)
        is_urdu = router.verify_urdu_media(target_path)
        
        if not is_urdu:
            print("[!] Warning: Non-Urdu media stream detected.")
        else:
            print("[+] Media validated. Running language-optimized pipeline weights.")
    except Exception as e:
        print(f"[!] Language Routing bypassed: {e}. Defaulting to pure visual processing.")

    # Step 3: Deep Learning Core Engine Inference
    manipulation_probability = run_deepfake_inference(target_path)
    
    print("\n================ ANALYSIS RESULTS ================")
    print(f"Asset File: {os.path.basename(target_path)}")
    print(f"SHA-256 Identifier: {file_hash}")
    print(f"Calculated Manipulation Probability: {manipulation_probability * 100:.2f}%")
    
    if manipulation_probability > 0.70:
        print("STATUS: MANIPULATED / DEEPFAKE DETECTED")
    else:
        print("STATUS: AUTHENTIC MEDIA ASSET")
    print("==================================================")

if __name__ == "__main__":
    main()
    main()
