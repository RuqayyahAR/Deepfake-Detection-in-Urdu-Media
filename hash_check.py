import hashlib

def generate_file_hash(file_path):
    """Generates a forensic SHA-256 hash for metadata validation."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read file in 64KB blocks to optimize memory allocation
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"[!] Forensic Failure: Asset not found at {file_path}")
        return None

if __name__ == "__main__":
    # Quick sanity check
    test_hash = generate_file_hash("test_broadcast.mp4")
    if test_hash:
        print(f"[+] Forensic SHA-256: {test_hash}")
