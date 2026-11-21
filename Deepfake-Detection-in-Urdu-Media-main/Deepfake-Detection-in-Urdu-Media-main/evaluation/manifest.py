"""
Validate a labelled test-set manifest (CSV) before any metric is computed.

Required columns:
  path, label (real|fake), split (train|val|test), source (dataset name),
  language (en|hi|ur|te|none), manipulation (none|faceswap|lipsync|synthesis|edit|other),
  licence (text), sha256 (optional: checked against the file if present)

Checks: file exists, label/split/language values valid, hashes match,
no identical file in two splits (leakage), and class balance per split.
"""
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict

VALID = {
    "label": {"real", "fake"},
    "split": {"train", "val", "test"},
    "language": {"en", "hi", "ur", "te", "none"},
    "manipulation": {"none", "faceswap", "lipsync", "synthesis", "edit", "other"},
}
REQUIRED = ["path", "label", "split", "source", "language", "manipulation", "licence"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(csv_path, base_dir="."):
    errors, rows = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            return {"ok": False, "errors": [f"missing columns: {missing}"]}
        for i, r in enumerate(reader, start=2):
            rows.append((i, r))
    hashes = {}
    for i, r in rows:
        for col, allowed in VALID.items():
            if r[col] not in allowed:
                errors.append(f"line {i}: {col}='{r[col]}' not in {sorted(allowed)}")
        p = os.path.join(base_dir, r["path"])
        if not os.path.exists(p):
            errors.append(f"line {i}: file not found: {r['path']}")
            continue
        h = sha256(p)
        if r.get("sha256") and r["sha256"].lower() != h:
            errors.append(f"line {i}: sha256 mismatch for {r['path']}")
        hashes.setdefault(h, []).append((i, r["split"], r["path"]))
        r["_sha"] = h
    for h, locs in hashes.items():
        splits = {s for _, s, _ in locs}
        if len(locs) > 1 and len(splits) > 1:
            errors.append(
                "leakage: identical file in splits " + str(sorted(splits)) + " -> " + ", ".join(p for _, _, p in locs)
            )
    balance = defaultdict(Counter)
    for _, r in rows:
        balance[r["split"]][r["label"]] += 1
    lang = Counter((r["split"], r["language"]) for _, r in rows)
    warnings = []
    for split in ("train", "val", "test"):
        c = balance.get(split, Counter())
        if c and (c["real"] == 0 or c["fake"] == 0):
            warnings.append(f"split '{split}' has only one class: {dict(c)}")
    if balance.get("test", Counter())["real"] < 30 or balance.get("test", Counter())["fake"] < 30:
        warnings.append("test split has fewer than 30 videos of a class: CIs will be wide")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "videos": len(rows),
        "balance": {k: dict(v) for k, v in balance.items()},
        "languages": {f"{s}/{l}": n for (s, l), n in sorted(lang.items())},
    }


if __name__ == "__main__":
    res = validate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
    import json

    print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)
