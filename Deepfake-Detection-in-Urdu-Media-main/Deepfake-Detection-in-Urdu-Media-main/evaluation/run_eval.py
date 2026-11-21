"""
Run a detector on one split of a labelled manifest and report AUC / EER with 95% CIs.

  python -m evaluation.run_eval --manifest set.csv --base /path/to/videos --detector heuristic --split test
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detectors
import manifest
import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to labelled CSV manifest")
    ap.add_argument(
        "--base",
        default=".",
        help="Base directory for video paths in the manifest",
    )
    ap.add_argument(
        "--detector",
        required=True,
        help="'heuristic' or module:Class",
    )
    ap.add_argument(
        "--engine-dir",
        default=PROJECT_ROOT,
        help="Directory containing deepfake_engine.py",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    v = manifest.validate(args.manifest, args.base)
    if not v["ok"]:
        print(json.dumps(v, indent=2))
        raise SystemExit("manifest has errors - fix them before evaluating")

    rows = [
        r
        for r in csv.DictReader(open(args.manifest, encoding="utf-8"))
        if r["split"] == args.split
    ]

    det = detectors.load(args.detector, args.engine_dir)

    y, s, langs = [], [], []
    for r in rows:
        y.append(1 if r["label"] == "fake" else 0)
        s.append(det.score(os.path.join(args.base, r["path"])))
        langs.append(r["language"])

    rep = metrics.report(y, s)
    rep.update(
        {
            "detector": det.name,
            "split": args.split,
            "warnings": v["warnings"],
            "per_language": {},
        }
    )

    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        yy, ss = [y[i] for i in idx], [s[i] for i in idx]
        if len(set(yy)) == 2:
            rep["per_language"][lang] = metrics.report(yy, ss, n_boot=500)
        else:
            rep["per_language"][lang] = {"skipped": "needs both REAL and FAKE videos"}

    json.dump(rep, open(args.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
