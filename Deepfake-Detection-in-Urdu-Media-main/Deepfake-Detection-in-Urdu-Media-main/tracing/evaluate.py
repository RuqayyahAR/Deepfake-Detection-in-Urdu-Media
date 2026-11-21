"""
Evaluation harness for source tracing.

Runs retrieval + segment accuracy + unseen-control check. Writes results.json.

Usage:
    python -m tracing.evaluate
"""
import glob
import os
import json
import numpy as np
from .fingerprint import video_hashes, audio_hashes
from .align import ham, best_offset, trace, VIDEO_T


def load(p):
    return video_hashes(p), audio_hashes(p)


def run_evaluation(refs_dir="data/refs", suspects_dir="data/suspects",
                   target="ref0", output="results.json"):
    """Run full evaluation and return results list."""
    ref_paths = sorted(glob.glob(os.path.join(refs_dir, "*.mp4")))
    refs = {os.path.basename(p)[:-4]: load(p) for p in ref_paths}

    # ---- calibration on NON-matching pairs only (never on the target)
    dists = []
    names = [n for n in refs if n != target]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            _, d = best_offset(refs[a][0][:20], refs[b][0])
            dists.append(d)
    dists = np.array(dists)
    print("non-matching pair video distance: min %.3f  median %.3f  p5 %.3f" %
          (dists.min(), np.median(dists), np.percentile(dists, 5)))

    # ---- ground truth per variant: which second-level label is expected
    def gt(name):
        n = 20
        if name.startswith("recomp") or name.startswith("trim"):
            return {}                              # everything MATCH
        if name.startswith("splice"):
            return {s: "VIDEO_ALTERED" for s in (10, 11, 12)}
        if name.startswith("region"):
            return {s: "VIDEO_ALTERED" for s in range(6, 12)}
        if name.startswith("audio"):
            return {s: "AUDIO_ALTERED" for s in range(20)}
        return None

    suspect_paths = sorted(glob.glob(os.path.join(suspects_dir, "*.mp4")))
    suspect_paths.append(os.path.join(os.path.dirname(suspects_dir),
                                      "unseen_control.mp4"))

    rows = []
    for p in suspect_paths:
        name = os.path.basename(p)[:-4]
        sv, sa = load(p)
        scores = {}
        for rn, (rv, ra) in refs.items():
            off, d = best_offset(sv, rv)
            scores[rn] = d
        best_ref = min(scores, key=scores.get)
        found = scores[best_ref] <= VIDEO_T
        correct_ref = (best_ref == target) and found
        res = trace(sv, sa, *refs[target])
        if name == "trim_5s_8s":
            print("   trim audio offset:", res["audio_offset_seconds"],
                  "video offset:", res["video_offset_seconds"])
        expected = gt(name)
        seg_acc = None
        if expected is not None and res.get("segments"):
            correct = 0
            for seg in res["segments"]:
                want = expected.get(seg["t"], "MATCH")
                correct += (seg["label"] == want)
            seg_acc = correct / len(res["segments"])
        rows.append({
            "variant": name,
            "retrieved_ref": best_ref if found else "NONE",
            "correct_original": bool(correct_ref),
            "best_distance": round(float(scores[best_ref]), 3),
            "offset_s": res.get("offset_seconds"),
            "segment_accuracy": None if seg_acc is None else round(float(seg_acc), 2),
            "labels": "".join(
                {"MATCH": ".", "VIDEO_ALTERED": "V", "AUDIO_ALTERED": "A",
                 "BOTH_ALTERED": "B"}[s["label"]]
                for s in res.get("segments", [])
            ),
        })

    print()
    print("%-26s %-9s %-8s %-8s %-7s %-9s %s" %
          ("variant", "retrieved", "correct", "dist", "offset", "seg_acc",
           "per-second labels (. match V video A audio)"))
    for r in rows:
        print("%-26s %-9s %-8s %-8s %-7s %-9s %s" %
              (r["variant"], r["retrieved_ref"], r["correct_original"],
               r["best_distance"], r["offset_s"], r["segment_acc"],
               r["labels"]))

    with open(output, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nResults written to", output)
    return rows
