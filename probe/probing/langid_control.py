#!/usr/bin/env python3
"""One-vs-rest logistic regression predicting the question language from the same pooled vectors the
probe reads (80/20 split, AUROC meaned over languages), at layers 19 and 20.

Output: probe/runs/qwen35-9b/bare/langid_control.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_probes import auroc, fit_logreg, layer_column, load_cache   # noqa: E402
from _device import pick_device                                        # noqa: E402

CACHES = REPO_ROOT / "probe/caches/qwen35-9b/bare"
# (label, cache, which language field to decode)
CELLS = [("TyDi train", "train", "lang_code"),
         ("UDHR, question lg.", "udhr", "question_lang"),
         ("UDHR, passage lg.", "udhr", "passage_lang")]


def langid(X, lang, device, l2, seed):
    """Mean one-vs-rest AUROC for decoding the language, plus the per-language spread."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(lang))
    cut = int(0.8 * len(lang))
    tr, te = perm[:cut], perm[cut:]
    per = {}
    for lg in sorted(set(lang.tolist())):
        y = (lang == lg).astype(int)
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue          # a language absent from one side has no AUROC to report
        per[lg] = float(auroc(fit_logreg(X[tr], y[tr], X[te], device, l2=l2), y[te]))
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", type=Path, default=CACHES)
    ap.add_argument("--layers", default="19,20", help="comma-separated model-layer indices")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "probe/runs/qwen35-9b/bare/langid_control.json")
    args = ap.parse_args()
    device = pick_device(args.device)
    layers = [int(x) for x in args.layers.split(",")]

    out = {"l2": args.l2, "seed": args.seed, "cells": {}}
    print(f"one-vs-rest language decoding, L2={args.l2}, 80/20 split\n")
    print(f"{'set':22s} {'layer':>5s} {'langs':>6s} {'rows':>8s} {'mean AUROC':>11s} {'min':>7s}")
    for label, cache, field in CELLS:
        d = args.caches / cache
        if not d.exists():
            print(f"{label:22s} -- cache {d} not found, skipped")
            continue
        cfg, meta, acts, idx = load_cache(d, keep_contaminated=False)
        lang = np.array([m[field] for m in meta])
        for layer in layers:
            X = np.asarray(acts[idx, layer_column(cfg, layer), :], dtype=np.float64)
            per = langid(X, lang, device, args.l2, args.seed)
            vals = np.array(list(per.values()))
            key = f"{cache}|{field}|L{layer}"
            out["cells"][key] = {"label": label, "layer": layer, "n_rows": int(len(lang)),
                                 "n_languages": len(per), "mean_auroc": float(vals.mean()),
                                 "min_auroc": float(vals.min()),
                                 "min_language": min(per, key=per.get),
                                 "per_language": per}
            print(f"{label:22s} {layer:5d} {len(per):6d} {len(lang):8,d} "
                  f"{vals.mean():11.4f} {vals.min():7.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
