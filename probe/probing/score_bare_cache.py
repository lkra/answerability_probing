#!/usr/bin/env python3
r"""Score a cache with the persisted directions in probe_directions.npz (no fitting), one output file
per probe, in the schema analyze_abstention.py joins on.

    python3 probe/probing/score_bare_cache.py --acts <cache> --ids-from <ids.jsonl> \
        --out probe/generation/qwen35-9b/probe_{probe}_<set>.jsonl
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import train_probes as tp
from export_probe_directions import project

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DIRECTIONS = REPO_ROOT / "probe/runs/qwen35-9b/bare/probe_directions.npz"


def load_directions(path):
    """npz -> (meta, {probe: {w, b, mu, sd}})."""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    probes = sorted({k.split("|", 1)[1] for k in d.files if k.startswith("w|")})
    params = {p: {k: d[f"{k}|{p}"] for k in ("w", "b", "mu", "sd")} for p in probes}
    for p in params:                                   # b is a 0-d array; project wants a scalar
        params[p]["b"] = float(params[p]["b"])
    return meta, params


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, required=True, help="activation cache directory")
    ap.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS)
    ap.add_argument("--probes", default=None,
                    help="comma-separated subset (default: every probe in the npz)")
    ap.add_argument("--ids-from", type=Path, default=None,
                    help="Restrict to the ids in this JSONL. Worth using: the bare translation "
                         "cache is 219,895 rows and materialising one layer of it is 3.6 GB, "
                         "against the ~16k rows the generation run actually covers.")
    ap.add_argument("--out", type=Path, required=True,
                    help="output path; must contain '{probe}' when scoring more than one")
    ap.add_argument("--keep-contaminated", action="store_true",
                    help="Keep rows in the organizer sample. Off by default, matching "
                         "probe_predictions.py: they are rows the shared task scores us on.")
    args = ap.parse_args()

    meta_d, params = load_directions(args.directions)
    probes = args.probes.split(",") if args.probes else list(params)
    unknown = [p for p in probes if p not in params]
    if unknown:
        raise SystemExit(f"no such probe in {args.directions.name}: {unknown} "
                         f"(has {list(params)})")
    if len(probes) > 1 and "{probe}" not in str(args.out):
        raise SystemExit("--out must contain '{probe}' when scoring several probes, or they "
                         "would overwrite each other")

    cfg, meta, acts, idx = tp.load_cache(args.acts, args.keep_contaminated)

    # The direction was fitted on activations from one representation. Applying it to another
    # produces a full set of plausible numbers describing nothing, so this is a hard stop
    # rather than a warning -- the same check `train_probe_multi.check_compatible` makes.
    for key in ("model", "pooling", "hidden_size"):
        if cfg.get(key) != meta_d.get(key, cfg.get(key)):
            raise SystemExit(f"{key} mismatch: cache {cfg.get(key)!r} vs direction "
                             f"{meta_d.get(key)!r} -- different representations, do not pool")
    if cfg.get("prompt_style") != meta_d["prompt_style"]:
        print(f"NOTE: cache prompt_style {cfg.get('prompt_style')!r} != direction "
              f"{meta_d['prompt_style']!r}. Deliberate for the shipped-probe arm; see the "
              f"docstring, and say which arm the number came from.")

    if args.ids_from:
        want = {json.loads(l)["id"] for l in open(args.ids_from, encoding="utf-8")}
        sel = np.array([i for i, m in enumerate(meta) if m["id"] in want])
        if not len(sel):
            raise SystemExit(f"--ids-from {args.ids_from} selected no rows of {args.acts}")
        # Almost all of the shortfall is normally contamination, not absence: the wanted ids
        # are in the cache, `load_cache` dropped them as organizer-sample rows before we got
        # here. Reporting the two together reads as "the caches do not line up", which is a
        # different and much more alarming problem, so separate them.
        # against the cache's *unfiltered* ids -- `meta` here has already lost the dropped rows
        absent = want - {json.loads(l)["id"]
                         for l in open(args.acts / "meta.jsonl", encoding="utf-8")}
        dropped = (len(want) - len(sel)) - len(absent)
        print(f"--ids-from {args.ids_from.name}: {len(meta):,} -> {len(sel):,} rows"
              + (f", {dropped:,} excluded as organizer sample" if dropped else "")
              + (f", {len(absent):,} NOT IN THIS CACHE" if absent else ""))
        meta = [meta[i] for i in sel]
        idx = idx[sel]

    layer = meta_d["layer"]
    col = tp.layer_column(cfg, layer)
    y = np.array([m["answerable"] for m in meta])
    print(f"{len(meta):,} rows, layer {layer}, {y.mean():.1%} answerable  <- {args.acts}")

    # Chunked so peak memory is the chunk, not the cache: `idx` is a fancy index into a
    # memmap, and doing it in one call materialises every selected row at once.
    X = np.empty((len(idx), cfg["hidden_size"]), dtype=np.float32)
    for i in range(0, len(idx), 20000):
        X[i:i + 20000] = np.asarray(acts[idx[i:i + 20000], col, :], dtype=np.float32)

    for probe in probes:
        z = project(params[probe], X)
        p = 1.0 / (1.0 + np.exp(-z))
        out = Path(str(args.out).replace("{probe}", probe.replace("+", "_")))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for m, pi in zip(meta, p):
                f.write(json.dumps({
                    "id": m["id"], "lang_code": m["lang_code"],
                    "passage_lang": m.get("passage_lang", m["lang_code"]),
                    "question_lang": m.get("question_lang", m["lang_code"]),
                    "answerable": m["answerable"], "prob_answerable": float(pi),
                    "layer": layer, "probe": probe,
                }, ensure_ascii=False) + "\n")

        both = len(set(y)) == 2
        print(f"\n{probe}: AUROC {tp.auroc(p, y):.4f}  acc {tp.accuracy(p, y):.4f}"
              if both else f"\n{probe}: one class only, AUROC undefined")
        by = collections.defaultdict(list)
        for i, m in enumerate(meta):
            by[m["lang_code"]].append(i)
        for lang in sorted(by):
            ii = np.array(by[lang])
            ok = len(set(y[ii])) == 2
            print(f"  {lang:10s} {len(ii):6,d} "
                  + (f"AUROC {tp.auroc(p[ii], y[ii]):.4f}" if ok else "AUROC n/a (one class)"))
        print(f"  wrote {out}  ({len(meta):,} rows)")


if __name__ == "__main__":
    main()
