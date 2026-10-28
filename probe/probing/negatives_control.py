#!/usr/bin/env python3
"""AUROC of both persisted probe directions on BELEBELE-8 with deleted-sentence negatives (belebele8
cache, monolingual diagonal) versus swap negatives (belebele-hard cache): same 128 items, 8
languages, 2,048 rows, identical answerable rows. No fitting.

Output: probe/runs/qwen35-9b/bare/negatives_control.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_probes import auroc, layer_column, load_cache, overlap_feature   # noqa: E402

# The jsonl the caches were built from, needed for the lexical bar -- the caches keep
# activations and labels but not the text.
SOURCES = {
    "delete": REPO_ROOT / "data/built/belebele_answerability_delete_perlang.jsonl",
    "hard": REPO_ROOT / "data/built/belebele_answerability_hard_b8.jsonl",
}

PROBES = ("tydi", "tydi+udhr")
DEFAULT_RUN = REPO_ROOT / "probe/runs/qwen35-9b/bare"
CACHES = REPO_ROOT / "probe/caches/qwen35-9b/bare"


def load(cache, layer, mono_only):
    """(X at `layer`, y, passage language, item) for one activation cache.

    Goes through `train_probes.load_cache` rather than reading the files directly: the caches
    are raw float16 memmaps whose shape lives in `config.json`, not `.npy` files, and the
    same helper applies the organizer-sample exclusion every other script uses.
    """
    cfg, meta, acts, idx = load_cache(cache, keep_contaminated=False)
    col = layer_column(cfg, layer)
    if mono_only:
        sel = np.flatnonzero([m["passage_lang"] == m["question_lang"] for m in meta])
        meta = [meta[i] for i in sel]
        idx = idx[sel]
    X = np.asarray(acts[idx, col, :], dtype=np.float64)
    y = np.array([m["answerable"] for m in meta])
    lang = np.array([m["passage_lang"] for m in meta])
    item = np.array([m["item_id"] for m in meta])
    return X, y, lang, item


def lexical_bar(path, items, langs):
    """Lexical-overlap AUROC and mean overlap per class on the same rows, from lines 0 and 1 (passage,
    question) of BELEBELE's six-line input.
    """
    ov, y = [], []
    for line in open(path):
        r = json.loads(line)
        if not (r["item_id"] in items and r["passage_lang"] in langs
                and r["passage_lang"] == r["question_lang"]):
            continue
        parts = r["input"].split("\n")
        ov.append(overlap_feature(parts[1], parts[0], r["passage_lang"]))
        y.append(r["answerable"])
    ov, y = np.array(ov), np.array(y)
    return {"auroc": float(auroc(ov, y)), "n": int(len(y)),
            "mean_overlap_answerable": float(ov[y == 1].mean()),
            "mean_overlap_unanswerable": float(ov[y == 0].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete-cache", type=Path, default=CACHES / "belebele8")
    ap.add_argument("--hard-cache", type=Path, default=CACHES / "belebele-hard")
    ap.add_argument("--directions", type=Path, default=DEFAULT_RUN / "probe_directions.npz")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--out", type=Path, default=DEFAULT_RUN / "negatives_control.json")
    args = ap.parse_args()

    for p in (args.delete_cache, args.hard_cache, args.directions):
        if not p.exists():
            raise SystemExit(f"{p} not found"
                             + (" -- run probe/runs/run_9b_hard_negatives.sh"
                                if p is args.hard_cache else ""))

    d = np.load(args.directions, allow_pickle=False)
    sets = {"delete": load(args.delete_cache, args.layer, mono_only=True),
            "hard": load(args.hard_cache, args.layer, mono_only=True)}

    # The comparison is only meaningful on shared items; report if they are not identical.
    items = {k: set(v[3].tolist()) for k, v in sets.items()}
    shared = items["delete"] & items["hard"]
    if items["delete"] != items["hard"]:
        print(f"note: item sets differ -- delete {len(items['delete'])}, hard {len(items['hard'])},"
              f" shared {len(shared)}; restricting to shared")
    for k, (X, y, lang, item) in sets.items():
        m = np.isin(item, list(shared))
        sets[k] = (X[m], y[m], lang[m], item[m])

    res = {"layer": args.layer, "n_items": len(shared), "sets": {}, "by_language": {}}
    print(f"layer {args.layer} | {len(shared)} shared items | "
          f"delete {len(sets['delete'][1]):,} rows, hard {len(sets['hard'][1]):,} rows\n")
    print(f"{'probe':12s} {'negatives':10s} {'AUROC':>7s}   per-language range")
    for probe in PROBES:
        w, mu, sd, b = (d[f"w|{probe}"], d[f"mu|{probe}"], d[f"sd|{probe}"], d[f"b|{probe}"])
        res["sets"][probe] = {}
        for kind, (X, y, lang, _) in sets.items():
            z = ((X - mu) / sd) @ w + b
            per = {g: auroc(z[lang == g], y[lang == g]) for g in sorted(set(lang.tolist()))}
            res["sets"][probe][kind] = {"auroc": float(auroc(z, y)), "n": int(len(y)),
                                        "per_language": {k: float(v) for k, v in per.items()}}
            lo, hi = min(per, key=per.get), max(per, key=per.get)
            print(f"{probe:12s} {kind:10s} {res['sets'][probe][kind]['auroc']:7.4f}   "
                  f"{lo} {per[lo]:.3f} .. {hi} {per[hi]:.3f}")
        a = res["sets"][probe]["hard"]["auroc"] - res["sets"][probe]["delete"]["auroc"]
        res["sets"][probe]["hard_minus_delete"] = float(a)
        print(f"{'':12s} {'delta':10s} {a:+7.4f}   (hard - delete)\n")

    # The lexical bar, without which a raw AUROC difference says nothing about the
    # representation: if the bar moves as much as the probe, the construction changed how much
    # surface overlap gives away, not how hard answerability is to read.
    langs = set(sets["delete"][2].tolist())
    res["lexical"] = {k: lexical_bar(SOURCES[k], shared, langs) for k in ("delete", "hard")}
    print()
    for k, v in res["lexical"].items():
        print(f"lexical bar, {k:6s} AUROC {v['auroc']:.4f}  "
              f"(overlap: answerable {v['mean_overlap_answerable']:.3f}, "
              f"unanswerable {v['mean_overlap_unanswerable']:.3f})")
    dbar = res["lexical"]["hard"]["auroc"] - res["lexical"]["delete"]["auroc"]
    res["lexical"]["hard_minus_delete"] = float(dbar)
    print(f"lexical bar moves {dbar:+.4f}")
    print("\nmargin over the bar (probe AUROC - lexical AUROC):")
    for probe in PROBES:
        m = {k: res["sets"][probe][k]["auroc"] - res["lexical"][k]["auroc"]
             for k in ("delete", "hard")}
        res["sets"][probe]["margin"] = {k: float(v) for k, v in m.items()}
        res["sets"][probe]["margin"]["hard_minus_delete"] = float(m["hard"] - m["delete"])
        print(f"  {probe:12s} delete {m['delete']:+.3f}   hard {m['hard']:+.3f}   "
              f"change {m['hard'] - m['delete']:+.3f}")

    # Per language, so a single language cannot drive the aggregate.
    for probe in PROBES:
        pl = {g: (res["sets"][probe]["hard"]["per_language"][g]
                  - res["sets"][probe]["delete"]["per_language"][g])
              for g in res["sets"][probe]["delete"]["per_language"]}
        res["by_language"][probe] = pl
        print(f"{probe} per-language (hard - delete): "
              + "  ".join(f"{g} {v:+.3f}" for g, v in sorted(pl.items(), key=lambda kv: kv[1])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
