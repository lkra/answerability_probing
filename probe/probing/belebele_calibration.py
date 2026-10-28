#!/usr/bin/env python3
"""Per-language offset spread of both probes on BELEBELE, which neither trains on. Reads
dataset_scores.npz, fits one log-odds offset per language by NLL (train_probe_multi.bias_spread) and
reports the sd across languages with the global bias removed, with paired item and language
bootstraps. Fits no probe.

Output: probe/runs/qwen35-9b/bare/belebele_calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from calibrate_offsets import fit_bias           # noqa: E402
from train_probes import auroc                   # noqa: E402

PROBES = ["tydi", "tydi+udhr"]
# TyDi's 11 training languages. Belebele covers all of them plus 16 it never saw, which makes
# the same in/out split `coverage_split.py` runs on udhr available here too.
TYDI_LANGS = {"arb", "ben", "eng", "fin", "ind", "jpn", "kor", "rus", "swh", "tel", "tha"}


def offsets(z, y, lang, langs):
    """One free NLL offset per language, and the sd across them."""
    b = {}
    for g in langs:
        s = lang == g
        if s.sum() == 0 or y[s].min() == y[s].max():
            continue          # unbounded; `fit_bias` returns its +-25 sentinel
        b[g] = float(fit_bias(z[s], y[s]))
    return b, float(np.std(list(b.values())))


def spread_on(z, y, lang, rows, langs):
    """sd of the per-language offsets over a subset of rows."""
    _, sd = offsets(z[rows], y[rows], lang[rows], langs)
    return sd


def bootstrap(scores, y, lang, item, langs, n, seed, unit):
    """Paired bootstrap of the per-arm sd and of the difference between the arms."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(item.tolist())) if unit == "item" else list(langs)
    groups = {k: np.flatnonzero((item == k) if unit == "item" else (lang == k))
              for k in keys}

    draws = {p: [] for p in PROBES}
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        for p in PROBES:
            if unit == "item":
                # A draw of items keeps all 27 languages; the offsets are refitted on the
                # resampled passages.
                rows = np.concatenate([groups[keys[j]] for j in pick])
                draws[p].append(spread_on(scores[p], y, lang, rows, langs))
            else:
                # A draw of languages duplicates whole languages, so the sd is taken over the
                # drawn multiset -- over the unique set it would not be a resample at all.
                draws[p].append(float(np.std(
                    [fit_bias(scores[p][groups[keys[j]]], y[groups[keys[j]]]) for j in pick])))
    out = {}
    for p in PROBES:
        v = np.array(draws[p])
        out[p] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
    d = np.array(draws["tydi+udhr"]) - np.array(draws["tydi"])
    out["delta"] = {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
                    "hi": float(np.percentile(d, 97.5)),
                    "p_shrinks": float((d < 0).mean()), "unit": unit, "draws": n}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=REPO_ROOT / "probe/runs/qwen35-9b/bare")
    ap.add_argument("--scores", type=Path, default=None,
                    help="defaults to <run-dir>/dataset_scores.npz")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <run-dir>/belebele_calibration.json")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scores_path = args.scores or (args.run_dir / "dataset_scores.npz")
    out_path = args.out or (args.run_dir / "belebele_calibration.json")
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found -- run export_probe_directions.py first")

    npz = np.load(scores_path, allow_pickle=False)
    y = npz["y|belebele"]
    item = npz["item|belebele"]
    z = {p: npz[f"z|belebele|{p}"].astype(np.float64) for p in PROBES}

    res = {
        "dataset": "belebele", "n_rows": int(len(y)), "n_items": len(set(item.tolist())),
        "source": scores_path.name,
        "frac_answerable": float(y.mean()),
        "note": "Neither arm trains on Belebele. Nothing here reads mist-test.",
        "by": {},
    }

    for by, key in (("question", "qlang"), ("context", "clang")):
        lang = npz[f"{key}|belebele"]
        langs = sorted(set(lang.tolist()))
        block = {"n_languages": len(langs), "languages": langs, "probes": {}}
        print(f"\n=== Belebele, by {by} language "
              f"({len(langs)} groups, {len(y):,} rows, {res['n_items']} items) ===")
        print(f"{'probe':12s} {'AUROC':>7s} {'global':>8s} {'offset sd':>10s} {'range':>8s}")
        for p in PROBES:
            b, sd = offsets(z[p], y, lang, langs)
            vals = np.array(list(b.values()))
            block["probes"][p] = {
                "auroc": auroc(z[p], y),
                "global_bias": float(fit_bias(z[p], y)),
                "offset_sd": sd,
                "offset_range": float(vals.max() - vals.min()),
                "offsets": b,
            }
            print(f"{p:12s} {block['probes'][p]['auroc']:7.4f} "
                  f"{block['probes'][p]['global_bias']:+8.3f} {sd:10.3f} "
                  f"{block['probes'][p]['offset_range']:8.3f}")

        # Does the gain concentrate on languages TyDi never trained on? Same split as
        # `coverage_split.py`, which asks this of udhr.
        split = {}
        for tag, keep in (("in_tydi", [g for g in langs if g in TYDI_LANGS]),
                          ("not_in_tydi", [g for g in langs if g not in TYDI_LANGS])):
            sel = np.isin(lang, keep)
            split[tag] = {"languages": keep,
                          **{p: spread_on(z[p], y, lang, sel, keep) for p in PROBES}}
        split["difference_in_differences"] = (
            (split["not_in_tydi"]["tydi+udhr"] - split["not_in_tydi"]["tydi"])
            - (split["in_tydi"]["tydi+udhr"] - split["in_tydi"]["tydi"]))
        block["coverage_split"] = split
        print(f"  offset sd, TyDi's own {len(split['in_tydi']['languages'])} languages: "
              f"{split['in_tydi']['tydi']:.3f} -> {split['in_tydi']['tydi+udhr']:.3f}")
        print(f"  offset sd, the other {len(split['not_in_tydi']['languages'])}:          "
              f"{split['not_in_tydi']['tydi']:.3f} -> {split['not_in_tydi']['tydi+udhr']:.3f}"
              f"   (diff-in-diff {split['difference_in_differences']:+.3f})")

        if by == "question":
            for unit in ("item", "language"):
                block[f"bootstrap_{unit}"] = bootstrap(
                    z, y, lang, item, langs, args.bootstrap, args.seed, unit)
                d = block[f"bootstrap_{unit}"]["delta"]
                print(f"  paired {unit:8s} bootstrap, sd delta "
                      f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
                      f"P(shrinks) {d['p_shrinks']:.3f}")
        res["by"][by] = block

    # The reference the numbers only mean something against.
    arms = args.run_dir / "arms.json"
    if arms.exists():
        a = json.loads(arms.read_text())
        res["reference"] = {
            "noise_floor_tydi_val_offset_sd": a["noise_floor"]["tydi_val_offset_sd"],
            "udhr_offset_sd": {p: a["probes"][p]["residual_offset_sd"] for p in PROBES},
            "note": "The noise floor is the same estimator on TyDi validation, which is "
                    "monolingual, in-domain and 50/50 and so has no cross-language bias to "
                    "find. Belebele's collapse is partial where udhr's is complete -- and "
                    "Belebele is the corpus neither arm trained on.",
        }
        print(f"\nreference: noise floor {res['reference']['noise_floor_tydi_val_offset_sd']:.3f}"
              f" | on udhr {res['reference']['udhr_offset_sd']['tydi']:.3f} -> "
              f"{res['reference']['udhr_offset_sd']['tydi+udhr']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
