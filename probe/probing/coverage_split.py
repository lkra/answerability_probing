#!/usr/bin/env python3
"""Split AUROC and the fitted per-language offset by whether TyDi covers the question language, for
both probes; the difference between the probes isolates the coverage effect from language
difficulty. --dataset udhr (arms_oof.npz) | test (the committed test predictions) | belebele |
tydi_val (dataset_scores.npz) | all; --group question|context|both.

    python3 probe/probing/coverage_split.py --dataset all --group both --out probe/runs/qwen35-9b/bare/coverage_split_all.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_probes import auroc
from calibrate_offsets import fit_bias

# TyDi's eleven, as bare ISO-639-3. swh/tel/tha have no UDHR-QA counterpart, so the split there
# is 8 against 16; on Belebele (eval_crosslingual.py's `by_seen_group`) it is 11 against 16.
TYDI_LANGS = {"arb", "ben", "eng", "fin", "ind", "jpn", "kor", "rus", "swh", "tel", "tha"}

# Languages the *control* probe (TyDi+UDHR) never trained on. It saw UDHR-QA's 24 question
# languages, so this is empty on udhr and mist-test and is `fra` alone on Belebele. A language
# in here cannot sit in either group of the split: the control column would then be measuring
# its own coverage effect rather than holding difficulty fixed.
CONTROL_BLIND = {"fra"}

DATASETS = ("udhr", "test", "belebele", "tydi_val")
PROBES = ("tydi", "tydi+udhr")
COMMITTED = {"tydi": "mist_test_predictions.jsonl",
             "tydi+udhr": "mist_test_predictions_tydi_udhr.jsonl"}


def welch(a, b):
    """mean(a) - mean(b) with its Welch standard error and t. Returns plain floats."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a.mean() - b.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": float(diff), "se": float(se), "t": float(diff / se) if se else float("nan")}


def load(dataset, run_dir):
    """(scores per probe, y, {group -> language array}) for one evaluation set."""
    if dataset == "udhr":
        z = np.load(run_dir / "arms_oof.npz", allow_pickle=True)
        return ({p: z[f"z|{p}"].astype(np.float64) for p in PROBES}, z["y"],
                {"question": np.asarray(z["qlang"], dtype=str),
                 "context": np.asarray(z["clang"], dtype=str)})
    if dataset == "test":
        s, y, q, c = {}, None, None, None
        for p, fn in COMMITTED.items():
            rows = [json.loads(l) for l in (run_dir / fn).read_text().splitlines() if l.strip()]
            pr = np.array([r["prob_answerable"] for r in rows], dtype=np.float64)
            # The committed files store p, not the logit. Nothing saturates -- |z| tops out
            # around 14 -- so this inversion is lossless to well past the printed precision.
            s[p] = np.log(pr) - np.log1p(-pr)
            y = np.array([r["silver_answerable"] for r in rows])
            q = np.array([r["question_lang"] for r in rows], dtype=str)
            c = np.array([r["context_lang"] for r in rows], dtype=str)
        return s, y, {"question": q, "context": c}
    if dataset in ("belebele", "tydi_val"):
        z = np.load(run_dir / "dataset_scores.npz", allow_pickle=False)
        return ({p: z[f"z|{dataset}|{p}"].astype(np.float64) for p in PROBES}, z[f"y|{dataset}"],
                {"question": np.asarray(z[f"qlang|{dataset}"], dtype=str),
                 "context": np.asarray(z[f"clang|{dataset}"], dtype=str)})
    raise SystemExit(f"unknown dataset {dataset!r}")


def per_language(z, y, lang):
    """One free NLL offset and one AUROC per language, skipping single-class groups."""
    off, au = {}, {}
    for g in sorted(set(lang.tolist())):
        s = lang == g
        if s.sum() == 0 or y[s].min() == y[s].max():
            continue        # `fit_bias` is unbounded here and would return its sentinel
        off[g] = float(fit_bias(z[s], y[s]))
        au[g] = float(auroc(z[s], y[s]))
    return off, au


def analyse(dataset, group, scores, y, lang, quiet=False):
    langs, rec = None, {"dataset": dataset, "group": group, "n_rows": int(len(y)),
                        "frac_answerable": float(y.mean()), "probes": {}}
    for p in PROBES:
        off, au = per_language(scores[p], y, lang)
        langs = sorted(off)
        vals = np.array([off[g] for g in langs])
        rec["probes"][p] = {
            "auroc_pooled": float(auroc(scores[p], y)),
            "offset_sd": float(vals.std()),
            "offset_range": float(vals.max() - vals.min()),
            "offsets": off, "per_lang_auroc": au,
        }
    rec["languages"] = langs

    split = [g for g in langs if g not in CONTROL_BLIND]
    seen = [g for g in split if g in TYDI_LANGS]
    unseen = [g for g in split if g not in TYDI_LANGS]
    rec["split"] = {"in_tydi": seen, "not_in_tydi": unseen,
                    "excluded_control_blind": sorted(set(langs) & CONTROL_BLIND)}
    for p in PROBES:
        d = rec["probes"][p]
        for key, src in (("auroc", d["per_lang_auroc"]), ("offset", d["offsets"])):
            d[f"{key}_split"] = {
                "in_tydi": float(np.mean([src[g] for g in seen])),
                "not_in_tydi": float(np.mean([src[g] for g in unseen])),
                **welch([src[g] for g in unseen], [src[g] for g in seen])}
    rec["coverage_effect"] = {
        k: rec["probes"]["tydi"][f"{k}_split"]["diff"]
           - rec["probes"]["tydi+udhr"][f"{k}_split"]["diff"] for k in ("auroc", "offset")}

    if not quiet:
        print(f"\n=== {dataset}, by {group} language "
              f"({len(langs)} groups, {len(y):,} rows, {y.mean():.1%} answerable) ===")
        print(f"  split: {len(seen)} in TyDi / {len(unseen)} not"
              + (f"  [dropped {rec['split']['excluded_control_blind']}: the control probe "
                 "never saw them]" if rec["split"]["excluded_control_blind"] else ""))
        print(f"  {'probe':11s} {'AUROC':>7s} {'off sd':>7s} {'range':>7s}   "
              f"{'AUROC in/out':>18s}   {'offset in/out':>18s}")
        for p in PROBES:
            d = rec["probes"][p]
            print(f"  {p:11s} {d['auroc_pooled']:7.4f} {d['offset_sd']:7.3f} "
                  f"{d['offset_range']:7.3f}   "
                  f"{d['auroc_split']['in_tydi']:.4f}/{d['auroc_split']['not_in_tydi']:.4f} "
                  f"({d['auroc_split']['diff']:+.4f})   "
                  f"{d['offset_split']['in_tydi']:+.3f}/{d['offset_split']['not_in_tydi']:+.3f} "
                  f"({d['offset_split']['diff']:+.3f})")
        print(f"  coverage effect (difference of differences): "
              f"offset {rec['coverage_effect']['offset']:+.3f} log-odds, "
              f"AUROC {rec['coverage_effect']['auroc']:+.4f}")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path,
                   default=REPO_ROOT / "probe/runs/qwen35-9b/bare")
    p.add_argument("--dataset", default="udhr",
                   help="one of " + ", ".join(DATASETS) + ", or 'all'")
    p.add_argument("--group", default="question", choices=["question", "context", "both"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    datasets = DATASETS if args.dataset == "all" else tuple(args.dataset.split(","))
    groups = ("question", "context") if args.group == "both" else (args.group,)

    out = {"run_dir": str(args.run_dir), "probes": list(PROBES),
           "tydi_languages": sorted(TYDI_LANGS), "results": {}}
    for ds in datasets:
        scores, y, langs = load(ds, args.run_dir)
        for g in groups:
            out["results"][f"{ds}|{g}"] = analyse(ds, g, scores, y, langs[g])

    print("\n" + "=" * 78)
    print(f"{'set':22s} {'group':9s} {'sd tydi':>9s} {'sd tydi+udhr':>13s} {'coverage':>10s}")
    for k, r in out["results"].items():
        ds, g = k.split("|")
        print(f"{ds:22s} {g:9s} {r['probes']['tydi']['offset_sd']:9.3f} "
              f"{r['probes']['tydi+udhr']['offset_sd']:13.3f} "
              f"{r['coverage_effect']['offset']:+10.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
