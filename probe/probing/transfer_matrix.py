#!/usr/bin/env python3
"""Language x language transfer matrix: probes trained without language L (rows) scored on every
language (columns), all on the same label-stratified --holdout rows carved out of every language
before training, so the diagonal (zero-shot) and the off-diagonal cells are comparable. Checked
against the --compare-to run record.

    python3 probe/probing/transfer_matrix.py --acts <train cache> --compare-to lolo_train.json --out transfer_matrix.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_activations import pick_device
from train_probes import accuracy, auroc, fit_logreg, layer_column, load_cache


def stratified_holdout(y, lang_of, langs, frac, seed):
    """Boolean mask: `frac` of each (language, label) cell, held out of all training.

    Stratified by label as well as language because the TyDi build emits rows grouped by label --
    an unstratified draw at a small `frac` can leave a language's holdout single-class, which
    makes its AUROC nan and its accuracy meaningless.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for lg in langs:
        for label in (0, 1):
            pool = np.flatnonzero((lang_of == lg) & (y == label))
            k = int(round(frac * len(pool)))
            if k == 0 and len(pool):
                k = 1
            mask[rng.choice(pool, size=k, replace=False)] = True
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--layer", type=int, default=None,
                   help="default: best_layer from --compare-to, else the middle layer")
    p.add_argument("--holdout", type=float, default=0.2,
                   help="fraction of every language held out and used as the evaluation set")
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-contaminated", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--compare-to", type=Path,
                   default=REPO_ROOT / "probe" / "runs" / "qwen35-4b" / "bare" / "lolo_train.json",
                   help="LOLO run to take the layer from and sanity-check the diagonal against")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "probe" / "runs" / "qwen35-4b" / "bare" / "transfer_matrix.json")
    args = p.parse_args()

    args.device = pick_device(args.device)
    cfg, meta, acts, idx = load_cache(args.acts, args.keep_contaminated)
    langs = sorted({m["lang_code"] for m in meta})
    y = np.array([m["answerable"] for m in meta])
    lang_of = np.array([m["lang_code"] for m in meta])

    prior = None
    if args.compare_to and args.compare_to.exists():
        prior = json.loads(args.compare_to.read_text())
    layer = args.layer if args.layer is not None else (
        prior["best_layer"] if prior and "best_layer" in prior else cfg["n_layers"] // 2)

    print(f"cache: {cfg['model']} | pooling={cfg['pooling']} | {len(meta)} rows | layer {layer}")

    held = stratified_holdout(y, lang_of, langs, args.holdout, args.seed)
    print(f"holdout: {held.sum()} of {len(y)} rows ({args.holdout:.0%} per language x label), "
          f"used as the evaluation set for every cell\n")

    X = np.asarray(acts[idx, layer_column(cfg, layer), :])
    eval_i = np.flatnonzero(held)
    eval_lang = lang_of[eval_i]
    eval_y = y[eval_i]

    matrix, eval_n = {}, {}
    for lg in langs:
        eval_n[lg] = int((eval_lang == lg).sum())

    for left_out in langs:
        # train on: every language except `left_out`, minus the global holdout rows
        tr = (~held) & (lang_of != left_out)
        probs = fit_logreg(X[tr], y[tr], X[eval_i], args.device, l2=args.l2)
        row = {}
        for lg in langs:
            sel = eval_lang == lg
            row[lg] = {"acc": accuracy(probs[sel], eval_y[sel]),
                       "auroc": auroc(probs[sel], eval_y[sel])}
        matrix[left_out] = row
        print(f"  probe without {left_out}: trained on {int(tr.sum())} rows", flush=True)

    # ---- summaries ---------------------------------------------------------------------
    summary = {}
    for lg in langs:
        zero = matrix[lg][lg]["auroc"]
        seen = float(np.mean([matrix[o][lg]["auroc"] for o in langs if o != lg]))
        summary[lg] = {"zero_shot_auroc": zero, "seen_auroc_mean": seen, "gap": zero - seen,
                       "zero_shot_acc": matrix[lg][lg]["acc"],
                       "seen_acc_mean": float(np.mean([matrix[o][lg]["acc"]
                                                       for o in langs if o != lg]))}

    print(f"\n=== AUROC: rows = language left out, columns = language scored ===")
    hdr = f"{'left out':10s} " + " ".join(f"{lg[:3]:>6s}" for lg in langs)
    print(hdr); print("-" * len(hdr))
    for lo in langs:
        cells = " ".join(
            (f"[{matrix[lo][lg]['auroc']:.3f}]" if lo == lg else f" {matrix[lo][lg]['auroc']:.3f} ")
            for lg in langs)
        print(f"{lo:10s} {cells}")
    print("[x] = zero-shot, the language this probe never saw\n")

    print(f"=== cost of not having seen the language (per column) ===")
    hdr = f"{'language':10s} {'zero-shot':>10s} {'seen':>8s} {'gap':>8s} {'eval n':>8s}"
    print(hdr); print("-" * len(hdr))
    for lg in langs:
        s = summary[lg]
        print(f"{lg:10s} {s['zero_shot_auroc']:10.3f} {s['seen_auroc_mean']:8.3f} "
              f"{s['gap']:+8.3f} {eval_n[lg]:8d}")
    print("-" * len(hdr))
    gaps = [summary[lg]["gap"] for lg in langs]
    print(f"{'mean':10s} {np.mean([summary[l]['zero_shot_auroc'] for l in langs]):10.3f} "
          f"{np.mean([summary[l]['seen_auroc_mean'] for l in langs]):8.3f} {np.mean(gaps):+8.3f}")

    # ---- cross-check against the main LOLO run ------------------------------------------
    check = None
    if prior and str(layer) in prior.get("folds", {}):
        d = [(lg, matrix[lg][lg]["auroc"], prior["folds"][str(layer)][lg]["auroc"]) for lg in langs]
        worst = max(d, key=lambda t: abs(t[1] - t[2]))
        check = {"max_abs_diff": abs(worst[1] - worst[2]), "worst_language": worst[0],
                 "here": worst[1], "lolo_run": worst[2]}
        print(f"\ndiagonal vs {args.compare_to.name} at layer {layer}: "
              f"max |diff| {check['max_abs_diff']:.4f} ({worst[0]}: {worst[1]:.4f} here vs "
              f"{worst[2]:.4f} there, on {1 - args.holdout:.0%} of the training data)")

    out = {"config": cfg, "layer": layer, "holdout_frac": args.holdout, "seed": args.seed,
           "l2": args.l2, "languages": langs, "eval_n": eval_n, "matrix": matrix,
           "summary": summary, "diagonal_check": check}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
