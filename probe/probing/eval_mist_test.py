#!/usr/bin/env python3
r"""Fit the TyDi probe on the train cache and score the 8,640 test rows against the silver labels.
Numbers are agreement with silver, not accuracy. --by-item aggregates per item (100 items).

    python3 probe/probing/eval_mist_test.py --train-acts probe/caches/qwen35-9b/bare/train \
        --test-acts probe/caches/qwen35-9b/bare/mist-test --layer 19
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import train_probes as tp
from _device import pick_device

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def acc_at(probs, y, thr):
    """Accuracy at an explicit threshold.

    `train_probes.accuracy` hardcodes 0.5, which is right when train and eval share a
    class prior and wrong here: the probe is fitted on a 50/50 TyDi split and applied to
    an 80/20 test set, so 0.5 is not its operating point. Using the hardcoded version
    reports numbers below the majority baseline for a probe that ranks at .995 AUROC.
    """
    return float(((np.asarray(probs) >= thr).astype(int) == np.asarray(y)).mean())


def rate(n, d):
    return f"{n / d:6.1%}" if d else "     -"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-acts", type=Path,
                    default=REPO_ROOT / "probe/caches/qwen35-9b/bare/train")
    ap.add_argument("--test-acts", type=Path,
                    default=REPO_ROOT / "probe/caches/qwen35-9b/bare/mist-test")
    ap.add_argument("--layer", type=int, required=True,
                    help="operating layer; must be one the caches actually store")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-confidence", default=None, choices=["medium", "high"],
                    help="drop silver labels below this confidence before scoring")
    ap.add_argument("--out", type=Path, default=None, help="write per-row scores as JSONL")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    cfg_tr, meta_tr, acts_tr, idx_tr = tp.load_cache(args.train_acts, False)
    # keep_contaminated=True: `in_organizer_sample` is False for every mist-test row by
    # construction -- this *is* the organizer set, so the flag means nothing here and the
    # filter would be a no-op that only invites confusion.
    cfg_te, meta_te, acts_te, idx_te = tp.load_cache(args.test_acts, True)

    for key in ("model", "prompt_style", "pooling", "max_len"):
        if cfg_tr.get(key) != cfg_te.get(key):
            raise SystemExit(f"{key} mismatch: train {cfg_tr.get(key)!r} vs "
                             f"test {cfg_te.get(key)!r} -- the probe would be reading a "
                             f"different representation than it was fitted on.")

    keep = np.ones(len(meta_te), dtype=bool)
    if args.min_confidence:
        order = {"low": 0, "medium": 1, "high": 2}
        lo = order[args.min_confidence]
        keep = np.array([order.get(m.get("confidence", "high"), 2) >= lo for m in meta_te])
        print(f"confidence >= {args.min_confidence}: {keep.sum():,} of {len(keep):,} rows")
    meta_te = [m for m, k in zip(meta_te, keep) if k]

    X_tr = np.asarray(acts_tr[idx_tr, tp.layer_column(cfg_tr, args.layer), :], dtype=np.float32)
    y_tr = np.array([m["answerable"] for m in meta_tr])
    X_te = np.asarray(acts_te[idx_te, tp.layer_column(cfg_te, args.layer), :],
                      dtype=np.float32)[keep]
    y_te = np.array([m["answerable"] for m in meta_te])

    print(f"fitting on {len(y_tr):,} train rows (layer {args.layer}), scoring "
          f"{len(y_te):,} test rows ({device})")
    p = tp.fit_logreg(X_tr, y_tr, X_te, device, l2=args.l2)

    pred = (p >= args.threshold).astype(int)
    maj = max(y_te.mean(), 1 - y_te.mean())
    print(f"\n{'=' * 72}\nPROBE vs SILVER LABELS\n{'=' * 72}")
    print(f"rows {len(y_te):,} | {y_te.mean():.1%} answerable | majority baseline {maj:.1%}")
    print(f"AUROC {tp.auroc(p, y_te):.4f}   accuracy {acc_at(p, y_te, args.threshold):.4f} "
          f"(at threshold {args.threshold})")

    tp_, fp = int(((pred == 1) & (y_te == 1)).sum()), int(((pred == 1) & (y_te == 0)).sum())
    fn, tn = int(((pred == 0) & (y_te == 1)).sum()), int(((pred == 0) & (y_te == 0)).sum())
    print(f"\n{'':22s}{'silver: answerable':>20s}{'silver: unanswerable':>22s}")
    print(f"{'probe: answerable':22s}{tp_:>20,}{fp:>22,}")
    print(f"{'probe: unanswerable':22s}{fn:>20,}{tn:>22,}")
    print(f"\nrecall on unanswerable {rate(tn, tn + fp)}   "
          f"precision {rate(tn, tn + fn)}")

    # ---- per item: the honest unit ----------------------------------------------------
    by_item = collections.defaultdict(list)
    for m, s in zip(meta_te, p):
        by_item[m.get("item_id")].append((s, m["answerable"]))
    item_p = np.array([np.mean([s for s, _ in v]) for v in by_item.values()])
    item_y = np.array([v[0][1] for v in by_item.values()])
    print(f"\n{'=' * 72}\nBY ITEM ({len(item_y)} distinct items — the effective sample)\n{'=' * 72}")
    print(f"mean probe score per item, AUROC {tp.auroc(item_p, item_y):.4f}   "
          f"accuracy {acc_at(item_p, item_y, args.threshold):.4f}")
    print(f"{item_y.sum():.0f} answerable / {(1 - item_y).sum():.0f} unanswerable items")
    wrong = [(i, float(np.mean([s for s, _ in v])), v[0][1])
             for i, v in by_item.items()
             if (np.mean([s for s, _ in v]) >= args.threshold) != bool(v[0][1])]
    if wrong:
        print(f"\n{len(wrong)} items where probe and silver disagree on the item mean:")
        for i, s, y in sorted(wrong, key=lambda t: t[1]):
            print(f"  {i}  silver={'answerable' if y else 'UNANSWERABLE':12s}  "
                  f"probe p(answerable)={s:.3f}")

    # ---- slices ------------------------------------------------------------------------
    for name, keyf in (("question language", lambda m: m["question_lang"]),
                       ("context language", lambda m: m["passage_lang"])):
        g = collections.defaultdict(list)
        for m, s, yy in zip(meta_te, p, y_te):
            g[keyf(m)].append((s, yy))
        print(f"\n{'=' * 72}\nBY {name.upper()}\n{'=' * 72}")
        print(f"{'lang':8s} {'n':>6s} {'AUROC':>8s} {'acc':>8s}")
        for k in sorted(g):
            ss = np.array([s for s, _ in g[k]]); yy = np.array([y for _, y in g[k]])
            print(f"{k:8s} {len(yy):6,d} {tp.auroc(ss, yy):8.4f} {acc_at(ss, yy, args.threshold):8.4f}")

    same = np.array([m["passage_lang"] == m["question_lang"] for m in meta_te])
    if same.any() and (~same).any():
        print(f"\n{'=' * 72}\nSAME- vs CROSS-LINGUAL\n{'=' * 72}")
        for lbl, sel in (("same-language", same), ("cross-lingual", ~same)):
            print(f"{lbl:16s} n={sel.sum():5,d}  AUROC {tp.auroc(p[sel], y_te[sel]):.4f}  "
                  f"acc {acc_at(p[sel], y_te[sel], args.threshold):.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for m, s in zip(meta_te, p):
                f.write(json.dumps({
                    "id": m["id"], "item_id": m.get("item_id"),
                    "question_lang": m["question_lang"], "context_lang": m["passage_lang"],
                    "silver_answerable": m["answerable"], "confidence": m.get("confidence"),
                    "prob_answerable": float(s), "layer": args.layer,
                }, ensure_ascii=False) + "\n")
        print(f"\nWrote {args.out}  ({len(meta_te):,} rows)")


if __name__ == "__main__":
    main()
