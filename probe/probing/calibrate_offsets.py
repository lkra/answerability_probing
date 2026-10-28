#!/usr/bin/env python3
"""Fit one log-odds offset per question language on a calibration cache, never the test set:
logit p' = logit p + b[language], b minimising NLL. --validate fits on one item-disjoint half and
scores the other; --by additive also fits context-language terms.

    python3 probe/probing/calibrate_offsets.py --calib-acts probe/caches/qwen35-9b/bare/udhr --out offsets_udhr.json
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

GLOBAL_PRIOR_SHIFT = 1.3863   # log(0.8/0.2); the shipped system's threshold-0.20 equivalent


def fit_bias(logits, y, lo=-25.0, hi=25.0, tol=1e-10):
    """The scalar `b` minimising NLL of sigmoid(logits + b) against y.

    Convex in b with derivative sum(sigmoid(logits + b) - y), which is strictly increasing,
    so bisection on the derivative is exact and needs no optimiser. Returns +-`hi` for a
    single-class group, where the minimiser is unbounded -- callers must not let a
    single-class group set a shipped offset.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.min() == y.max():
        return hi if y[0] == 1 else lo
    d = lambda b: float((1.0 / (1.0 + np.exp(-(logits + b))) - y).sum())
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if d(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_additive(logits, y, gq, gc, rounds=50):
    """Offsets additive in two groupings, b = bq[question] + bc[context]: 24 + 25 parameters instead
    of 600 cells of ~29 rows. Coordinate descent over 1-D bisections; bc is centred each round for
    identifiability.
    """
    bq = collections.defaultdict(float)
    bc = collections.defaultdict(float)
    idx_q = {g: np.flatnonzero(gq == g) for g in np.unique(gq)}
    idx_c = {g: np.flatnonzero(gc == g) for g in np.unique(gc)}
    for _ in range(rounds):
        prev = (dict(bq), dict(bc))
        off_c = np.array([bc[g] for g in gc])
        for g, ix in idx_q.items():
            bq[g] = fit_bias(logits[ix] + off_c[ix], y[ix])
        off_q = np.array([bq[g] for g in gq])
        for g, ix in idx_c.items():
            bc[g] = fit_bias(logits[ix] + off_q[ix], y[ix])
        mean_c = float(np.mean([bc[g] for g in idx_c]))
        for g in idx_c:
            bc[g] -= mean_c
        for g in idx_q:
            bq[g] += mean_c
        if prev[0] and max(abs(bq[g] - prev[0][g]) for g in bq) < 1e-6:
            break
    return dict(bq), dict(bc)


def nll(logits, y):
    z = np.clip(logits, -30, 30)
    return float(np.mean(np.logaddexp(0, z) - y * z))


def scores(logits, y, name):
    """Decision-time numbers at the natural boundary (logit 0, i.e. p = 0.5 after correction)."""
    pred = (logits >= 0).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 0) & (y == 1)).sum())   # false abstention: called unanswerable, is not
    fn = int(((pred == 1) & (y == 0)).sum())
    rec = tn / max(tn + fn, 1)
    prec = tn / max(tn + fp, 1)
    return {"name": name, "abstentions": tn + fp, "true": tn, "false": fp,
            "recall": rec, "precision": prec,
            "f1": 2 * rec * prec / max(rec + prec, 1e-9),
            "accuracy": float((pred == y).mean()), "nll": nll(logits, y)}


def report(rows, title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'rule':28s} {'abst':>6s} {'true':>6s} {'false':>6s} {'rec':>6s} {'prec':>6s} "
          f"{'F1':>6s} {'acc':>6s} {'NLL':>7s}")
    for r in rows:
        print(f"{r['name']:28s} {r['abstentions']:6,d} {r['true']:6,d} {r['false']:6,d} "
              f"{r['recall']:6.3f} {r['precision']:6.3f} {r['f1']:6.3f} {r['accuracy']:6.3f} "
              f"{r['nll']:7.4f}")


def group_key(meta, by):
    return np.array([m["question_lang"] if by == "question" else m["passage_lang"]
                     for m in meta])


def item_split(meta, seed=0):
    """Two halves with disjoint items -- never a random row split.

    Every item appears in ~24 languages, so a row-level split puts the same item on both
    sides and the offset is then fitted and scored on the same content in different
    languages. That would measure memorisation of items, not transfer of a per-language
    bias, and would look good for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    items = sorted({m.get("item_id") for m in meta})
    if items == [None]:
        # TyDi has no item structure: it is monolingual, and every row is its own
        # (passage, question) pair that appears in exactly one language. There is nothing for
        # a row split to leak *across*, so a row split is the honest one there. Anywhere an
        # item_id exists, refusing to fall back matters -- see the docstring.
        print("no item_id in this cache (TyDi): splitting by row. Safe only because no item "
              "recurs across languages here.")
        a = rng.random(len(meta)) < 0.5
        return a, ~a
    perm = rng.permutation(len(items))
    half = {items[i] for i in perm[: len(items) // 2]}
    a = np.array([m.get("item_id") in half for m in meta])
    return a, ~a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-acts", type=Path,
                    default=REPO_ROOT / "probe/caches/qwen35-9b/bare/train")
    ap.add_argument("--calib-acts", type=Path,
                    default=REPO_ROOT / "probe/caches/qwen35-9b/bare/udhr")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--by", default="question", choices=["question", "context", "additive"],
                    help="grouping for the offset; `additive` fits question + context jointly")
    ap.add_argument("--validate", action="store_true",
                    help="two-fold item-disjoint validation instead of writing offsets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    cfg_tr, meta_tr, acts_tr, idx_tr = tp.load_cache(args.train_acts, False)
    cfg_ca, meta_ca, acts_ca, idx_ca = tp.load_cache(args.calib_acts, True)
    for key in ("model", "prompt_style", "pooling", "max_len"):
        if cfg_tr.get(key) != cfg_ca.get(key):
            raise SystemExit(f"{key} mismatch: train {cfg_tr.get(key)!r} vs "
                             f"calib {cfg_ca.get(key)!r} -- different representations.")

    X_tr = np.asarray(acts_tr[idx_tr, tp.layer_column(cfg_tr, args.layer), :], dtype=np.float32)
    y_tr = np.array([m["answerable"] for m in meta_tr])
    X_ca = np.asarray(acts_ca[idx_ca, tp.layer_column(cfg_ca, args.layer), :], dtype=np.float32)
    y = np.array([m["answerable"] for m in meta_ca])

    print(f"probe: {len(y_tr):,} train rows, layer {args.layer} ({device})")
    print(f"calibration set: {len(y):,} rows, {y.mean():.1%} answerable, "
          f"{len({m.get('item_id') for m in meta_ca})} items, "
          f"{len({m['question_lang'] for m in meta_ca})} question x "
          f"{len({m['passage_lang'] for m in meta_ca})} context languages")
    z = tp.fit_logreg(X_tr, y_tr, X_ca, device, l2=args.l2, return_logits=True).astype(np.float64)
    print(f"uncorrected AUROC on the calibration set: {tp.auroc(z, y):.4f}")

    gq = group_key(meta_ca, "question")
    gc = group_key(meta_ca, "context")
    g = gq if args.by == "question" else gc

    def apply_offsets(zz, sel, bq, bc=None):
        out = zz[sel].copy()
        keys = (gq if args.by == "question" else gc)[sel]
        out = out + np.array([bq.get(k, GLOBAL_PRIOR_SHIFT) for k in keys])
        if bc is not None:
            out = out + np.array([bc.get(k, 0.0) for k in gc[sel]])
        return out

    if args.validate:
        folds = item_split(meta_ca, args.seed)
        for i, (fit_sel, eval_sel) in enumerate(((folds[0], folds[1]), (folds[1], folds[0]))):
            n_items_fit = len({m.get('item_id') for m, s in zip(meta_ca, fit_sel) if s})
            n_items_ev = len({m.get('item_id') for m, s in zip(meta_ca, eval_sel) if s})
            if args.by == "additive":
                bq, bc = fit_additive(z[fit_sel], y[fit_sel], gq[fit_sel], gc[fit_sel])
            else:
                bq = {k: fit_bias(z[fit_sel & (g == k)], y[fit_sel & (g == k)])
                      for k in np.unique(g[fit_sel])}
                bc = None
            # The centred variant is the one that can actually ship. A fitted offset is
            # (domain level + language deviation) and only the second half is known to
            # transfer to the test set: this calibration set is legal-register like the
            # test set but not identical to it, so importing its *level* means calibrating
            # the decision threshold on another dataset's difficulty. Centring keeps the
            # level at the shipped +1.3863 -- justified by the prior alone, which is the
            # one calibration move already agreed -- and transfers only the per-language
            # structure, which is what the control established is real.
            mean_b = float(np.mean(list(bq.values())))
            bq_centred = {k: v - mean_b + GLOBAL_PRIOR_SHIFT for k, v in bq.items()}
            rows = [
                scores(z[eval_sel] + GLOBAL_PRIOR_SHIFT, y[eval_sel], "global +1.3863 (shipped)"),
                scores(z[eval_sel] + fit_bias(z[fit_sel], y[fit_sel]), y[eval_sel],
                       "global, fitted on fold"),
                scores(apply_offsets(z, eval_sel, bq_centred, bc), y[eval_sel],
                       f"per-{args.by} centred on +1.3863"),
                scores(apply_offsets(z, eval_sel, bq, bc), y[eval_sel],
                       f"per-{args.by}, fitted on fold"),
            ]
            report(rows, f"FOLD {i + 1}: fit on {n_items_fit} items, score on {n_items_ev} "
                         f"held-out items ({int(eval_sel.sum()):,} rows)")
            b = np.array(sorted(bq.values()))
            print(f"\noffset spread across {len(b)} groups: sd {b.std():.3f}, "
                  f"range {b.min():+.2f} to {b.max():+.2f} "
                  f"(a flat set of offsets means the correction has nothing to do)")
            if bc is not None:
                c = np.array(sorted(bc.values()))
                print(f"context-language component: sd {c.std():.3f}, "
                      f"range {c.min():+.2f} to {c.max():+.2f}")
        return

    # ---- fit on everything and write ----------------------------------------------------
    if args.by == "additive":
        bq, bc = fit_additive(z, y, gq, gc)
    else:
        bq = {k: fit_bias(z[g == k], y[g == k]) for k in np.unique(g)}
        bc = None

    print(f"\n{'=' * 78}\nOFFSETS (fitted on the full calibration set)\n{'=' * 78}")
    print(f"{'group':10s} {'n':>7s} {'unans':>7s} {'offset':>8s} {'vs global':>10s}")
    for k in sorted(bq):
        sel = g == k
        print(f"{k:10s} {int(sel.sum()):7,d} {int((y[sel] == 0).sum()):7,d} "
              f"{bq[k]:+8.3f} {bq[k] - GLOBAL_PRIOR_SHIFT:+10.3f}")
    rows = [scores(z + GLOBAL_PRIOR_SHIFT, y, "global +1.3863 (shipped)"),
            scores(apply_offsets(z, np.ones(len(y), bool), bq, bc), y, f"per-{args.by}")]
    report(rows, "IN-SAMPLE (not a validation -- see --validate)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "by": args.by, "layer": args.layer, "l2": args.l2,
            "train_acts": str(args.train_acts), "calib_acts": str(args.calib_acts),
            "model": cfg_tr.get("model"), "prompt_style": cfg_tr.get("prompt_style"),
            "global_prior_shift": GLOBAL_PRIOR_SHIFT,
            "question_offsets": {k: float(v) for k, v in bq.items()},
            "context_offsets": {k: float(v) for k, v in (bc or {}).items()},
            "note": "logit p' = logit p + question_offsets[qlang] (+ context_offsets[clang]); "
                    "abstain when p' < 0.5. Fitted on held-out data, never on the test set.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
