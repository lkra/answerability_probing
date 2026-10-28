#!/usr/bin/env python3
"""Fit the tydi and tydi+udhr probes once on the training rows that produced the committed test
predictions, verify that each reproduces those predictions row by row (max |dp| and mean signed
bias gates), and persist them. Scoring a cache is then ((X - mu) / sd) @ w + b.

Output: probe_directions.npz (w, b, mu, sd and provenance per probe); dataset_scores.npz (log-odds
of TyDi validation and BELEBELE under both probes, laid out like arms_oof.npz).

    python3 probe/probing/export_probe_directions.py --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_probes as tp                                          # noqa: E402
from train_probe_multi import Cache, MIST_TEST, check_compatible, item_folds, fold_of  # noqa: E402
from _device import pick_device                                    # noqa: E402

# probe -> the committed mist-test predictions it must reproduce.
COMMITTED = {
    "tydi": "mist_test_predictions.jsonl",
    "tydi+udhr": "mist_test_predictions_tydi_udhr.jsonl",
}

# Caches that have no committed per-row scores, and so are what this script is for.
TO_SCORE = ("tydi_val", "belebele")


def build_training_sets(caches, folds, seed, calib_fold):
    """(probe -> (X, y, description)), each on exactly the rows its run record used."""
    tydi, udhr = caches["tydi"], caches["udhr"]
    uf = fold_of(udhr, item_folds(udhr.item, folds, seed))
    keep_u = uf != calib_fold
    return {
        "tydi": (tydi.X[tydi.clean], tydi.y[tydi.clean],
                 f"TyDi train, {int(tydi.clean.sum()):,} rows, organizer sample excluded"),
        "tydi+udhr": (np.concatenate([tydi.X[tydi.clean], udhr.X[keep_u]]),
                      np.concatenate([tydi.y[tydi.clean], udhr.y[keep_u]]),
                      f"TyDi train ({int(tydi.clean.sum()):,}) + udhr outside calibration "
                      f"fold {calib_fold} ({int(keep_u.sum()):,})"),
    }


def project(params, X):
    """The whole point: a probe applied to new rows, with no fitting anywhere."""
    return ((np.asarray(X, dtype=np.float32) - params["mu"]) / params["sd"]) @ params["w"] \
        + params["b"]


def verify(probe, params, test, run_dir, tolerance, bias_tolerance):
    """Reproduce the committed mist-test predictions, or refuse to write the direction."""
    path = run_dir / COMMITTED[probe]
    if not path.exists():
        raise SystemExit(f"{path} not found -- cannot verify the recovered `{probe}` probe, "
                         f"and an unverified direction is not worth writing")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if [r["id"] for r in rows] != [m["id"] for m in test.meta]:
        raise SystemExit(f"{path} is not row-aligned with the mist-test cache")
    want = np.array([r["prob_answerable"] for r in rows])
    got = 1.0 / (1.0 + np.exp(-project(params, test.X)))
    worst = float(np.abs(want - got).max())
    bias = float(np.mean(got - want))
    rho = float(np.corrcoef(want, got)[0, 1])
    bad = worst > tolerance or abs(bias) > bias_tolerance
    print(f"  {probe:10s} vs {path.name}: max |dp| {worst:.2e}  mean bias {bias:+.2e}  "
          f"r {rho:.8f}  [{'MISMATCH' if bad else 'OK'}]")
    if bad:
        raise SystemExit(
            f"recovered `{probe}` probe does not reproduce {path.name} "
            f"(max |dp| {worst:.2e} vs {tolerance:.0e}, bias {bias:+.2e} vs "
            f"{bias_tolerance:.0e}). The training rows were not rebuilt as the run record "
            f"had them; writing this direction would silently ship a different probe. "
            f"Do not raise the tolerances to get past this without knowing why -- the "
            f"docstring records what each failure mode actually measures."
        )
    return {"max_abs_dp": worst, "mean_bias": bias, "pearson_r": rho, "against": path.name}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--run-dir", type=Path,
                    default=REPO_ROOT / "probe/runs/qwen35-9b/bare")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-fold", type=int, default=0,
                    help="the udhr fold held out of the shipped probe's training set. Must "
                         "match what --mode final was run with, or verification fails.")
    ap.add_argument("--tolerance", type=float, default=5e-3,
                    help="max |dp| against the committed predictions. Sits in the measured "
                         "gap between CPU-vs-GPU arithmetic (9.2e-4) and the nearest wrong "
                         "training set (8.6e-2) -- see the docstring.")
    ap.add_argument("--bias-tolerance", type=float, default=5e-4,
                    help="max mean signed dp. The sharper of the two gates: wrong training "
                         "rows shift every score one way (6.6e-3), arithmetic does not "
                         "(2.5e-5).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    print("caches")
    names = ("tydi", "udhr", "tydi_val", "belebele")
    caches = {n: Cache(n, args.layer, args.root) for n in names}
    test = Cache("mist_test", args.layer, args.root, rel=MIST_TEST)
    check_compatible([caches[n] for n in names] + [test])

    training = build_training_sets(caches, args.folds, args.seed, args.calib_fold)

    print(f"\nfitting {len(training)} probes at layer {args.layer} ({device}) -- once, so "
          f"nothing after this has to")
    directions, checks = {}, {}
    for probe, (X_tr, y_tr, desc) in training.items():
        print(f"  {probe:10s} {desc}")
        # The scores are discarded; `params` is what this script exists to keep.
        _, params = tp.fit_logreg(X_tr, y_tr, X_tr[:1], device, l2=args.l2,
                                  return_logits=True, return_params=True)
        directions[probe] = params

    print("\nverifying against the committed predictions")
    for probe, params in directions.items():
        checks[probe] = verify(probe, params, test, args.run_dir, args.tolerance,
                               args.bias_tolerance)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": caches["tydi"].cfg.get("model"),
        "prompt_style": caches["tydi"].cfg.get("prompt_style"),
        "pooling": caches["tydi"].cfg.get("pooling"),
        "max_len": caches["tydi"].cfg.get("max_len"),
        "layer": args.layer, "l2": args.l2, "folds": args.folds, "seed": args.seed,
        "calib_fold": args.calib_fold,
        "training": {p: d for p, (_, _, d) in training.items()},
        "verified_against": checks,
        "note": "logit p = ((x - mu) / sd) @ w + b, with x the layer-`layer` activation of "
                "the cache this was fitted on. Scoring a new cache is that projection; it "
                "does not require refitting.",
    }
    out_dir = args.run_dir / "probe_directions.npz"
    np.savez_compressed(
        out_dir, meta=np.array(json.dumps(meta)),
        **{f"{k}|{p}": directions[p][k] for p in directions for k in ("w", "mu", "sd")},
        **{f"b|{p}": np.array(directions[p]["b"]) for p in directions})
    print(f"\nwrote {out_dir}")

    print("\nscoring the caches that had no committed per-row predictions, by projection")
    payload, cover = {}, {}
    for name in TO_SCORE:
        c = caches[name]
        # See the docstring: kept for tydi_val on purpose, a no-op for belebele.
        payload[f"y|{name}"] = c.y
        payload[f"qlang|{name}"] = c.qlang
        payload[f"clang|{name}"] = c.clang
        # Belebele renders ~15 items into every (question, context) language pair, so anything
        # putting an interval on a Belebele statistic has to resample items, not rows. TyDi
        # has no item structure and its column is empty strings.
        payload[f"item|{name}"] = c.item
        cover[name] = sorted(set(c.qlang.tolist()))
        for probe, params in directions.items():
            z = project(params, c.X)
            payload[f"z|{name}|{probe}"] = z
            print(f"  {name:9s} {probe:10s} {len(z):6,d} rows  AUROC {tp.auroc(z, c.y):.4f}")
    payload["datasets"] = np.array(list(TO_SCORE))
    payload["probes"] = np.array(list(directions))
    payload["meta"] = np.array(json.dumps(
        {"layer": args.layer, "l2": args.l2, "source": str(out_dir.name),
         "question_languages": cover,
         "contamination": "organizer-sample rows are excluded from training and KEPT when "
                          "scoring tydi_val, per train_probe_multi.noise_floor"}))
    scores = args.run_dir / "dataset_scores.npz"
    np.savez_compressed(scores, **payload)
    print(f"\nwrote {scores}")
    for name in TO_SCORE:
        print(f"  {name:9s} {len(cover[name])} question languages: {' '.join(cover[name])}")


if __name__ == "__main__":
    main()
