#!/usr/bin/env python3
"""Fit the probe on a train cache and write per-row scores for an evaluation cache, keyed by id.
--layer sweep (default) picks the layer on a held-out slice of the train cache.

Output rows: id, lang_code, passage_lang, question_lang, answerable, prob_answerable, layer
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

import train_probes as tp
from _device import pick_device

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def attention_kinds(model_name):
    """layer index -> "full" | "linear" for Qwen3.5-4B's hybrid stack (full attention at indices 3, 7,
    11, ..., 31); None for a uniform model. Layer 0 (embeddings) is unlabelled.
    """
    try:
        from transformers import AutoConfig
        from _loading import text_config
        tc = text_config(AutoConfig.from_pretrained(model_name, trust_remote_code=True))
    except Exception:
        return None
    types = getattr(tc, "layer_types", None)
    if not types:
        return None
    # Hidden state i+1 is the output of decoder layer i; hidden state 0 is the embedding.
    return {i + 1: ("full" if t == "full_attention" else "linear") for i, t in enumerate(types)}


def sweep_layer(cfg, meta, acts, idx, y, device, l2, holdout):
    """Pick the operating layer on a held-out slice of TRAIN, never on the eval cache.

    Choosing the layer by eval AUROC and then reporting that same AUROC is selection on
    the test set: with ~33 candidates it buys a visible amount of nothing, and every
    downstream abstention rate inherits the optimism. The split is stratified by
    (language, label) so no language lands entirely on one side.
    """
    saved = tp.saved_layers(cfg)
    kinds = attention_kinds(cfg["model"])

    rng = np.random.default_rng(0)
    groups = {}
    for i, m in enumerate(meta):
        groups.setdefault((m["lang_code"], m["answerable"]), []).append(i)
    dev_mask = np.zeros(len(meta), dtype=bool)
    for key in sorted(groups):
        g = np.array(groups[key])
        n = max(1, int(round(holdout * len(g))))
        dev_mask[rng.choice(g, size=min(n, len(g)), replace=False)] = True
    fit_ix, dev_ix = np.where(~dev_mask)[0], np.where(dev_mask)[0]
    print(f"layer sweep: {len(saved)} layers, fit on {len(fit_ix):,} / pick on {len(dev_ix):,} "
          f"held-out train rows")

    print(f"\n{'layer':>6s} {'attn':>7s} {'AUROC':>8s} {'acc':>8s}")
    scores = {}
    for layer in saved:
        col = tp.layer_column(cfg, layer)
        X = np.asarray(acts[idx, col, :], dtype=np.float32)
        p = tp.fit_logreg(X[fit_ix], y[fit_ix], X[dev_ix], device, l2=l2)
        scores[layer] = tp.auroc(p, y[dev_ix])
        kind = (kinds or {}).get(layer, "-")
        print(f"{layer:6d} {kind:>7s} {scores[layer]:8.4f} "
              f"{tp.accuracy(p, y[dev_ix]):8.4f}", flush=True)

    best = max(scores, key=lambda l: (scores[l], -l))
    print(f"\nbest layer {best} (held-out AUROC {scores[best]:.4f})")

    if kinds:
        report_attention_contrast(scores, kinds)
        print(f"best layer is a {kinds.get(best, '?')}-attention layer")
    return best


def report_attention_contrast(scores, kinds):
    """Compare each full-attention layer's AUROC with its immediate linear-attention neighbours (same
    depth) rather than group means, which are confounded with depth.
    """
    full = sorted(l for l in scores if kinds.get(l) == "full")
    if not full:
        return
    deltas = []
    for f in full:
        neigh = [scores[n] for n in (f - 1, f + 1)
                 if n in scores and kinds.get(n) == "linear"]
        if neigh:
            deltas.append(scores[f] - float(np.mean(neigh)))
    if not deltas:
        return
    mean_d = float(np.mean(deltas))
    # Spread across the 8 comparisons, as a crude stand-in for a significance test:
    # with n=8 a mean delta smaller than the spread is not worth a claim.
    sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    print(f"\nfull-attention layers vs their immediate linear neighbours "
          f"(depth-matched, n={len(deltas)}):")
    print(f"  mean AUROC delta {mean_d:+.4f}  (sd {sd:.4f}, "
          f"{sum(d > 0 for d in deltas)}/{len(deltas)} positive)")
    if abs(mean_d) < sd / max(len(deltas) ** 0.5, 1):
        print("  -> within noise: no evidence the signal prefers full-attention layers.")
    elif mean_d > 0:
        print("  -> the signal is stronger at full-attention layers.")
    else:
        print("  -> the signal is stronger at linear-attention layers.")
    # The undepth-matched contrast, printed only to show how misleading it is.
    lin = [l for l in scores if kinds.get(l) == "linear"]
    print(f"  (raw group means, NOT depth-matched and not to be quoted: "
          f"full {np.mean([scores[l] for l in full]):.4f} vs "
          f"linear {np.mean([scores[l] for l in lin]):.4f})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-acts", type=Path, default=REPO_ROOT / "probe/caches/train")
    ap.add_argument("--eval-acts", type=Path, default=REPO_ROOT / "probe/caches/validation")
    ap.add_argument("--layer", default="sweep",
                    help="Operating layer, or 'sweep' (default) to choose one. Do not hardcode: "
                         "the old default of 21 was Qwen3-4B's best of 36 layers, and Qwen3.5-4B "
                         "has 32 -- a carried-over index is not even the same relative depth.")
    ap.add_argument("--sweep-holdout", type=float, default=0.2,
                    help="Fraction of the TRAIN cache held out to pick the layer on. The layer is "
                         "never chosen on the eval cache: that would be selecting on the test set "
                         "and would inflate every number downstream.")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "probe/generation/probe_predictions.jsonl")
    ap.add_argument("--keep-contaminated", action="store_true",
                    help="Keep rows that appear in the organizer sample. Off by default, as "
                         "elsewhere -- but note the exclusion is not label-neutral.")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--device", default=None,
                    help="Override the device. Default picks the largest CUDA device this "
                         "torch build can actually run kernels on, else CPU.")
    args = ap.parse_args()
    device = pick_device(args.device)

    cfg_tr, meta_tr, acts_tr, idx_tr = tp.load_cache(args.train_acts, args.keep_contaminated)
    cfg_te, meta_te, acts_te, idx_te = tp.load_cache(args.eval_acts, args.keep_contaminated)

    if cfg_tr["model"] != cfg_te["model"]:
        raise SystemExit(f"model mismatch: {cfg_tr['model']} vs {cfg_te['model']}")
    # Caches built under different prompts encode different things while looking identical.
    # `prompt_style` supersedes the old `prompt_template` key; check whichever is present.
    for key in ("prompt_style", "prompt_template", "max_len", "pooling"):
        if cfg_tr.get(key) != cfg_te.get(key):
            raise SystemExit(f"{key} mismatch between the two caches: "
                             f"{cfg_tr.get(key)!r} vs {cfg_te.get(key)!r}")

    y_tr = np.array([m["answerable"] for m in meta_tr])
    y_te = np.array([m["answerable"] for m in meta_te])

    if str(args.layer) == "sweep":
        layer = sweep_layer(cfg_tr, meta_tr, acts_tr, idx_tr, y_tr, device,
                            args.l2, args.sweep_holdout)
    else:
        layer = int(args.layer)

    col_tr = tp.layer_column(cfg_tr, layer)
    col_te = tp.layer_column(cfg_te, layer)
    X_tr = np.asarray(acts_tr[idx_tr, col_tr, :], dtype=np.float32)
    X_te = np.asarray(acts_te[idx_te, col_te, :], dtype=np.float32)

    print(f"\nfitting on {len(y_tr):,} train rows, layer {layer}, "
          f"scoring {len(y_te):,} validation rows ({device})")
    probs = tp.fit_logreg(X_tr, y_tr, X_te, device, l2=args.l2)

    print(f"\noverall  AUROC {tp.auroc(probs, y_te):.4f}   acc {tp.accuracy(probs, y_te):.4f}")
    print(f"\n{'lang':10s} {'n':>6s} {'AUROC':>7s} {'acc':>7s}")
    by = collections.defaultdict(list)
    for i, m in enumerate(meta_te):
        by[m["lang_code"]].append(i)
    for lang in sorted(by):
        ii = np.array(by[lang])
        print(f"{lang:10s} {len(ii):6,d} {tp.auroc(probs[ii], y_te[ii]):7.4f} "
              f"{tp.accuracy(probs[ii], y_te[ii]):7.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for m, p in zip(meta_te, probs):
            f.write(json.dumps({
                "id": m["id"], "lang_code": m["lang_code"],
                "passage_lang": m.get("passage_lang", m["lang_code"]),
                "question_lang": m.get("question_lang", m["lang_code"]),
                "answerable": m["answerable"], "prob_answerable": float(p),
                "layer": layer,
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {args.out}  ({len(meta_te):,} rows)")


if __name__ == "__main__":
    main()
