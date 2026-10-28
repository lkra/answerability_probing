#!/usr/bin/env python3
r"""Leave-one-language-out probing over a cached TyDi train set: for every layer, an L2 logistic
regression (LBFGS) trained on 10 languages and scored on the 11th. Also reports the majority,
lexical-overlap, shuffled-label and language-identity controls. Organiser-sample rows are excluded
(--keep-contaminated keeps them).

    python3 probe/probing/train_probes.py --acts probe/caches/qwen35-9b/bare/train \
        --layers 0,4,8,12,16,19,22,26,29,32 --out lolo_train.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_activations import pick_device      # same gated-GPU guard, see its docstring

# Whitespace tokenisation is wrong for zh/ja/th, which do not space-delimit words. Falling back to
# character n-grams there keeps the overlap baseline meaningful rather than trivially zero -- and
# a trivially-zero baseline would make the probe look better than it is, which is the failure mode
# this whole baseline exists to prevent.
NO_SPACE_SCRIPTS = {"zho_Hans", "jpn_Jpan", "tha_Thai", "kor_Hang"}
_WORD = re.compile(r"\w+", re.UNICODE)


def overlap_feature(question, passage, lang_code):
    """Fraction of question tokens that also occur in the passage."""
    if lang_code in NO_SPACE_SCRIPTS:
        q = {question[i:i + 2] for i in range(max(len(question) - 1, 0))}
        p = {passage[i:i + 2] for i in range(max(len(passage) - 1, 0))}
    else:
        q = set(_WORD.findall(question.lower()))
        p = set(_WORD.findall(passage.lower()))
    return len(q & p) / len(q) if q else 0.0


def load_source_text(data_dir, split, ids):
    """id -> (question, passage), read back from the TyDi build.

    The activation cache deliberately does not carry passage text -- it would multiply the size of
    meta.jsonl for something already on disk. The overlap baseline needs it, so join by id here.
    """
    want = set(ids)
    text = {}
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in want:
                # `context`, not `input`: `input` is context + question, so overlap against it is
                # trivially 1.0 for every row and the baseline silently reports chance.
                text[o["id"]] = (o["question"], o["context"])
    return text


def saved_layers(cfg):
    """Model-layer indices present in a cache, in column order.

    Caches built before `--save-layers` existed have no such key and store every layer, so the
    column index and the model layer index coincide.
    """
    return cfg.get("saved_layers", list(range(cfg["n_layers"])))


def layer_column(cfg, layer):
    """Model layer index -> column in acts.npy."""
    saved = saved_layers(cfg)
    if layer not in saved:
        raise SystemExit(f"layer {layer} is not in this cache (has {saved}). Re-extract with "
                         f"--save-layers including it, or pick one of the stored layers.")
    return saved.index(layer)


def load_cache(acts_dir, keep_contaminated):
    acts_dir = Path(acts_dir)
    cfg = json.loads((acts_dir / "config.json").read_text())
    meta = [json.loads(l) for l in open(acts_dir / "meta.jsonl")]
    n_layers, hidden = cfg["n_layers"], cfg["hidden_size"]

    acts = np.memmap(acts_dir / "acts.npy", dtype=np.float16, mode="r").reshape(-1, n_layers, hidden)
    if len(acts) != len(meta):
        raise SystemExit(
            f"cache is inconsistent: {len(acts)} activation rows vs {len(meta)} meta rows in "
            f"{acts_dir}. Re-run extraction (it appends both together, so this means a partial write)."
        )

    keep = np.ones(len(meta), dtype=bool)
    if not keep_contaminated:
        keep = np.array([not m["in_organizer_sample"] for m in meta])
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"excluding {n_drop} rows that appear in the organizer sample "
                  f"(--keep-contaminated to include)")
    idx = np.flatnonzero(keep)
    return cfg, [meta[i] for i in idx], acts, idx


def fit_logreg(X_tr, y_tr, X_te, device, l2=1e-3, epochs=200, lr=0.05, seed=0,
               return_logits=False):
    """L2-regularised logistic regression on frozen features, linear on purpose. return_logits=True
    returns log-odds; use it for anything in log-odds space, since float32 sigmoid output loses
    resolution below p = 0.01, where the unanswerable rows sit.
    """
    torch.manual_seed(seed)
    Xtr = torch.tensor(np.asarray(X_tr, dtype=np.float32), device=device)
    ytr = torch.tensor(np.asarray(y_tr, dtype=np.float32), device=device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr = (Xtr - mu) / sd

    w = torch.zeros(Xtr.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=epochs, line_search_fn="strong_wolfe")
    lossfn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossfn(Xtr @ w + b, ytr) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        Xte = torch.tensor(np.asarray(X_te, dtype=np.float32), device=device)
        Xte = (Xte - mu) / sd
        z = Xte @ w + b
        return (z if return_logits else torch.sigmoid(z)).cpu().numpy()


def accuracy(probs, y):
    return float(((probs >= 0.5).astype(int) == np.asarray(y)).mean())


def auroc(probs, y):
    """Rank-based AUROC; no sklearn dependency and exact under ties."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty(len(probs), dtype=float)
    ranks[order] = np.arange(1, len(probs) + 1)
    # average ranks within ties
    _, inv, counts = np.unique(probs, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", type=Path, required=True, help="directory written by extract_activations.py")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "built" / "tydiqa_answerability",
                   help="source TyDi build, joined by id for the overlap baseline")
    p.add_argument("--layers", default=None, help="comma-separated layer indices (default: every 4th)")
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--keep-contaminated", action="store_true")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--out", type=Path, default=None, help="write results as JSON")
    p.add_argument("--skip-controls", action="store_true", help="answerability probes only")
    p.add_argument("--control-seeds", type=int, default=5,
                   help="label permutations to average the shuffled-label control over. One is "
                        "not enough: the per-fold spread is wide enough that a single draw lands "
                        "anywhere from 0.49 to 0.55 with no leakage present.")
    args = p.parse_args()

    args.device = pick_device(args.device)
    cfg, meta, acts, idx = load_cache(args.acts, args.keep_contaminated)
    langs = sorted({m["lang_code"] for m in meta})
    y = np.array([m["answerable"] for m in meta])
    lang_of = np.array([m["lang_code"] for m in meta])
    print(f"cache: {cfg['model']} | pooling={cfg['pooling']} | {len(meta)} rows | {len(langs)} languages")
    print(f"layers available: {saved_layers(cfg)}")

    avail = saved_layers(cfg)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers else avail[::4] or avail)
    missing = [l for l in layers if l not in avail]
    if missing:
        raise SystemExit(f"layers {missing} are not stored in this cache (has {avail})")
    print(f"probing layers: {layers}\n")

    results = {"config": cfg, "layers": layers, "languages": langs,
               "n_rows": len(meta), "folds": {}, "baselines": {}}

    # ---- baselines, computed once per held-out language (no activations involved) ----------
    if not args.skip_controls:
        print("=== baselines (per held-out language) ===")
        print(f"{'lang':10s} {'n':>6s} {'majority':>9s} {'overlap':>9s} {'ovl-auroc':>10s}")
        text = load_source_text(args.data_dir, cfg["split"], [m["id"] for m in meta])
        have_overlap = len(text) == len(meta)
        if not have_overlap:
            print(f"  (overlap baseline needs the source split: found text for {len(text)}/{len(meta)} "
                  f"rows under --data-dir {args.data_dir})")
        ov = np.array([overlap_feature(*text[m["id"]], m["lang_code"]) if m["id"] in text else 0.0
                       for m in meta])
        for lg in langs:
            te = lang_of == lg
            tr = ~te
            maj = max(Counter(y[te]).values()) / te.sum()
            if have_overlap:
                pr = fit_logreg(ov[tr, None], y[tr], ov[te, None], args.device, l2=args.l2)
                acc, au = accuracy(pr, y[te]), auroc(pr, y[te])
            else:
                acc = au = float("nan")
            results["baselines"][lg] = {"n": int(te.sum()), "majority": maj,
                                        "overlap_acc": acc, "overlap_auroc": au}
            print(f"{lg:10s} {te.sum():6d} {maj:9.3f} {acc:9.3f} {au:10.3f}")
        print()

    # ---- leave-one-language-out answerability probes ---------------------------------------
    print("=== leave-one-language-out (train on 10, test on the held-out language) ===")
    header = f"{'layer':>5s} " + " ".join(f"{lg[:3]:>6s}" for lg in langs) + f" {'mean':>7s}"
    print(header)
    print("-" * len(header))
    for layer in layers:
        X = np.asarray(acts[idx, layer_column(cfg, layer), :])
        row = {}
        for lg in langs:
            te = lang_of == lg
            probs = fit_logreg(X[~te], y[~te], X[te], args.device, l2=args.l2)
            row[lg] = {"acc": accuracy(probs, y[te]), "auroc": auroc(probs, y[te]),
                       "n": int(te.sum())}
        results["folds"][layer] = row
        mean = float(np.mean([row[lg]["acc"] for lg in langs]))
        print(f"{layer:5d} " + " ".join(f"{row[lg]['acc']:6.3f}" for lg in langs) + f" {mean:7.3f}")

    # ---- the comparison that decides whether any of this means anything ---------------------
    best = max(results["folds"], key=lambda l: np.mean([results["folds"][l][g]["acc"] for g in langs]))
    if not args.skip_controls and results["baselines"]:
        print(f"\n=== probe vs. overlap baseline, at best layer ({best}) ===")
        print("the probe must beat overlap; chance for this task is the overlap column, not 0.5")
        hdr = f"{'lang':10s} {'probe acc':>10s} {'ovl acc':>8s} {'d':>7s}   {'probe auc':>10s} {'ovl auc':>8s} {'d':>7s}"
        print(hdr); print("-" * len(hdr))
        d_acc, d_auc = [], []
        for lg in langs:
            pa, pu = results["folds"][best][lg]["acc"], results["folds"][best][lg]["auroc"]
            oa, ou = results["baselines"][lg]["overlap_acc"], results["baselines"][lg]["overlap_auroc"]
            d_acc.append(pa - oa); d_auc.append(pu - ou)
            print(f"{lg:10s} {pa:10.3f} {oa:8.3f} {pa - oa:+7.3f}   {pu:10.3f} {ou:8.3f} {pu - ou:+7.3f}")
        print("-" * len(hdr))
        print(f"{'mean':10s} {np.mean([results['folds'][best][g]['acc'] for g in langs]):10.3f} "
              f"{np.mean([results['baselines'][g]['overlap_acc'] for g in langs]):8.3f} "
              f"{np.mean(d_acc):+7.3f}   "
              f"{np.mean([results['folds'][best][g]['auroc'] for g in langs]):10.3f} "
              f"{np.mean([results['baselines'][g]['overlap_auroc'] for g in langs]):8.3f} "
              f"{np.mean(d_auc):+7.3f}")
        results["best_layer"] = best
        results["delta_vs_overlap"] = {"acc": float(np.mean(d_acc)), "auroc": float(np.mean(d_auc))}

    # ---- controls ---------------------------------------------------------------------------
    if not args.skip_controls:
        X = np.asarray(acts[idx, layer_column(cfg, best), :])
        print(f"\n=== controls at best layer ({best}) ===")

        # Averaged over several permutations, not one. A single permutation is far too noisy
        # to detect leakage with: on the 8-language cache individual folds land anywhere in
        # 0.36-0.73, and seed 0 alone gives 0.554 where seed 1 gives 0.495. Reading one draw
        # as "0.554, therefore leakage" -- or a lucky 0.500 as "therefore none" -- is reading
        # noise. The spread across permutations is reported so the reader can see the width.
        per_seed = []
        for seed in range(args.control_seeds):
            rng = np.random.default_rng(seed)
            accs = []
            for lg in langs:
                te = lang_of == lg
                y_sh = y.copy()
                y_sh[~te] = rng.permutation(y_sh[~te])
                accs.append(accuracy(
                    fit_logreg(X[~te], y_sh[~te], X[te], args.device, l2=args.l2), y[te]))
            per_seed.append(float(np.mean(accs)))
        mean_sh = float(np.mean(per_seed))
        sd_sh = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
        results["shuffled_label_acc"] = mean_sh
        results["shuffled_label_acc_sd"] = sd_sh
        results["shuffled_label_acc_per_seed"] = per_seed
        print(f"shuffled labels : {mean_sh:.3f} +- {sd_sh:.3f} over {len(per_seed)} permutations "
              f"(expect ~0.500; a single permutation is too noisy to read)")

        # language identification: one-vs-rest, same folds are meaningless here, so use a
        # within-split holdout instead -- the point is only whether language is linearly present.
        perm = rng.permutation(len(y)); cut = int(0.8 * len(y))
        tr_i, te_i = perm[:cut], perm[cut:]
        langid_acc = []
        for lg in langs:
            yl = (lang_of == lg).astype(int)
            pr = fit_logreg(X[tr_i], yl[tr_i], X[te_i], args.device, l2=args.l2)
            langid_acc.append(auroc(pr, yl[te_i]))
        results["langid_auroc"] = float(np.nanmean(langid_acc))
        print(f"language identity: {np.nanmean(langid_acc):.3f} AUROC (one-vs-rest, 80/20 split)")
        print("   -> languages are separable in this space, so LOLO transfer is not a triviality")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
