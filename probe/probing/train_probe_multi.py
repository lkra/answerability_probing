#!/usr/bin/env python3
r"""Fit probes on several caches at once (TyDi, UDHR, BELEBELE) with item-disjoint K-fold rotation:
fold f evaluates, fold f+1 calibrates the decision rule, the rest train. Language codes are
normalised to ISO 639-3 first.

--mode arms (default): every training-set arm, AUROC per evaluation set, per-language offset spread,
paired item bootstrap, noise floor (--save-oof writes the out-of-fold logits). --mode lolo:
leave-one-language-out on UDHR (--lolo-probes, --lolo-removal question-udhr|question-all|union-all,
--lolo-budget fixed, --lolo-budget-seeds). --mode lolo-tydi: the same on TyDi's 11 languages.
--mode final: fit --final-probe with --final-rule and score the test set (the only mode that reads
it).

    python3 probe/probing/train_probe_multi.py --layer 19 --folds 5 --bootstrap 2000 --out arms.json --save-oof arms_oof.npz
    python3 probe/probing/train_probe_multi.py --mode final --layer 19 --folds 5 --final-probe tydi+udhr \
        --final-rule globalF1 --calib-fold 0 --bias-from cv-mean --out mist_test_predictions_tydi_udhr.jsonl
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_probes as tp                       # noqa: E402
from calibrate_offsets import fit_bias, GLOBAL_PRIOR_SHIFT   # noqa: E402
from _device import pick_device                 # noqa: E402

# name -> cache path relative to the repo root.
#
# Every cache is loaded whole and the `in_organizer_sample` rows are masked at the point of
# use instead (`Cache.clean`), because the two uses want different things. Training and the
# external-validity read-out must exclude them -- they are rows the shared task will score
# us on. The TyDi-validation *noise-floor control* must not: dropping 3,512 of 13,325 rows
# is heavily uneven across languages and leaves `ben` with 108 rows that are all
# unanswerable, where the per-language bias is unbounded and `fit_bias` correctly returns
# its -25 sentinel. That one group then dominates the sd and the control reads 6.9 instead
# of 0.28 -- i.e. the control silently stops being a noise floor.
SOURCES = {
    "tydi":     "probe/caches/qwen35-9b/bare/train",
    "tydi_val": "probe/caches/qwen35-9b/bare/validation",
    "udhr":     "probe/caches/qwen35-9b/bare/udhr",
    "belebele": "probe/caches/qwen35-9b/bare/belebele",
}

# The test set, kept out of SOURCES on purpose: `--mode final` is the only path that touches
# it, it is never a training or a model-selection input, and having to name it separately
# means no loop over SOURCES can pick it up by accident.
MIST_TEST = "probe/caches/qwen35-9b/bare/mist-test"

# probe -> the caches pooled to train it. The reported arms A-E are (probe, decision rule)
# pairs, not four separate fits: A and B are the same probe read at two operating points,
# which is the comparison the plan asks for and is cheaper and less error-prone than
# fitting it twice.
PROBES = {
    "tydi":            ["tydi"],
    "udhr":            ["udhr"],
    "tydi+udhr":       ["tydi", "udhr"],
    "tydi+udhr+beleb": ["tydi", "udhr", "belebele"],
}

# The plan's five arms, all read at the *same* operating-point rule so that the comparison
# is about training data and nothing else. `+` rows add per-language offsets on top of an
# in-domain arm, which is the question E2 leaves open if the calibration gap survives
# training. The two `ref` rows are the system as actually shipped -- same probe, but with
# the level fixed at the +1.3863 prior rather than fitted.
ARMS = [
    ("A",   "tydi",            "globalF1",      "TyDi only (baseline)"),
    ("B",   "tydi",            "centredF1",     "TyDi + per-language offsets"),
    ("C",   "udhr",            "globalF1",      "in-domain multilingual, no offsets"),
    ("D",   "tydi+udhr",       "globalF1",      "TyDi + udhr, no offsets"),
    ("E",   "tydi+udhr+beleb", "globalF1",      "maximum diversity, no offsets"),
    ("C+",  "udhr",            "centredF1",     "C + per-language offsets"),
    ("D+",  "tydi+udhr",       "centredF1",     "D + per-language offsets"),
    ("E+",  "tydi+udhr+beleb", "centredF1",     "E + per-language offsets"),
    ("ref", "tydi",            "global@prior",  "as shipped: variant1, global threshold 0.20"),
    ("ref", "tydi",            "centred@prior", "as shipped: primary, centred on +1.3863"),
]

# Caches whose rows are folded by item. TyDi is deliberately absent: it is monolingual and
# every (passage, question) pair occurs exactly once, so there is nothing for a fold to
# leak across and holding items out would only shrink the training set for no gain.
FOLDED = ("udhr", "belebele")


def norm_lang(code):
    """FLORES `arb_Arab` / ISO-639-3 `arb` -> `arb`.

    Deliberately lossy on script: `zho_Hans` and `zho_Hant` both become `zho`, which is what
    the test set's own language labels do. Keeping them apart would create a language the
    evaluation set has no counterpart for.
    """
    return code.split("_")[0]


class Cache:
    """One activation cache, reduced to the single layer being probed."""

    def __init__(self, name, layer, root, rel=None):
        path = Path(root) / (rel or SOURCES[name])
        cfg, meta, acts, idx = tp.load_cache(path, True)
        self.name = name
        self.path = path
        self.cfg = cfg
        self.meta = meta
        self.X = np.asarray(acts[idx, tp.layer_column(cfg, layer), :], dtype=np.float32)
        self.y = np.array([m["answerable"] for m in meta], dtype=np.int64)
        self.qlang = np.array([norm_lang(m["question_lang"]) for m in meta])
        self.clang = np.array([norm_lang(m["passage_lang"]) for m in meta])
        self.item = np.array([m.get("item_id") or "" for m in meta])
        self.clean = np.array([not m["in_organizer_sample"] for m in meta])
        print(f"  {name:9s} {len(self.y):6,d} rows  {self.y.mean():5.1%} answerable  "
              f"{len(set(self.qlang)):2d} question langs  "
              f"{len(set(self.item) - {''}):4d} items  "
              f"{int((~self.clean).sum()):5,d} in organizer sample  <- {path}")


def check_compatible(caches):
    """Refuse to pool caches built from different representations.

    Nothing downstream would fail loudly if this were skipped -- the arrays have the same
    width -- so the run would produce a plausible table describing a probe fitted across two
    different models.
    """
    keys = ("model", "prompt_style", "pooling", "max_len", "hidden_size")
    ref_name, ref = caches[0].name, caches[0].cfg
    for c in caches[1:]:
        for k in keys:
            if c.cfg.get(k) != ref.get(k):
                raise SystemExit(f"{k} mismatch: {ref_name} {ref.get(k)!r} vs "
                                 f"{c.name} {c.cfg.get(k)!r} -- these are different "
                                 f"representations and must not be pooled.")


def item_folds(items, k, seed):
    """item id -> fold index, balanced in count, shuffled by seed."""
    uniq = sorted(set(items) - {""})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    return {uniq[j]: int(i % k) for i, j in enumerate(perm)}


def fold_of(cache, folds):
    """Per-row fold index for a folded cache."""
    return np.array([folds[i] for i in cache.item])


# ---------------------------------------------------------------------------------------
# metrics.  The positive class is *abstention* throughout: the decision the system actually
# makes is "write the not-answerable literal", and its precision and recall are what the
# submission is judged on.  `z < 0` abstains, matching calibrate_offsets.scores().
# ---------------------------------------------------------------------------------------

def f1_abstain(z, y):
    pred = z < 0
    true = y == 0
    hit = int((pred & true).sum())
    false_abst = int((pred & ~true).sum())
    missed = int((~pred & true).sum())
    prec = hit / max(hit + false_abst, 1)
    rec = hit / max(hit + missed, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12)


def per_lang_f1(z, y, lang, langs):
    return {g: f1_abstain(z[lang == g], y[lang == g]) for g in langs}


def worst_k(d, k=3):
    return float(np.mean(sorted(d.values())[:k]))


def summarise(z, y, lang, langs):
    per = per_lang_f1(z, y, lang, langs)
    vals = np.array([per[g] for g in langs])
    pred = z < 0
    return {
        "micro_f1": f1_abstain(z, y),
        "macro_f1": float(vals.mean()),
        "worst3_f1": worst_k(per),
        "sd_across_langs": float(vals.std()),
        "worst_lang": min(per, key=per.get),
        "abstentions": int(pred.sum()),
        "false_abstentions": int((pred & (y == 1)).sum()),
        "recall": float((pred & (y == 0)).sum() / max((y == 0).sum(), 1)),
        "per_lang_f1": {g: float(per[g]) for g in langs},
    }


def fit_bias_f1(z, y):
    """The bias b maximising abstention-F1 of z + b < 0, computed exactly by sweeping the cut over the
    sorted scores; the median of tied cuts is taken. Differs from the NLL-optimal bias on an 80/20
    set (0.046 macro-F1 for the TyDi arm).
    """
    order = np.argsort(z)
    zs, ys = z[order], np.asarray(y)[order]
    n_neg = int((ys == 0).sum())
    if n_neg == 0 or n_neg == len(ys):
        return fit_bias(z, y)          # degenerate: fall back to the NLL sentinel behaviour
    hits = np.concatenate([[0], np.cumsum(ys == 0)])            # hits(i), i = 0..n
    f1 = 2.0 * hits / (np.arange(len(hits)) + n_neg)
    best = np.flatnonzero(f1 >= f1.max() - 1e-12)
    i = int(best[len(best) // 2])
    # threshold t with `abstain iff z < t`, placed between the two rows it separates
    lo = zs[i - 1] if i > 0 else zs[0] - 1.0
    hi = zs[i] if i < len(zs) else zs[-1] + 1.0
    return float(-(lo + hi) / 2)


def bias_spread(z, y, lang, langs):
    """sd across languages of a free per-language offset fitted on the arm's out-of-fold logits.
    Single-class languages are dropped.
    """
    b, kept = {}, []
    for g in langs:
        s = lang == g
        if s.sum() == 0 or y[s].min() == y[s].max():
            continue
        b[g] = fit_bias(z[s], y[s])
        kept.append(g)
    return float(np.std([b[g] for g in kept])), {g: float(b[g]) for g in kept}


# ---------------------------------------------------------------------------------------
# decision rules
# ---------------------------------------------------------------------------------------

def decision_rules(z_eval, lang_eval, z_cal, y_cal, lang_cal):
    """Adjusted eval logits under each rule, all fitted on the calibration fold: global, centred
    per-language and per-language biases, each for NLL and for F1 (`*F1`); plus the `@prior` pair
    with the level pinned at +1.3863 (log 0.8/0.2), which reproduces the earlier TyDi-only
    submissions and is a reference point only. Arms are compared at globalF1 and centredF1.
    """
    langs = sorted(set(lang_cal))

    def group_biases(fit):
        g = fit(z_cal, y_cal)
        per = {k: fit(z_cal[lang_cal == k], y_cal[lang_cal == k]) for k in langs}
        return g, per, float(np.mean([per[k] for k in langs]))

    b_glob, b_lang, mean_b = group_biases(fit_bias)
    b_globF1, b_langF1, mean_bF1 = group_biases(fit_bias_f1)

    def spread(per, mean, level):
        return np.array([per.get(g, mean) - mean + level for g in lang_eval])

    def raw(per, fallback):
        return np.array([per.get(g, fallback) for g in lang_eval])

    return {
        "global": z_eval + b_glob,
        "centred": z_eval + spread(b_lang, mean_b, b_glob),
        "fitted": z_eval + raw(b_lang, b_glob),
        "globalF1": z_eval + b_globF1,
        "centredF1": z_eval + spread(b_langF1, mean_bF1, b_globF1),
        "fittedF1": z_eval + raw(b_langF1, b_globF1),
        "global@prior": z_eval + GLOBAL_PRIOR_SHIFT,
        "centred@prior": z_eval + spread(b_lang, mean_b, GLOBAL_PRIOR_SHIFT),
    }, {"nll_global": b_glob, "nll_lang": b_lang,
        "f1_global": b_globF1, "f1_lang": b_langF1}


RULES = ["globalF1", "centredF1", "fittedF1",
         "global", "centred", "fitted", "global@prior", "centred@prior"]


# ---------------------------------------------------------------------------------------
# the cross-validated run
# ---------------------------------------------------------------------------------------

def run_arms(caches, args, device):
    udhr, beleb, tydi_val = caches["udhr"], caches["belebele"], caches["tydi_val"]
    k = args.folds
    uf = fold_of(udhr, item_folds(udhr.item, k, args.seed))
    bf = fold_of(beleb, item_folds(beleb.item, k, args.seed))
    langs = sorted(set(udhr.qlang))

    print(f"\n{k}-fold rotation over {len(set(udhr.item))} udhr items "
          f"(eval f, calibrate f+1, train the rest); {len(langs)} question languages")

    # out-of-fold predictions, pooled: every udhr row scored exactly once, by a probe that
    # saw neither its item nor the items its decision rule was calibrated on.
    oof = {p: {"z": np.full(len(udhr.y), np.nan),
               "rules": {r: np.full(len(udhr.y), np.nan) for r in RULES},
               "bias": [], "ext": collections.defaultdict(list)}
           for p in PROBES}

    for f in range(k):
        ev = uf == f
        cal = uf == (f + 1) % k
        tr_u = ~(ev | cal)
        beleb_ev = bf == f
        print(f"\nfold {f}: train {int(tr_u.sum()):,} udhr rows "
              f"({len({i for i, s in zip(udhr.item, tr_u) if s})} items) | "
              f"calibrate {int(cal.sum()):,} | evaluate {int(ev.sum()):,}")

        # every eval target for this fold, concatenated so each probe is fitted once
        val_ext = tydi_val.clean            # E4 read-out excludes organizer-sample rows
        blocks = [("udhr_eval", udhr.X[ev]), ("udhr_cal", udhr.X[cal]),
                  ("tydi_val", tydi_val.X[val_ext]), ("belebele", beleb.X[beleb_ev])]
        X_te = np.concatenate([b for _, b in blocks])
        bounds = np.cumsum([0] + [len(b) for _, b in blocks])

        for pname, srcs in PROBES.items():
            parts_X, parts_y = [], []
            for s in srcs:
                c = caches[s]
                sel = tr_u if s == "udhr" else (bf != f) if s == "belebele" else c.clean
                parts_X.append(c.X[sel])
                parts_y.append(c.y[sel])
            X_tr = np.concatenate(parts_X)
            y_tr = np.concatenate(parts_y)
            z_all = tp.fit_logreg(X_tr, y_tr, X_te, device, l2=args.l2,
                                  return_logits=True).astype(np.float64)
            del X_tr
            cut = {name: z_all[bounds[i]:bounds[i + 1]] for i, (name, _) in enumerate(blocks)}

            rules, bias = decision_rules(
                cut["udhr_eval"], udhr.qlang[ev], cut["udhr_cal"], udhr.y[cal], udhr.qlang[cal])
            o = oof[pname]
            o["z"][ev] = cut["udhr_eval"]
            for r in RULES:
                o["rules"][r][ev] = rules[r]
            o["bias"].append(bias)
            o["ext"]["tydi_val_auroc"].append(tp.auroc(cut["tydi_val"], tydi_val.y[val_ext]))
            o["ext"]["belebele_auroc"].append(tp.auroc(cut["belebele"], beleb.y[beleb_ev]))
            print(f"    {pname:16s} {len(y_tr):6,d} train rows | "
                  f"udhr-eval AUROC {tp.auroc(cut['udhr_eval'], udhr.y[ev]):.4f} | "
                  f"tydi-val {o['ext']['tydi_val_auroc'][-1]:.4f} | "
                  f"belebele {o['ext']['belebele_auroc'][-1]:.4f}")

    # ---- pooled results -----------------------------------------------------------------
    results = {"layer": args.layer, "l2": args.l2, "folds": k, "seed": args.seed,
               "languages": langs, "n_udhr_rows": int(len(udhr.y)),
               "n_udhr_items": len(set(udhr.item)), "probes": {}}
    for pname, o in oof.items():
        assert not np.isnan(o["z"]).any(), "some udhr row was never held out"
        sd, resid = bias_spread(o["z"], udhr.y, udhr.qlang, langs)
        results["probes"][pname] = {
            "sources": PROBES[pname],
            "oof_auroc": tp.auroc(o["z"], udhr.y),
            "residual_offset_sd": sd,
            "residual_offsets": resid,
            "calibrated_global_bias": {
                "nll": float(np.mean([b["nll_global"] for b in o["bias"]])),
                "f1": float(np.mean([b["f1_global"] for b in o["bias"]])),
            },
            "calibration_offsets_nll": {
                g: float(np.mean([b["nll_lang"][g] for b in o["bias"]])) for g in langs},
            "calibration_offsets_f1": {
                g: float(np.mean([b["f1_lang"][g] for b in o["bias"]])) for g in langs},
            "external": {kk: float(np.mean(v)) for kk, v in o["ext"].items()},
            "external_sd": {kk: float(np.std(v)) for kk, v in o["ext"].items()},
            "rules": {r: summarise(o["rules"][r], udhr.y, udhr.qlang, langs) for r in RULES},
        }

    results["bootstrap"] = bootstrap(oof, udhr, langs, args)
    results["noise_floor"] = noise_floor(caches, args, device)
    return results, oof, udhr


def bootstrap(oof, udhr, langs, args):
    """Paired item bootstrap: every arm is resampled on the same draw of items.

    Items, not rows. 17,280 rows are 164 items rendered ~105 times each, so a row bootstrap
    would report an interval ~10x too narrow and every arm would look significantly
    different from every other one.
    """
    rows_of = collections.defaultdict(list)
    for i, it in enumerate(udhr.item):
        rows_of[it].append(i)
    items = sorted(rows_of)
    rows_of = {it: np.array(v) for it, v in rows_of.items()}
    rng = np.random.default_rng(args.seed)
    keys = [(p, r) for p in PROBES for r in RULES]
    draws = {kk: [] for kk in keys}
    for _ in range(args.bootstrap):
        pick = rng.integers(0, len(items), len(items))
        ix = np.concatenate([rows_of[items[j]] for j in pick])
        y, lang = udhr.y[ix], udhr.qlang[ix]
        present = [g for g in langs if (lang == g).any()]
        for p, r in keys:
            draws[(p, r)].append(worst_k(per_lang_f1(oof[p]["rules"][r][ix], y, lang, present)))
    draws = {kk: np.array(v) for kk, v in draws.items()}
    out = {f"{p}|{r}": {"worst3_lo": float(np.percentile(v, 2.5)),
                        "worst3_hi": float(np.percentile(v, 97.5))}
           for (p, r), v in draws.items()}

    # Paired differences against the two incumbents. Marginal intervals are what the plan
    # named, and they are kept above -- but two arms scored on the *same* items are
    # correlated, so comparing marginal intervals is a strictly weaker test than the paired
    # one and can call a real difference a tie. Both are reported; the paired interval is
    # the one that answers "is this arm better than what we ship".
    out["paired_vs"] = {}
    for base in [("tydi", "globalF1"), ("tydi", "centred@prior"), ("tydi+udhr", "globalF1")]:
        b = draws[base]
        out["paired_vs"][f"{base[0]}|{base[1]}"] = {
            f"{p}|{r}": {"delta": float((draws[(p, r)] - b).mean()),
                         "lo": float(np.percentile(draws[(p, r)] - b, 2.5)),
                         "hi": float(np.percentile(draws[(p, r)] - b, 97.5)),
                         "p_better": float((draws[(p, r)] > b).mean())}
            for p, r in keys}
    return out


def noise_floor(caches, args, device):
    """The negative control: same estimator where no cross-language bias should exist.

    TyDi validation is monolingual, in-domain and 50/50, so a per-language offset fitted on
    it has nothing to correct and its spread is the measurement noise of the estimator
    itself. Residual spreads in the table above are only interesting relative to this.
    """
    tydi, val = caches["tydi"], caches["tydi_val"]
    z = tp.fit_logreg(tydi.X[tydi.clean], tydi.y[tydi.clean], val.X, device, l2=args.l2,
                      return_logits=True).astype(np.float64)
    langs = sorted(set(val.qlang))
    sd, b = bias_spread(z, val.y, val.qlang, langs)
    print(f"\nnoise floor (TyDi train -> TyDi validation): per-language offset sd {sd:.3f} "
          f"over {len(langs)} languages")
    return {"tydi_val_offset_sd": sd, "tydi_val_offsets": b,
            "tydi_val_auroc": tp.auroc(z, val.y)}


# ---------------------------------------------------------------------------------------
# E3: leave-one-language-out on udhr
# ---------------------------------------------------------------------------------------

def run_lolo(caches, args, device):
    """Hold a question language out of training and score the probe on it. `lang`: scored on all of
    L's rows (items shared with training); `lang+item`: scored on held-out items only, the transfer
    number. AUROC.
    """
    udhr = caches["udhr"]
    k = args.folds
    uf = fold_of(udhr, item_folds(udhr.item, k, args.seed))
    bf = fold_of(caches["belebele"], item_folds(caches["belebele"].item, k, args.seed))
    langs = sorted(set(udhr.qlang))
    probes = args.lolo_probes.split(",")
    positions = list(range(min(args.lolo_folds, k)))
    print(f"\nLOLO over {len(langs)} languages x {len(positions)} fold position(s), "
          f"probes {probes}")

    # The reference line: the same probe with the language left *in* training, scored on the
    # same held-out items. Without it a LOLO AUROC is uninterpretable -- a language can score
    # low because transfer failed or because that language is simply hard, and only the gap
    # between the two separates them.
    ref = {pname: collections.defaultdict(list) for pname in probes}
    for f in positions:
        ev_item = uf == f
        for pname in probes:
            parts_X, parts_y = [], []
            for s in PROBES[pname]:
                c = caches[s]
                sel = ~ev_item if s == "udhr" else (bf != f) if s == "belebele" else c.clean
                parts_X.append(c.X[sel])
                parts_y.append(c.y[sel])
            z = tp.fit_logreg(np.concatenate(parts_X), np.concatenate(parts_y),
                              udhr.X[ev_item], device, l2=args.l2,
                              return_logits=True).astype(np.float64)
            lg_ev, y_ev = udhr.qlang[ev_item], udhr.y[ev_item]
            for lg in langs:
                ref[pname][lg].append(tp.auroc(z[lg_ev == lg], y_ev[lg_ev == lg]))

    out = {"layer": args.layer, "folds": k, "seed": args.seed,
           "fold_positions": positions, "languages": langs, "probes": {}}
    for pname in probes:
        srcs = PROBES[pname]
        per = {}
        for lg in langs:
            au_lang, au_both = [], []
            for f in positions:
                ev_item = uf == f
                held = udhr.qlang == lg
                tr_u = (~ev_item) & (~held)
                parts_X, parts_y = [], []
                for s in srcs:
                    c = caches[s]
                    sel = tr_u if s == "udhr" else (bf != f) if s == "belebele" else c.clean
                    parts_X.append(c.X[sel])
                    parts_y.append(c.y[sel])
                sel_lang = held
                sel_both = held & ev_item
                X_te = np.concatenate([udhr.X[sel_lang], udhr.X[sel_both]])
                z = tp.fit_logreg(np.concatenate(parts_X), np.concatenate(parts_y), X_te,
                                  device, l2=args.l2, return_logits=True).astype(np.float64)
                n1 = int(sel_lang.sum())
                au_lang.append(tp.auroc(z[:n1], udhr.y[sel_lang]))
                au_both.append(tp.auroc(z[n1:], udhr.y[sel_both]))
            per[lg] = {"lang": float(np.mean(au_lang)), "lang+item": float(np.mean(au_both)),
                       "in_training": float(np.mean(ref[pname][lg]))}
            print(f"  {pname:16s} {lg:4s}  lang {per[lg]['lang']:.4f}  "
                  f"lang+item {per[lg]['lang+item']:.4f}  "
                  f"in-training {per[lg]['in_training']:.4f}  "
                  f"gap {per[lg]['lang+item'] - per[lg]['in_training']:+.4f}")
        mean = {k2: float(np.mean([v[k2] for v in per.values()]))
                for k2 in ("lang", "lang+item", "in_training")}
        out["probes"][pname] = {
            "sources": srcs, "per_lang": per, "mean_lang": mean["lang"],
            "mean_lang_item": mean["lang+item"], "mean_in_training": mean["in_training"],
            "mean_gap": mean["lang+item"] - mean["in_training"],
            "worst_lang_item": min(per, key=lambda g: per[g]["lang+item"]),
        }
        print(f"  {pname:16s} MEAN  lang {mean['lang']:.4f}  "
              f"lang+item {mean['lang+item']:.4f}  in-training {mean['in_training']:.4f}  "
              f"gap {mean['lang+item'] - mean['in_training']:+.4f}")
    return out


# ---------------------------------------------------------------------------------------
# E5: the model that ships
# ---------------------------------------------------------------------------------------

def run_final(caches, args, device):
    """Fit the selected arm on TyDi plus every UDHR item outside --calib-fold and score the test set
    once. The shipped bias is --bias-from cv-mean, the mean of the five held-out-fold estimates (a
    single 33-item fold swings it by about 0.6), each from the same configuration that ships.
    """
    udhr = caches["udhr"]
    uf = fold_of(udhr, item_folds(udhr.item, args.folds, args.seed))
    srcs = PROBES[args.final_probe]
    test = Cache("mist_test", args.layer, args.root, rel=MIST_TEST)
    check_compatible([caches["udhr"], test])
    langs = sorted(set(udhr.qlang))

    def train_and_calibrate(cf, score_test):
        cal = uf == cf
        parts_X, parts_y = [], []
        for s in srcs:
            c = caches[s]
            sel = ~cal if s == "udhr" else c.clean
            parts_X.append(c.X[sel])
            parts_y.append(c.y[sel])
        X_te = np.concatenate([udhr.X[cal], test.X]) if score_test else udhr.X[cal]
        z = tp.fit_logreg(np.concatenate(parts_X), np.concatenate(parts_y), X_te, device,
                          l2=args.l2, return_logits=True).astype(np.float64)
        n = int(cal.sum())
        return z[:n], udhr.y[cal], udhr.qlang[cal], z[n:], int(sum(len(p) for p in parts_y))

    # The sensitivity sweep deliberately does not score the test set: which calibration fold
    # to keep is a choice, and choices are not made with the test set in the room even when
    # its labels are not consulted.
    #
    # Both estimators are printed because the choice between them is decided here rather
    # than by the arms table, where they score the same. F1-optimal thresholds are step
    # functions of the cut point and their maximum is often a broad plateau, so which end of
    # the plateau a fold lands on is close to arbitrary; NLL is smooth and convex in the
    # bias. When two rules perform identically, ship the better-conditioned one.
    print(f"\nsensitivity of the operating point to which fold is held back for calibration")
    print(f"{'calib fold':>10s} {'items':>6s} {'F1 bias':>8s} {'thr':>9s} "
          f"{'NLL bias':>9s} {'thr':>9s}")
    biases, biases_nll, per_lang = {}, {}, {}
    z_test = n_train = None
    for c in range(args.folds):
        want_test = (c == args.calib_fold)
        zc, yc, lc, zt, nt = train_and_calibrate(c, score_test=want_test)
        if want_test:
            z_test, n_train = zt, nt
        _, bias_c = decision_rules(np.zeros(0), np.array([]), zc, yc, lc)
        biases[c], biases_nll[c] = bias_c["f1_global"], bias_c["nll_global"]
        per_lang[c] = bias_c["f1_lang" if args.final_rule.endswith("F1") else "nll_lang"]
        print(f"{c:10d} {len({i for i, s in zip(udhr.item, uf == c) if s}):6d} "
              f"{biases[c]:+8.3f} {1 / (1 + np.exp(biases[c])):9.4f} "
              f"{biases_nll[c]:+9.3f} {1 / (1 + np.exp(biases_nll[c])):9.4f}")
    for name, d in (("F1", biases), ("NLL", biases_nll)):
        v = np.array(list(d.values()))
        print(f"  {name:3s} across folds: mean {v.mean():+.3f}  sd {v.std(ddof=1):.3f}  "
              f"(sd of the mean {v.std(ddof=1) / np.sqrt(len(v)):.3f})  "
              f"range {v.min():+.3f} to {v.max():+.3f}")

    cf = args.calib_fold
    chosen = biases if args.final_rule.endswith("F1") else biases_nll
    b = (float(np.mean(list(chosen.values()))) if args.bias_from == "cv-mean"
         else chosen[cf])
    lang_bias = {g: float(np.mean([per_lang[c][g] for c in per_lang])) for g in langs}
    if args.bias_from == "fold":
        lang_bias = {g: float(v) for g, v in per_lang[cf].items()}
    rules = {
        "global": z_test + b,
        "centred": z_test + np.array(
            [lang_bias[g] - float(np.mean(list(lang_bias.values()))) + b
             for g in test.qlang]),
    }
    thr = float(1 / (1 + np.exp(b)))
    print(f"\nshipping: probe `{args.final_probe}` on {n_train:,} rows "
          f"(TyDi + {int((uf != cf).sum()):,} udhr rows, fold {cf} held out), "
          f"bias from `{args.bias_from}`")
    print(f"  global bias {b:+.4f}  ->  abstain when p(answerable) < {thr:.6f}")
    print(f"  test rows scored: {len(z_test):,}")
    for r, z in rules.items():
        print(f"  abstentions, {r:8s} rule: {int((z < 0).sum()):,} "
              f"({(z < 0).mean():.1%})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        p_test = 1.0 / (1.0 + np.exp(-z_test))
        with open(args.out, "w", encoding="utf-8") as f:
            for m, s in zip(test.meta, p_test):
                f.write(json.dumps({
                    "id": m["id"], "item_id": m.get("item_id"),
                    "question_lang": m["question_lang"], "context_lang": m["passage_lang"],
                    "silver_answerable": m["answerable"], "confidence": m.get("confidence"),
                    "prob_answerable": float(s), "layer": args.layer,
                }, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}  ({len(z_test):,} rows)")

        # Both rules are written in `calibrate_offsets.py`'s format so `build_submission.py`
        # consumes them unchanged -- including the *global* rule, as an offsets file whose
        # every language carries the same bias. Passing a rounded `--threshold` on the
        # command line instead would move a handful of borderline rows, and going through
        # log-odds keeps the two files exactly the two rules that were measured.
        for tag, per in (("global", {g: b for g in langs}), ("centred", lang_bias)):
            off = args.out.with_name(f"{args.out.stem}_offsets_{tag}.json")
            off.write_text(json.dumps({
                "by": "question", "layer": args.layer, "l2": args.l2,
                "rule": f"{tag}F1", "probe": args.final_probe, "sources": srcs,
                "calib_fold": cf, "model": caches["udhr"].cfg.get("model"),
                "prompt_style": caches["udhr"].cfg.get("prompt_style"),
                "global_prior_shift": float(b),
                "question_offsets": {k: float(v) for k, v in per.items()},
                "context_offsets": {},
                    "bias_from": args.bias_from,
                "note": "Biases fitted on held-out udhr calibration folds for a "
                        "probe trained on TyDi + the other udhr folds, then averaged over "
                        "the folds so all 164 items contribute. `global_prior_shift` "
                        "is that probe's global (single-bias) operating point, NOT the "
                        "0.20-threshold prior +1.3863 -- a probe trained on 80/20 data does "
                        "not need the prior correction, and applying it double-counts. "
                        "logit p' = logit p + question_offsets[qlang]; abstain when p' < 0.5. "
                        "Use --offsets-mode centred.",
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {off}")
    return {"probe": args.final_probe, "rule": args.final_rule, "calib_fold": cf,
            "global_bias": float(b), "threshold_p_answerable": thr,
            "bias_by_calib_fold_f1": {str(k): float(v) for k, v in biases.items()},
            "bias_by_calib_fold_nll": {str(k): float(v) for k, v in biases_nll.items()},
            "bias_from": args.bias_from, "per_lang_bias": lang_bias,
            "abstentions": {r: int((z < 0).sum()) for r, z in rules.items()},
            "n_train_rows": n_train, "languages": langs}


# ---------------------------------------------------------------------------------------

def print_table(res):
    print(f"\n{'=' * 108}\nARMS -- pooled out-of-fold on udhr items, "
          f"{res['n_udhr_items']} items / {res['n_udhr_rows']:,} rows\n{'=' * 108}")
    hdr = (f"{'arm':4s} {'probe':17s} {'rule':14s} {'worst3':>7s} {'95% CI':>15s} "
           f"{'macro':>7s} {'micro':>7s} {'sd':>6s} {'rec':>6s} {'abst':>7s} {'AUROC':>7s}  "
           f"{'worst lang':<10s}")
    print(hdr + "\n" + "-" * len(hdr))

    def line(tag, p, r):
        s = res["probes"][p]["rules"][r]
        ci = res["bootstrap"][f"{p}|{r}"]
        print(f"{tag:4s} {p:17s} {r:14s} {s['worst3_f1']:7.3f} "
              f"[{ci['worst3_lo']:.3f},{ci['worst3_hi']:.3f}] "
              f"{s['macro_f1']:7.3f} {s['micro_f1']:7.3f} {s['sd_across_langs']:6.3f} "
              f"{s['recall']:6.3f} {s['abstentions']:7,d} {res['probes'][p]['oof_auroc']:7.4f}  "
              f"{s['worst_lang']} {s['per_lang_f1'][s['worst_lang']]:.3f}")

    for tag, p, r, _ in ARMS:
        line(tag, p, r)
    print("-" * len(hdr))
    print("selection rule (fixed in advance): mean F1 of the bottom three languages, "
          "bootstrap CI over items;\noverlapping intervals are ties and the simpler arm wins."
          "\n\nall (probe x rule) combinations:")
    for p in PROBES:
        for r in RULES:
            line("", p, r)

    for base, tbl in res["bootstrap"]["paired_vs"].items():
        print(f"\n{'=' * 78}\npaired item bootstrap: worst-3 F1 minus `{base}`\n"
              f"     (same resampled items for both arms, so this is the powered test)\n"
              f"{'=' * 78}")
        print(f"{'arm':4s} {'probe':17s} {'rule':14s} {'delta':>7s} {'95% CI':>16s} "
              f"{'P(better)':>10s}")
        for tag, p, r, _ in ARMS:
            d = tbl[f"{p}|{r}"]
            print(f"{tag:4s} {p:17s} {r:14s} {d['delta']:+7.3f} "
                  f"[{d['lo']:+.3f},{d['hi']:+.3f}] {d['p_better']:10.3f}")

    print(f"\n{'=' * 78}\nE2 -- residual per-language offset spread (sd of a free offset "
          f"refit on\n     each arm's own out-of-fold logits)\n{'=' * 78}")
    print(f"{'probe':17s} {'resid sd':>9s} {'vs noise floor':>15s}")
    nf = res["noise_floor"]["tydi_val_offset_sd"]
    for p in PROBES:
        sd = res["probes"][p]["residual_offset_sd"]
        print(f"{p:17s} {sd:9.3f} {sd / nf:14.1f}x")
    print(f"{'(noise floor)':17s} {nf:9.3f} {1.0:14.1f}x   TyDi train -> TyDi validation")

    print(f"\n{'=' * 78}\nE4 -- external validity (AUROC, mean over folds; neither set is "
          f"trained on\n     in the fold it is scored in)\n{'=' * 78}")
    print(f"{'probe':17s} {'TyDi val':>10s} {'Belebele':>10s}")
    for p in PROBES:
        e = res["probes"][p]["external"]
        print(f"{p:17s} {e['tydi_val_auroc']:10.4f} {e['belebele_auroc']:10.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--mode", default="arms", choices=["arms", "lolo", "final"])
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--lolo-probes", default="udhr,tydi+udhr,tydi+udhr+beleb")
    ap.add_argument("--lolo-folds", type=int, default=1,
                    help="how many of the --folds rotation positions to average LOLO over")
    ap.add_argument("--final-probe", default="tydi+udhr", choices=list(PROBES))
    ap.add_argument("--final-rule", default="globalF1", choices=RULES)
    ap.add_argument("--calib-fold", type=int, default=0,
                    help="which fold is held out of the shipped probe's training set")
    ap.add_argument("--bias-from", default="cv-mean", choices=["cv-mean", "fold"],
                    help="cv-mean averages the operating point over all five held-out "
                         "folds (all 164 items); fold uses --calib-fold's alone")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-oof", type=Path, default=None,
                    help="npz of every arm's held-out decision per udhr row (arms mode)")
    args = ap.parse_args()

    if args.folds < 3:
        raise SystemExit("--folds must be >= 3: the rotation needs an evaluation fold, a "
                         "calibration fold and at least one training fold.")
    device = pick_device(args.device)
    print(f"layer {args.layer}, l2 {args.l2}, device {device}\ncaches:")
    caches = {n: Cache(n, args.layer, args.root) for n in SOURCES}
    check_compatible(list(caches.values()))

    if args.mode == "final":
        # writes its own outputs (predictions JSONL + offsets), so `--out` means something
        # different here and the summary goes next to it
        res = run_final(caches, args, device)
        if args.out:
            args.out.with_suffix(".summary.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    if args.mode == "arms":
        res, oof, udhr = run_arms(caches, args, device)
        print_table(res)
        if args.save_oof:
            # Every arm's held-out decision for every udhr row. Re-analysis (a different
            # metric, another paired comparison, a per-language breakdown) then costs
            # seconds instead of a refit, and -- more to the point -- reads the *same*
            # numbers the table above was built from rather than a fresh fit that might
            # differ.
            args.save_oof.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.save_oof, item=udhr.item, qlang=udhr.qlang, clang=udhr.clang,
                y=udhr.y, rules=np.array(RULES), probes=np.array(list(PROBES)),
                **{f"z|{p}": oof[p]["z"] for p in PROBES},
                **{f"{p}|{r}": oof[p]["rules"][r] for p in PROBES for r in RULES})
            print(f"wrote {args.save_oof}")
    else:
        res = run_lolo(caches, args, device)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
