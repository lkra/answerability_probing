#!/usr/bin/env python3
"""Lexical-overlap baseline (one-feature logistic regression, two-fold, AUROC) on exactly the rows
of each cache's meta.jsonl: TyDi validation, UDHR, BELEBELE and the cross-lingual set. Cross-
lingually two values: presented (overlap on the pair shown) and bar (overlap with the question in
the passage's language); bar is what the paper reports.

Output: probe/runs/qwen35-9b/bare/lexical_baselines.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_probes import accuracy, auroc, fit_logreg, overlap_feature   # noqa: E402
from eval_crosslingual import belebele_texts                            # noqa: E402

CACHES = {
    "tydi_val": "probe/caches/qwen35-9b/bare/validation",
    "udhr":     "probe/caches/qwen35-9b/bare/udhr",
    "belebele": "probe/caches/qwen35-9b/bare/belebele",
}
TYDI_DIR = "data/built/tydiqa_answerability"
UDHR_ROWS = "data/udhr/data/udhr_qa_train.jsonl"
BELEBELE_ROWS = "data/built/belebele_answerability_delete_strict.jsonl"


def read_meta(path):
    with open(Path(path) / "meta.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def score(feat, y, device, l2=1e-3, seed=0):
    """Two-fold out-of-sample scores for a one-feature baseline, then its metrics.

    Returns None for a slice with a single class, matching `eval_crosslingual.cell`: AUROC is
    undefined there and a 1.0 accuracy would be reported as if it meant something.
    """
    feat, y = np.asarray(feat, dtype=np.float64), np.asarray(y)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return None
    half = np.random.default_rng(seed).permutation(len(y))
    a, b = half[:len(half) // 2], half[len(half) // 2:]
    pr = np.empty(len(y))
    pr[b] = fit_logreg(feat[a, None], y[a], feat[b, None], device, l2=l2)
    pr[a] = fit_logreg(feat[b, None], y[b], feat[a, None], device, l2=l2)
    frac_pos = float(np.mean(y))
    return {"n": int(len(y)), "acc": accuracy(pr, y), "auroc": auroc(pr, y),
            "majority": max(frac_pos, 1 - frac_pos), "frac_answerable": frac_pos}


def by_group(feat, y, group, device, **kw):
    out = {}
    for g in sorted(set(group)):
        sel = np.asarray(group) == g
        c = score(np.asarray(feat)[sel], np.asarray(y)[sel], device, **kw)
        if c:
            out[g] = c
    return out


# ---------------------------------------------------------------------------------------
# the sets
# ---------------------------------------------------------------------------------------

def tydi_validation(root, device):
    """Monolingual, natural negatives. `presented` and `bar` are the same feature here.

    Scored on the clean rows only, because that is what `train_probe_multi.run_arms` reads
    the probe's `tydi_val_auroc` on -- the organizer-sample rows are ones the shared task
    will score us on. Keeping them would also unbalance the set: the overlap is almost
    entirely answerable rows.
    """
    meta = [m for m in read_meta(root / CACHES["tydi_val"]) if not m["in_organizer_sample"]]
    text = {}
    with open(root / TYDI_DIR / "validation.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            # `context`, not `input`: `input` is context + question, so overlap against it is
            # trivially 1.0 for every row and the baseline silently reports chance.
            text[o["id"]] = o["context"]
    missing = [m["id"] for m in meta if m["id"] not in text]
    if missing:
        raise SystemExit(f"{len(missing)} cache rows have no source passage, e.g. {missing[:3]}")

    y = np.array([m["answerable"] for m in meta])
    lang = np.array([m["lang_code"] for m in meta])
    ov = np.array([overlap_feature(m["question"], text[m["id"]], m["lang_code"]) for m in meta])
    return {
        "rows": len(meta), "cross_lingual_rows": 0,
        "bar_label": "overlap (monolingual: question and passage are the same language)",
        "presented": score(ov, y, device), "bar": score(ov, y, device),
        "by_passage_language": by_group(ov, y, lang, device),
    }


def udhr(root, device):
    """UDHR: 24 question x 25 context languages, authored negatives. `bar` looks up the question
    rendered in the passage language by (question_item_id, passage_lang), since 1,776 of the 3,504
    unanswerable rows are transplants. Rows with `fra` as passage language have no such rendering
    and are excluded from `bar`.
    """
    meta = read_meta(root / CACHES["udhr"])
    raw = {}
    with open(root / UDHR_ROWS, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            raw[o["id"]] = o
    missing = [m["id"] for m in meta if m["id"] not in raw]
    if missing:
        raise SystemExit(f"{len(missing)} cache rows are not in {UDHR_ROWS}, e.g. {missing[:3]}")

    # (question item, language) -> question text. Verified one-to-one: 90 items x 24 languages.
    question_of = {}
    for o in raw.values():
        key = (o["question_item_id"], o["question_lang"])
        q = _udhr_question(root, o)
        if question_of.setdefault(key, q) != q:
            raise SystemExit(f"two different questions for {key}; the bar lookup is not "
                             f"well defined")

    y = np.array([m["answerable"] for m in meta])
    p_lang = np.array([m["passage_lang"] for m in meta])
    q_lang = np.array([m["question_lang"] for m in meta])
    passage = [_udhr_passage(root, raw[m["id"]]) for m in meta]

    presented = np.array([overlap_feature(m["question"], p, m["question_lang"])
                          for m, p in zip(meta, passage)])
    bar, have = [], []
    for m, p in zip(meta, passage):
        same = question_of.get((raw[m["id"]]["question_item_id"], m["passage_lang"]))
        have.append(same is not None)
        bar.append(overlap_feature(same, p, m["passage_lang"]) if same else 0.0)
    bar, have = np.array(bar), np.array(have)
    xl = p_lang != q_lang

    return {
        "rows": len(meta), "cross_lingual_rows": int(xl.sum()),
        "bar_label": "overlap, question localised into the passage's language",
        "bar_rows_dropped": int((~have).sum()),
        "bar_languages_dropped": sorted(set(p_lang[~have].tolist())),
        "presented": score(presented, y, device),
        "bar": score(bar[have], y[have], device),
        "same_language": {"presented": score(presented[~xl], y[~xl], device),
                          "bar": score(bar[~xl & have], y[~xl & have], device)},
        "cross_lingual": {"presented": score(presented[xl], y[xl], device),
                          "bar": score(bar[xl & have], y[xl & have], device)},
        "by_passage_language": by_group(bar[have], y[have], p_lang[have], device),
    }


_UDHR_SPLIT = {}
_TEMPLATES = {}


def _udhr_parts(root, o):
    """(passage, question) recovered from a rendered qa-context prompt.

    Same exact-template split as `_sources.load_rows_udhr`, and a mismatch is an error there
    too: a mis-split moves instruction text into the passage, which the overlap feature would
    happily score.
    """
    if o["id"] in _UDHR_SPLIT:
        return _UDHR_SPLIT[o["id"]]
    if not _TEMPLATES:
        _TEMPLATES.update(json.loads(
            (root / "data" / "qa_context_templates.json")
            .read_text(encoding="utf-8")))
    spec = _TEMPLATES
    t = spec["templates"][o["question_lang"]]
    parts = o["prompt"].split(spec["sep"])
    if (len(parts) != 4 or parts[0] != t["intro"]
            or not parts[2].startswith(t["q_prefix"]) or parts[3] != t["instruction"]):
        raise SystemExit(f"udhr row {o['id']} does not match its qa-context template")
    val = (parts[1], parts[2][len(t["q_prefix"]):])
    _UDHR_SPLIT[o["id"]] = val
    return val


def _udhr_passage(root, o):
    return _udhr_parts(root, o)[0]


def _udhr_question(root, o):
    return _udhr_parts(root, o)[1]


def belebele(root, device):
    """Cross-lingual, delete-negatives, parallel across 26 passage languages.

    Recomputed here only as a consistency check: these two numbers are already in
    `belebele_crosslingual_*.json` and the table quotes them from there. If this file and that
    one disagree, one of the two row selections is wrong.
    """
    meta = read_meta(root / CACHES["belebele"])
    passage_of, question_of = belebele_texts(root / BELEBELE_ROWS)
    y = np.array([m["answerable"] for m in meta])
    p_lang = np.array([m["passage_lang"] for m in meta])
    q_lang = np.array([m["question_lang"] for m in meta])
    presented = np.array([overlap_feature(m["question"], passage_of[m["id"]], m["question_lang"])
                          for m in meta])
    bar = np.array([overlap_feature(question_of.get((m["item_id"], m["passage_lang"]), ""),
                                    passage_of[m["id"]], m["passage_lang"])
                    for m in meta])
    xl = p_lang != q_lang
    return {
        "rows": len(meta), "cross_lingual_rows": int(xl.sum()),
        "bar_label": "overlap, question in the passage's language (human translation)",
        "presented": score(presented, y, device), "bar": score(bar, y, device),
        "same_language": {"presented": score(presented[~xl], y[~xl], device),
                          "bar": score(bar[~xl], y[~xl], device)},
        "cross_lingual": {"presented": score(presented[xl], y[xl], device),
                          "bar": score(bar[xl], y[xl], device)},
        "by_passage_language": by_group(bar[xl], y[xl], p_lang[xl], device),
    }


SETS = {"tydi_val": tydi_validation, "udhr": udhr, "belebele": belebele}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    for name in args.sets.split(","):
        if name not in SETS:
            raise SystemExit(f"unknown set {name!r}; have {list(SETS)}")
        print(f"\n=== {name} ===")
        r = SETS[name](args.root, args.device)
        results[name] = r
        print(f"{r['rows']:,} rows ({r['cross_lingual_rows']:,} cross-lingual)")
        if r.get("bar_rows_dropped"):
            print(f"bar excludes {r['bar_rows_dropped']:,} rows whose passage language "
                  f"{r['bar_languages_dropped']} has no question rendering")
        for k in ("presented", "bar"):
            c = r[k]
            print(f"  {k:10s} n={c['n']:6,d} acc={c['acc']:.3f} AUROC={c['auroc']:.4f}")
        for slice_name in ("same_language", "cross_lingual"):
            if slice_name in r and r[slice_name]["bar"]:
                s = r[slice_name]
                print(f"  {slice_name:14s} presented {s['presented']['auroc']:.4f}  "
                      f"bar {s['bar']['auroc']:.4f}  (n={s['bar']['n']:,})")

    print(f"\n{'=' * 64}\nthe `lexical` column of the main results table\n{'=' * 64}")
    print(f"{'evaluation set':28s} {'AUROC':>7s}  which feature")
    label = {"tydi_val": "TyDi validation", "udhr": "UDHR-QA, 24x25 lg."}
    for name in ("tydi_val", "udhr"):
        if name in results:
            r = results[name]
            print(f"{label[name]:28s} {r['bar']['auroc']:7.3f}  {r['bar_label']}")
    if "belebele" in results:
        for slice_name, lbl in (("same_language", "BELEBELE, same-lang."),
                                ("cross_lingual", "BELEBELE, cross-lingual")):
            print(f"{lbl:28s} {results['belebele'][slice_name]['bar']['auroc']:7.3f}  "
                  f"{results['belebele']['bar_label']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
