#!/usr/bin/env python3
"""Score {"id", "output"} predictions against data/silver/silver_test_labels_final.jsonl: chrF on
answerable rows, abstention on unanswerable rows (the language's refusal string, taken from the
labels), false abstention on answerable rows, off-target language. Rows flagged
reference_unreliable are excluded unless --keep-unreliable.

    python3 adapter/infer/score_qa_context.py --predictions <file> --exclude-lang ckb --by question_lang [--bertscore]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_LABELS = REPO_ROOT / "data" / "silver" / "silver_test_labels_final.jsonl"
# fastText lid.176 (https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz);
# place it here or set LID_MODEL.
DEFAULT_LID_MODEL = Path(os.environ.get("LID_MODEL", REPO_ROOT / "data" / "upstream" / "lid.176.ftz"))

# ISO 639-3 (what the test set uses) -> fasttext lid.176 label. Verified against the
# model's own label set: all 24 task languages are covered, including the low-resource
# ones (ckb, bho, yor, hat) that lighter pure-python detectors omit.
LID_CODE = {
    "arb": "ar", "ben": "bn", "bho": "bh", "ces": "cs", "ckb": "ckb", "deu": "de",
    "eng": "en", "fin": "fi", "hat": "ht", "hin": "hi", "ind": "id", "ita": "it",
    "jpn": "ja", "kor": "ko", "mar": "mr", "pes": "fa", "por": "pt", "rus": "ru",
    "slk": "sk", "spa": "es", "tur": "tr", "vie": "vi", "yor": "yo", "zho": "zh",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_abstention_map(labels):
    """question_lang -> that language's abstention string, taken from the labels."""
    out = {}
    for row in labels:
        if not row["answerable"] and row.get("target_source") == "exact":
            out.setdefault(row["question_lang"], row["target"])
    return out


def normalize(text):
    return " ".join((text or "").split()).strip().strip('."،؟').lower()


def looks_like_abstention(prediction, abstention_string):
    if not abstention_string:
        return False
    return normalize(abstention_string) in normalize(prediction)


def chrf_score(hypotheses, references):
    from sacrebleu.metrics import CHRF

    if not hypotheses:
        return None
    return CHRF().corpus_score(hypotheses, [references]).score


# ---- recall / containment ----------------------------------------------------

def _char_ngrams(text, n=4):
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return [text[i : i + n] for i in range(max(0, len(text) - n + 1))]


def reference_recall(reference, hypothesis, n=4):
    """Fraction of the reference's character n-grams present in the hypothesis. Recall separates a
    verbose but correct answer from a wrong one, which chrF does not.
    """
    ref = _char_ngrams(reference, n)
    if not ref:
        return None
    hyp = set(_char_ngrams(hypothesis, n))
    return sum(1 for g in ref if g in hyp) / len(ref)


# ---- off-target language ------------------------------------------------------

def load_lid(model_path):
    try:
        import fasttext
    except ImportError:
        return None
    if not Path(model_path).exists():
        return None
    try:
        return fasttext.load_model(str(model_path))
    except Exception:
        return None


def detect_lang(lid, text):
    """fasttext label for `text`, or None if undetectable/too short to judge.

    Goes through the raw C++ binding (`lid.f.predict`) rather than the Python wrapper:
    fasttext-wheel's `predict()` calls `np.array(probs, copy=False)`, which numpy 2.x
    rejects outright ("Unable to avoid copy while creating an array as requested").
    The raw binding returns [(prob, label)] and skips that conversion entirely.
    """
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) < 10:
        return None
    try:
        pairs = lid.f.predict(clean, 1, 0.0, "strict")
    except Exception:
        return None
    return pairs[0][1].replace("__label__", "") if pairs else None


def summarize(pairs, label):
    """pairs: list of (hypothesis, reference). chrF + mean reference-recall."""
    if not pairs:
        return f"  {label:22s}      n=0"
    hyps = [p for p, _ in pairs]
    refs = [r for _, r in pairs]
    score = chrf_score(hyps, refs)
    recalls = [r for r in (reference_recall(ref, hyp) for hyp, ref in pairs) if r is not None]
    mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")
    empty = sum(1 for h in hyps if not h.strip())
    extra = f"   empty={empty}" if empty else ""
    return (f"  {label:22s} chrF={score:6.2f}  recall={mean_recall:5.2f}"
            f"   n={len(pairs):5d}{extra}")


def bertscore_f1(hypotheses, references, model_type, batch_size=64):
    """Corpus mean BERTScore F1, or None if unavailable.

    Complements chrF specifically because 6,446 of 8,640 references are themselves model
    translations (silver_translated) - surface n-gram overlap penalizes legitimate
    paraphrase between two independent translations of the same answer, which is a
    property of the labels, not of the system.
    """
    try:
        from bert_score import score as _bertscore
    except ImportError:
        return None
    import torch

    _, _, f1 = _bertscore(
        hypotheses, references, model_type=model_type,
        batch_size=batch_size, verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return f1.mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--by", default=None,
                         help="Also break the content score down by this label field "
                              "(e.g. question_lang, context_lang, target_source).")
    parser.add_argument("--keep-unreliable", action="store_true",
                         help="Include reference_unreliable rows in the headline content score "
                              "(excluded by default - see data/silver/README.md).")
    parser.add_argument("--exclude-lang", nargs="*", default=[],
                         help="question_lang codes to drop from the headline score, e.g. "
                              "'ckb' whose silver references are in the wrong Kurdish variety.")
    parser.add_argument("--lid-model", type=Path, default=DEFAULT_LID_MODEL,
                         help="fasttext lid.176 model for the off-target language check.")
    parser.add_argument("--bertscore", action="store_true",
                         help="Also compute BERTScore F1 (needs a GPU to be quick, and "
                              "downloads the model on first use).")
    parser.add_argument("--bertscore-model", default="xlm-roberta-large",
                         help="Multilingual encoder for --bertscore.")
    args = parser.parse_args()

    labels = {row["id"]: row for row in load_jsonl(args.labels)}
    preds = load_jsonl(args.predictions)

    matched = [(p, labels[p["id"]]) for p in preds if p["id"] in labels]
    missing = len(preds) - len(matched)

    print(f"predictions: {len(preds)}   matched to labels: {len(matched)}"
          + (f"   UNMATCHED: {missing}" if missing else ""))
    if not matched:
        raise SystemExit("error: no predictions matched the label file - wrong file?")

    abstention = build_abstention_map(labels.values())

    answerable, unanswerable = [], []
    for pred, lab in matched:
        (answerable if lab["answerable"] else unanswerable).append((pred, lab))

    langs = {lab["question_lang"] for _, lab in matched}
    print(f"question languages: {len(langs)} ({', '.join(sorted(langs))})")
    print(f"answerable: {len(answerable)}   unanswerable: {len(unanswerable)}")

    # ---- content, on answerable rows ----
    unreliable = [(p, l) for p, l in answerable if l.get("reference_unreliable")]
    scored = answerable if args.keep_unreliable else [
        (p, l) for p, l in answerable if not l.get("reference_unreliable")
    ]
    dropped_lang = [(p, l) for p, l in scored if l["question_lang"] in args.exclude_lang]
    scored = [(p, l) for p, l in scored if l["question_lang"] not in args.exclude_lang]

    print("\nCONTENT (answerable rows) - chrF vs silver reference, recall = fraction of the")
    print("reference's char-4grams present in the prediction (verbosity-insensitive)")
    if unreliable and not args.keep_unreliable:
        print(f"  excluding {len(unreliable)} reference_unreliable row(s); "
              f"--keep-unreliable to include")
    if dropped_lang:
        print(f"  excluding {len(dropped_lang)} row(s) in {sorted(set(args.exclude_lang))}")
    print(summarize([(p["output"], l["target"]) for p, l in scored], "all"))

    if args.bertscore:
        f1 = bertscore_f1([p["output"] for p, _ in scored],
                          [l["target"] for _, l in scored], args.bertscore_model)
        if f1 is None:
            print("  BERTScore: unavailable (pip install bert_score)")
        else:
            print(f"  {'BERTScore F1':22s} {f1:6.4f}   ({args.bertscore_model})")

    if args.by:
        groups = defaultdict(list)
        for p, l in scored:
            groups[l.get(args.by, "?")].append((p["output"], l["target"]))
        print(f"\n  by {args.by}:")
        for key in sorted(groups, key=lambda k: -len(groups[k])):
            print("  " + summarize(groups[key], str(key)))

    # ---- off-target language ----
    lid = load_lid(args.lid_model)
    print("\nOFF-TARGET LANGUAGE (answerable rows: did it answer in the requested language?)")
    if lid is None:
        print(f"  unavailable - need fasttext + {args.lid_model}")
    else:
        # The silver reference is scored through the same detector as a CONTROL. Without it
        # this metric is badly misleading: lid.176 cannot separate Bhojpuri from Hindi, or
        # place Haitian Creole and Yoruba reliably, and reports 95%/90%/79% "off-target" for
        # the *references* in those languages. Reading the prediction column alone showed
        # bho 100% / hat 68% / yor 58% off-target - all detector artifacts, not model errors.
        # Rule of thumb: only a prediction rate well ABOVE the reference rate is a real
        # model problem. (Where the reference rate is high but the prediction rate is low -
        # ckb here - it is the *labels* that are wrong; see data/silver/README.md.)
        per_lang, off_examples = defaultdict(lambda: [0, 0, 0]), []
        for p, l in answerable:
            want = LID_CODE.get(l["question_lang"])
            if want is None:
                continue
            got_p = detect_lang(lid, p["output"])
            got_r = detect_lang(lid, l["target"])
            if got_p is None or got_r is None:
                continue
            row = per_lang[l["question_lang"]]
            row[2] += 1
            if got_p != want:
                row[0] += 1
                off_examples.append((l["question_lang"], got_p))
            if got_r != want:
                row[1] += 1
        tot_off = sum(v[0] for v in per_lang.values())
        tot_ref = sum(v[1] for v in per_lang.values())
        tot_n = sum(v[2] for v in per_lang.values())
        print(f"  overall: prediction {tot_off}/{tot_n} ({100*tot_off/max(1,tot_n):.1f}%)"
              f"   reference {tot_ref}/{tot_n} ({100*tot_ref/max(1,tot_n):.1f}%, control)")
        print(f"  {'lang':5s} {'pred':>7s} {'ref':>7s}   note")
        for lang, (off, ref_off, n) in sorted(per_lang.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][2])):
            if not off and not ref_off:
                continue
            p_pct, r_pct = 100 * off / n, 100 * ref_off / n
            if r_pct > 50 and p_pct < 20:
                note = "LABELS likely wrong (model on-target, refs not)"
            elif r_pct > 50:
                note = "detector unreliable for this language - ignore"
            elif p_pct > r_pct + 20:
                note = "MODEL off-target"
            else:
                note = ""
            got = defaultdict(int)
            for ql, g in off_examples:
                if ql == lang:
                    got[g] += 1
            top = ", ".join(f"{g}:{c}" for g, c in sorted(got.items(), key=lambda kv: -kv[1])[:3])
            print(f"  {lang:5s} {p_pct:6.1f}% {r_pct:6.1f}%   {note}" + (f"  [{top}]" if top and note else ""))

    # ---- abstention, on unanswerable rows ----
    print("\nABSTENTION (unanswerable rows)")
    if unanswerable:
        hits = sum(1 for p, l in unanswerable
                   if looks_like_abstention(p["output"], abstention.get(l["question_lang"])))
        pct = 100 * hits / len(unanswerable)
        print(f"  abstained correctly       {hits}/{len(unanswerable)} ({pct:.1f}%)")
        print("  NOTE: this fine-tune was trained on answerable rows only - abstention is the")
        print("        activation probe's job, so ~0% here is expected, not a regression.")
    else:
        print("  (no unanswerable rows in this prediction file)")

    # ---- false abstention, on answerable rows ----
    false_abstain = sum(1 for p, l in answerable
                        if looks_like_abstention(p["output"], abstention.get(l["question_lang"])))
    print("\nFALSE ABSTENTION (answerable rows wrongly abstaining)")
    print(f"  {false_abstain}/{len(answerable)} ({100*false_abstain/max(1,len(answerable)):.1f}%)")


if __name__ == "__main__":
    main()
