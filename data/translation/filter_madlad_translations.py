#!/usr/bin/env python3
r"""Filter the MADLAD-400 job translations for Bhojpuri, Central Kurdish, Finnish, Haitian Creole,
Slovak and Yoruba. A row is dropped when its answer fails any check; a failing question alone is
kept unless --drop-on-question-fail.

Checks: empty output; repetition loop; a known fixed hallucination; copy-through when source and
target scripts differ; target/source length ratio outside [LEN_RATIO_LO, LEN_RATIO_HI]; the answer
not recoverable from the answer sentence (character 4-gram recall below ANSWER_CONTAINMENT_MIN); the
same answer string more than MAX_PER_ANSWER times per language. --calibrate reports the check
distributions on the Hy-MT2 output instead of filtering.

    python3 data/translation/filter_madlad_translations.py \
        --translations-dir <madlad output> --source-dir data/translated/to-translate --out-dir <dir>
    # writes <dir>/kept/<lang>.jsonl and <dir>/excluded/<lang>.jsonl (excluded rows carry _excluded_because)
"""

import argparse
import collections
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent  # data/

# The six languages hymt2 cannot reach (translate_questions.py MADLAD_EXTRA_LANGUAGES),
# mapped to their to-translate/ FLORES filename.
MADLAD_ONLY = {
    "Bhojpuri": "bho_Deva",
    "Central Kurdish": "ckb_Arab",
    "Finnish": "fin_Latn",
    "Haitian Creole": "hat_Latn",
    "Slovak": "slk_Latn",
    "Yoruba": "yor_Latn",
}

# --- thresholds (calibrate with --calibrate before trusting) ---
ANSWER_CONTAINMENT_MIN = 0.50   # char-4gram recall of answer within answer_sentence
LEN_RATIO_LO, LEN_RATIO_HI = 0.25, 4.0
REPEAT_WINDOW = 12              # char window for the repetition-loop detector
REPEAT_MAX = 5                  # a window occurring this many times is a loop
MAX_PER_ANSWER = 20             # frequency cap per distinct answer string, per language

# data/translation/README.md documents a fixed hallucination emitted regardless of the actual question,
# reproducibly triggered by Telugu-sourced village-area/measurement questions.
HALLUCINATION_PATTERNS = [
    re.compile(r"how many values in the set", re.I),
    re.compile(r"\{\s*1\s*,\s*2\s*,\s*3", re.I),
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def char_ngrams(text, n=4):
    text = re.sub(r"\s+", " ", text.strip())
    return collections.Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))


def containment(needle, haystack, n=4):
    """Fraction of `needle`'s char-ngrams present in `haystack`. Fuzzy by construction, so
    inflectional differences between an answer in isolation and the same answer inside a
    sentence do not force a rejection (matters most for Finnish/Slovak)."""
    a, b = char_ngrams(needle, n), char_ngrams(haystack, n)
    if not a:
        return 1.0
    hit = sum(min(c, b.get(g, 0)) for g, c in a.items())
    return hit / sum(a.values())


def is_repetitive(text, window=REPEAT_WINDOW, max_rep=REPEAT_MAX):
    t = re.sub(r"\s+", " ", text.strip())
    if len(t) < window * 3:
        return False
    counts = collections.Counter(t[i:i + window] for i in range(len(t) - window))
    return bool(counts) and counts.most_common(1)[0][1] >= max_rep


def dominant_script(text):
    names = []
    for ch in text:
        if ch.isalpha():
            try:
                names.append(unicodedata.name(ch).split()[0])
            except ValueError:
                pass
    return collections.Counter(names).most_common(1)[0][0] if names else None


def load_sources(source_dir, flores_code):
    """to-translate/<code>.tsv - qid -> source text. csv.reader with the default dialect
    handles the embedded newlines inside quoted fields that a naive split('\\t') would
    shred (several answer_sentence rows contain literal newlines)."""
    path = Path(source_dir) / f"{flores_code}.tsv"
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("qid"):
                out[row["qid"]] = row.get("question") or ""
    return out


def evaluate_language(rows, sources):
    """Attach per-row check results. Returns (rows_by_item, stats)."""
    by_item = collections.defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r.get("field") or r["id"].split("__")[-1]] = r

    answer_freq = collections.Counter(
        r["question"].strip() for r in rows
        if (r.get("field") or r["id"].split("__")[-1]) == "answer"
    )

    for item_id, fields in by_item.items():
        for field, r in fields.items():
            text = (r.get("question") or "").strip()
            src = sources.get(r["id"], "")
            reasons = []

            if not text:
                reasons.append("empty")
            else:
                if is_repetitive(text):
                    reasons.append("repetition_loop")
                if any(p.search(text) for p in HALLUCINATION_PATTERNS):
                    reasons.append("hallucination_pattern")
                if src:
                    if text == src.strip() and dominant_script(text) == dominant_script(src) \
                       and len(text) > 12:
                        # only meaningful when the script should have changed; a numeric or
                        # named-entity answer may legitimately be identical
                        pass
                    if text == src.strip() and dominant_script(src) not in (None, dominant_script(text)):
                        reasons.append("copy_through")
                    ratio = len(text) / max(1, len(src.strip()))
                    if not (LEN_RATIO_LO <= ratio <= LEN_RATIO_HI):
                        reasons.append(f"len_ratio={ratio:.2f}")
            r["_reasons"] = reasons

        ans, sent = fields.get("answer"), fields.get("answer_sentence")
        if ans and sent and ans.get("question") and sent.get("question"):
            c = containment(ans["question"], sent["question"])
            ans["_containment"] = c
            if c < ANSWER_CONTAINMENT_MIN:
                ans["_reasons"].append(f"answer_not_in_sentence={c:.2f}")
        elif ans:
            ans["_containment"] = None

        if ans and ans.get("question"):
            if answer_freq[ans["question"].strip()] > MAX_PER_ANSWER:
                ans["_reasons"].append(
                    f"answer_repeated_x{answer_freq[ans['question'].strip()]}")

    return by_item


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--translations-dir", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path,
                    default=DATA_DIR / "translated/to-translate")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Writes kept/<lang>.jsonl and excluded/<lang>.jsonl. Omit for a "
                         "report-only dry run.")
    ap.add_argument("--languages", nargs="+", default=sorted(MADLAD_ONLY))
    ap.add_argument("--calibrate", action="store_true",
                    help="Report check distributions instead of filtering - point this at the "
                         "trusted hymt2 output to ground the thresholds.")
    ap.add_argument("--drop-on-question-fail", action="store_true",
                    help="Also drop an item whose question failed. Off by default: a noisy "
                         "question with a good answer still supervises the right target.")
    args = ap.parse_args()

    langs = args.languages
    if args.calibrate:
        langs = [p.stem.replace("_", " ").title() for p in
                 sorted(args.translations_dir.glob("*.jsonl"))]

    grand = collections.Counter()
    for lang in langs:
        path = args.translations_dir / f"{lang.lower().replace(' ', '_')}.jsonl"
        if not path.exists():
            path = args.translations_dir / f"{lang.lower().replace(' ', '')}.jsonl"
        if not path.exists():
            print(f"  {lang:18s} NO FILE ({path.name})")
            continue

        rows = list(load_jsonl(path))
        sources = load_sources(args.source_dir, MADLAD_ONLY.get(lang, ""))
        by_item = evaluate_language(rows, sources)

        answers = [f["answer"] for f in by_item.values() if "answer" in f]
        bad = [a for a in answers if a["_reasons"]]
        cont = [a["_containment"] for a in answers if a.get("_containment") is not None]
        cont.sort()

        print(f"\n=== {lang} ===  items={len(by_item)}  answers={len(answers)}")
        if cont:
            print(f"  answer-in-sentence containment: p10={cont[len(cont)//10]:.2f} "
                  f"p50={cont[len(cont)//2]:.2f} p90={cont[int(len(cont)*.9)]:.2f}")
        reasons = collections.Counter(
            re.sub(r"=.*", "", x) for a in answers for x in a["_reasons"])
        for k, v in reasons.most_common():
            print(f"    {k:28s} {v:6d} ({100*v/max(1,len(answers)):.1f}%)")
        keep_n = len(answers) - len(bad)
        print(f"  ANSWERS KEPT: {keep_n}/{len(answers)} ({100*keep_n/max(1,len(answers)):.1f}%)")
        grand[lang] = keep_n

        if args.out_dir and not args.calibrate:
            kept_dir, exc_dir = args.out_dir / "kept", args.out_dir / "excluded"
            kept_dir.mkdir(parents=True, exist_ok=True)
            exc_dir.mkdir(parents=True, exist_ok=True)
            fn = f"{lang.lower().replace(' ', '_')}.jsonl"
            with open(kept_dir / fn, "w", encoding="utf-8") as fk, \
                 open(exc_dir / fn, "w", encoding="utf-8") as fx:
                for item_id, fields in by_item.items():
                    ans = fields.get("answer")
                    drop = bool(ans and ans["_reasons"]) or ans is None
                    if args.drop_on_question_fail and fields.get("question", {}).get("_reasons"):
                        drop = True
                    for field, r in fields.items():
                        out = {k: v for k, v in r.items() if not k.startswith("_")}
                        if drop:
                            out["_excluded_because"] = (ans or {}).get("_reasons", ["no_answer_row"])
                        (fx if drop else fk).write(
                            json.dumps(out, ensure_ascii=False) + "\n")
            print(f"  wrote {kept_dir/fn} and {exc_dir/fn}")

    if grand:
        print(f"\nTOTAL answers kept: {sum(grand.values()):,}")


if __name__ == "__main__":
    main()
