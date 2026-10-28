#!/usr/bin/env python3
"""Merge the localisation output and measure grounding against the official target-language
passage: exact_span (verbatim substring) and coverage (character 4-gram fraction, threshold 0.8;
not comparable across scripts). Answers whose English original was a span but whose localisation
is not are listed in repair_jobs.json.

Output: data/localised.json, data/repair_jobs.json, data/localisation_report.json
"""

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

COVERAGE_THRESHOLD = 0.8
NGRAM = 4

_WS_RE = re.compile(r"\s+")
_SENT_END = ".!?。！？؟।"


def clean(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def ngrams(s: str, n: int = NGRAM) -> set[str]:
    s = _WS_RE.sub(" ", s.lower())
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 0))} or {s}


def coverage(answer: str, passage: str) -> float:
    a = ngrams(answer)
    return len(a & ngrams(passage)) / len(a)


def is_exact_span(answer: str, passage: str) -> bool:
    a = clean(answer).rstrip(_SENT_END).strip().lower()
    return bool(a) and a in clean(passage).lower()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", type=Path,
                    default=SCRIPT_DIR / "data/translation_outputs")
    ap.add_argument("--inputs-dir", type=Path,
                    default=SCRIPT_DIR / "data/translation_inputs")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data")
    args = ap.parse_args()

    files = sorted(args.outputs_dir.glob("*.json"))
    expected = sorted(args.inputs_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"error: no localisation output in {args.outputs_dir}")
    missing = {p.name for p in expected} - {p.name for p in files}
    if missing:
        print(f"WARNING: {len(missing)} job files have no output: "
              f"{', '.join(sorted(missing))}")

    localised: dict[str, dict] = collections.defaultdict(dict)
    repair, stats = [], collections.Counter()
    per_lang: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        lang = data["lang"]
        for u in data["units"]:
            item_id = u["item_id"]
            q = clean(u.get("question_tgt") or "")
            if not q:
                stats["missing_question"] += 1
                per_lang[lang]["missing_question"] += 1
                continue

            rec = {
                "question": q,
                "answer": None,
                "exact_span": False,
                "coverage": None,
                "uncertain": bool(u.get("uncertain")),
                "note": (u.get("note") or "").strip(),
            }
            per_lang[lang]["units"] += 1
            if rec["uncertain"]:
                per_lang[lang]["uncertain"] += 1

            if u["answerable"]:
                a = clean(u.get("answer_tgt") or "")
                if not a:
                    stats["missing_answer"] += 1
                    per_lang[lang]["missing_answer"] += 1
                    repair.append({**{k: u[k] for k in
                                      ("item_id", "passage_id", "question_eng",
                                       "answer_eng", "answer_is_span", "passage_tgt")},
                                   "lang": lang, "reason": "missing_answer"})
                    continue
                passage_tgt = u["passage_tgt"]
                rec["answer"] = a
                rec["exact_span"] = is_exact_span(a, passage_tgt)
                rec["coverage"] = round(coverage(a, passage_tgt), 4)
                per_lang[lang]["answers"] += 1
                if rec["exact_span"]:
                    per_lang[lang]["exact_span"] += 1
                if rec["coverage"] >= COVERAGE_THRESHOLD:
                    per_lang[lang]["covered"] += 1

                # The defect case: the English answer was a verbatim span, so the
                # target answer should have been located in the official text.
                if u["answer_is_span"] and not rec["exact_span"] \
                        and rec["coverage"] < COVERAGE_THRESHOLD:
                    stats["span_not_located"] += 1
                    per_lang[lang]["span_not_located"] += 1
                    repair.append({
                        "lang": lang, "item_id": item_id,
                        "passage_id": u["passage_id"],
                        "question_eng": u["question_eng"],
                        "answer_eng": u["answer_eng"],
                        "answer_is_span": True,
                        "answer_tgt_current": a,
                        "coverage": rec["coverage"],
                        "passage_tgt": passage_tgt,
                        "reason": "english_answer_was_a_span_but_target_is_not",
                    })

            localised[lang][item_id] = rec
            stats["kept"] += 1

    # ---- report --------------------------------------------------------------
    print(f"languages: {len(localised)}   units kept: {stats['kept']:,}")
    print(f"\n{'lang':6s} {'units':>6s} {'answers':>8s} {'exact span':>11s} "
          f"{'covered>=.8':>12s} {'uncertain':>10s} {'to repair':>10s}")
    print("-" * 72)
    for lang in sorted(per_lang):
        c = per_lang[lang]
        n_a = max(c["answers"], 1)
        print(f"{lang:6s} {c['units']:6d} {c['answers']:8d} "
              f"{c['exact_span']:6d} {c['exact_span'] / n_a:5.0%} "
              f"{c['covered']:6d} {c['covered'] / n_a:5.0%} "
              f"{c['uncertain']:10d} {c['span_not_located']:10d}")

    tot_a = sum(c["answers"] for c in per_lang.values())
    tot_s = sum(c["exact_span"] for c in per_lang.values())
    tot_c = sum(c["covered"] for c in per_lang.values())
    print("-" * 72)
    print(f"{'ALL':6s} {sum(c['units'] for c in per_lang.values()):6d} {tot_a:8d} "
          f"{tot_s:6d} {tot_s / max(tot_a, 1):5.0%} "
          f"{tot_c:6d} {tot_c / max(tot_a, 1):5.0%} "
          f"{sum(c['uncertain'] for c in per_lang.values()):10d} "
          f"{len(repair):10d}")

    if stats:
        print("\ncounts:")
        for k, v in sorted(stats.items()):
            print(f"  {k:24s} {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "localised.json").write_text(
        json.dumps(localised, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "repair_jobs.json").write_text(
        json.dumps({"n": len(repair), "jobs": repair}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    (args.out_dir / "localisation_report.json").write_text(
        json.dumps({
            "coverage_threshold": COVERAGE_THRESHOLD,
            "ngram": NGRAM,
            "per_language": {k: dict(v) for k, v in sorted(per_lang.items())},
            "totals": dict(stats),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out_dir / 'localised.json'}")
    print(f"wrote {args.out_dir / 'repair_jobs.json'} ({len(repair)} units)")
    print(f"wrote {args.out_dir / 'localisation_report.json'}")
    print("\nNext: python3 data/udhr/build_rows.py")


if __name__ == "__main__":
    main()
