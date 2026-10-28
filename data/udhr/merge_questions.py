#!/usr/bin/env python3
"""Merge the authored and reviewed question files into data/items.json, one row per (passage,
question). Recomputed mechanically: answer_is_span (substring test); source_article must be one of
the passage's articles, else the question is dropped; five questions per passage with exactly one
unanswerable. Passages without a reviewed file fall back to the raw file and are reported.
"""

import argparse
import collections
import difflib
import json
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

QUESTION_TYPES = {"extractive", "ordinal_count", "inferential", "unanswerable"}

_WS_RE = re.compile(r"\s+")
# Sentence-final punctuation across the scripts in play, for the one-sentence check.
_SENT_END = ".!?。！？؟।"


def clean(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def is_span(answer: str, passage: str) -> bool:
    """Verbatim-substring test, tolerant only of case and trailing punctuation."""
    a = clean(answer).rstrip(_SENT_END).strip().lower()
    return bool(a) and a in clean(passage).lower()


def count_sentences(s: str) -> int:
    """Sentence-enders that are actually followed by more text."""
    n = 0
    for i, ch in enumerate(s):
        if ch in _SENT_END and s[i + 1:].strip():
            n += 1
    return n + 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verified-dir", type=Path,
                    default=SCRIPT_DIR / "data/questions_verified")
    ap.add_argument("--raw-dir", type=Path, default=SCRIPT_DIR / "data/questions_raw")
    ap.add_argument("--passages", type=Path, default=SCRIPT_DIR / "data/passages.json")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "data/items.json")
    args = ap.parse_args()

    passages = {p["passage_id"]: p
                for p in json.loads(args.passages.read_text(encoding="utf-8"))["passages"]}

    items, stats = [], collections.Counter()
    unreviewed, dropped = [], []

    for pid in sorted(passages):
        path = args.verified_dir / f"{pid}.json"
        source = "verified"
        if not path.exists():
            path = args.raw_dir / f"{pid}.json"
            source = "raw"
            unreviewed.append(pid)
        if not path.exists():
            dropped.append((pid, "no question file at all"))
            stats["passage_missing"] += 1
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        passage_eng = passages[pid]["text"]["eng"]
        articles = set(passages[pid]["articles"])
        qs = data.get("questions", [])

        if len(qs) != 5:
            stats["passage_wrong_question_count"] += 1
            dropped.append((pid, f"{len(qs)} questions, expected 5"))

        n_unans = sum(1 for q in qs if not q.get("answerable"))
        if n_unans != 1:
            stats["passage_wrong_unanswerable_count"] += 1
            dropped.append((pid, f"{n_unans} unanswerable, expected 1"))

        for k, q in enumerate(qs):
            question = clean(q.get("question", ""))
            answerable = bool(q.get("answerable"))
            answer = q.get("answer")
            answer = clean(answer) if answer else None
            src_art = q.get("source_article")
            src_art = str(src_art) if src_art is not None else None

            if not question:
                stats["dropped_empty_question"] += 1
                continue
            if answerable and not answer:
                stats["dropped_answerable_without_answer"] += 1
                dropped.append((f"{pid}#{k}", "answerable but no answer"))
                continue
            if not answerable and answer:
                # An unanswerable item must not carry a reference answer; the
                # reference for these is the language's abstention literal.
                stats["cleared_answer_on_unanswerable"] += 1
                answer = None
            if answerable and src_art is not None and src_art not in articles:
                stats["dropped_article_not_in_passage"] += 1
                dropped.append((f"{pid}#{k}", f"source_article {src_art} not in {sorted(articles)}"))
                continue
            if answerable and src_art is None:
                stats["answerable_without_article"] += 1

            qtype = q.get("question_type")
            if qtype not in QUESTION_TYPES:
                qtype = "unanswerable" if not answerable else "extractive"
                stats["question_type_defaulted"] += 1
            if not answerable and qtype != "unanswerable":
                qtype = "unanswerable"

            # Recomputed, not trusted.
            span = is_span(answer, passage_eng) if answer else False
            if bool(q.get("answer_is_span")) != span:
                stats["answer_is_span_corrected"] += 1

            n_sent = count_sentences(answer) if answer else 0
            if n_sent > 1:
                stats["answer_multi_sentence"] += 1

            items.append({
                "item_id": f"udhr_i{len(items) + 1:03d}",
                "passage_id": pid,
                "articles": passages[pid]["articles"],
                "source_article": src_art if answerable else None,
                "question": question,
                "question_type": qtype,
                "answerable": answerable,
                "answer": answer,
                "article_specific": bool(q.get("article_specific")) and answerable,
                "answer_is_span": span,
                "answer_n_sentences": n_sent,
                "origin": "authored",
                "review": source,
            })
            stats["kept"] += 1

    # ---- near-duplicate detection -------------------------------------------
    # A consequence of build_passages.py's two overlapping packings: an article
    # appears in a forward-packed passage and a reverse-packed one, two agents
    # author against it independently, and they converge on the same obvious
    # question. Two pairs here are byte-identical in English.
    #
    # Not strictly redundant -- the passages differ, so the retrieval task is not
    # identical -- but the set has less question diversity than its item count
    # suggests, and a duplicated question that also transplants well gets its
    # duplication amplified downstream. Recorded rather than dropped; filter with
    # build_rows.py --drop-near-duplicates.
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            qa, qb = items[a]["question"].lower(), items[b]["question"].lower()
            ratio = difflib.SequenceMatcher(None, qa, qb).ratio()
            if ratio > 0.70:
                items[b].setdefault("near_duplicate_of", items[a]["item_id"])
                items[b]["near_duplicate_ratio"] = round(ratio, 3)
                stats["near_duplicate_pairs"] += 1

    # ---- report --------------------------------------------------------------
    n_unans = sum(1 for i in items if not i["answerable"])
    n_dup = sum(1 for i in items if "near_duplicate_of" in i)
    if n_dup:
        print(f"near-duplicate questions: {n_dup} items flagged "
              f"({stats['near_duplicate_pairs']} pairs > 0.70 similarity)")
        for i in items:
            if "near_duplicate_of" in i:
                print(f"  {i['item_id']} ~ {i['near_duplicate_of']} "
                      f"({i['near_duplicate_ratio']:.2f}): {i['question'][:64]}")
    n_spec = sum(1 for i in items if i["article_specific"])
    print(f"items kept: {len(items)} from {len(passages)} passages")
    print(f"  unanswerable: {n_unans} ({n_unans / max(len(items), 1):.1%})  "
          f"[test set: 20%]")
    print(f"  answerable and article_specific (transplantable): {n_spec}")
    print(f"  answers that are verbatim spans: "
          f"{sum(1 for i in items if i['answer_is_span'])}/"
          f"{sum(1 for i in items if i['answer'])}")

    print(f"\nquestion types:")
    for t, n in collections.Counter(i["question_type"] for i in items).most_common():
        print(f"  {t:14s} {n:3d}")

    print(f"\nper-passage item counts:")
    per = collections.Counter(i["passage_id"] for i in items)
    odd = {p: n for p, n in per.items() if n != 5}
    print(f"  {len(per)} passages, {len(odd)} not equal to 5" +
          (f": {odd}" if odd else ""))

    if unreviewed:
        print(f"\nWARNING: {len(unreviewed)} passages used the UNREVIEWED raw file: "
              f"{', '.join(unreviewed)}")
    if stats:
        print(f"\nmechanical re-checks:")
        for k, v in sorted(stats.items()):
            print(f"  {k:34s} {v}")
    if dropped:
        print(f"\ndropped ({len(dropped)}):")
        for what, why in dropped:
            print(f"  {what}: {why}")

    args.out.write_text(
        json.dumps({
            "n_items": len(items),
            "n_unanswerable": n_unans,
            "n_article_specific": n_spec,
            "items": items,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\nNext: python3 data/udhr/build_transplants.py")


if __name__ == "__main__":
    main()
