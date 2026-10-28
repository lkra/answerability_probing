#!/usr/bin/env python3
"""Propose transplanted unanswerable items: a question flagged article_specific paired with a
passage that does not contain its article. Candidates are grouped by target passage for a
verification pass; nothing becomes data without it.

Output: data/transplant_jobs/<target_passage_id>.json
"""

import argparse
import collections
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=SCRIPT_DIR / "data/items.json")
    ap.add_argument("--passages", type=Path, default=SCRIPT_DIR / "data/passages.json")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data/transplant_jobs")
    ap.add_argument("--max-per-target", type=int, default=40,
                    help="Cap on candidate questions proposed against one passage, so a "
                         "single reviewer's job stays a job it can do carefully.")
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))["items"]
    passages = json.loads(args.passages.read_text(encoding="utf-8"))["passages"]

    pool = [i for i in items if i["answerable"] and i["article_specific"]
            and i["source_article"]]
    print(f"transplantable questions: {len(pool)} of {len(items)} items "
          f"(answerable and article_specific)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.json"):
        old.unlink()

    total, per_source = 0, collections.Counter()
    print(f"\n{'target':14s} {'articles':20s} {'candidates':>10s}")
    print("-" * 50)
    for p in passages:
        target_articles = set(p["articles"])
        cands = [i for i in pool if i["source_article"] not in target_articles]
        # Spread the cap across source passages rather than taking a prefix, so
        # no single passage's questions dominate one reviewer's job.
        if len(cands) > args.max_per_target:
            by_src = collections.defaultdict(list)
            for i in cands:
                by_src[i["passage_id"]].append(i)
            spread, k = [], 0
            while len(spread) < args.max_per_target:
                added = False
                for src in sorted(by_src):
                    if k < len(by_src[src]) and len(spread) < args.max_per_target:
                        spread.append(by_src[src][k])
                        added = True
                if not added:
                    break
                k += 1
            cands = spread

        for i in cands:
            per_source[i["passage_id"]] += 1
        total += len(cands)

        (args.out_dir / f"{p['passage_id']}.json").write_text(
            json.dumps({
                "target_passage_id": p["passage_id"],
                "target_articles": p["articles"],
                "target_passage": p["text"]["eng"],
                "candidates": [{
                    "item_id": i["item_id"],
                    "question": i["question"],
                    "source_passage_id": i["passage_id"],
                    "source_article": i["source_article"],
                    "question_type": i["question_type"],
                    "answer_in_source": i["answer"],
                } for i in cands],
            }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

        print(f"{p['passage_id']:14s} {'+'.join(p['articles']):20s} {len(cands):10d}")

    print(f"\n{total} candidate pairs across {len(passages)} target passages")
    print(f"each transplantable question is proposed against "
          f"{total / max(len(pool), 1):.1f} targets on average")
    print(f"wrote {args.out_dir}/*.json")
    print("\nNext: the transplant verification fan-out (see README.md), then "
          "merge_transplants.py")


if __name__ == "__main__":
    main()
