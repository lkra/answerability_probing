#!/usr/bin/env python3
"""Keep the transplant pairs the verification pass accepted at high confidence, capped per source
question and per target passage by interleaving so that no question or passage dominates.

Output: data/transplants.json
"""

import argparse
import collections
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verified-dir", type=Path,
                    default=SCRIPT_DIR / "data/transplants_verified")
    ap.add_argument("--jobs-dir", type=Path, default=SCRIPT_DIR / "data/transplant_jobs")
    ap.add_argument("--items", type=Path, default=SCRIPT_DIR / "data/items.json")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "data/transplants.json")
    ap.add_argument("--max-per-question", type=int, default=3)
    ap.add_argument("--max-per-target", type=int, default=8)
    ap.add_argument("--min-confidence", choices=["high", "medium", "low"], default="high")
    args = ap.parse_args()

    items = {i["item_id"]: i for i in
             json.loads(args.items.read_text(encoding="utf-8"))["items"]}

    rank = {"high": 3, "medium": 2, "low": 1}
    floor = rank[args.min_confidence]

    files = sorted(args.verified_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"error: no verified transplants in {args.verified_dir}")
    expected = {p.stem for p in args.jobs_dir.glob("*.json")}
    missing = expected - {p.stem for p in files}
    if missing:
        print(f"WARNING: {len(missing)} target passages unreviewed: "
              f"{', '.join(sorted(missing))}")

    survivors, stats = [], collections.Counter()
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        target = data["target_passage_id"]
        for j in data.get("judgements", []):
            stats["judged"] += 1
            item_id = j.get("item_id")
            if item_id not in items:
                stats["unknown_item"] += 1
                continue
            if not j.get("unanswerable"):
                stats["rejected_answerable"] += 1
                continue
            conf = (j.get("confidence") or "low").lower()
            if rank.get(conf, 0) < floor:
                stats[f"rejected_confidence_{conf}"] += 1
                continue
            if items[item_id]["passage_id"] == target:
                stats["rejected_same_passage"] += 1
                continue
            survivors.append({
                "item_id": item_id,
                "target_passage_id": target,
                "source_passage_id": items[item_id]["passage_id"],
                "source_article": items[item_id]["source_article"],
                "question_type": items[item_id]["question_type"],
                "confidence": conf,
                "reason": (j.get("reason") or "").strip(),
            })
            stats["survived"] += 1

    print(f"judged {stats['judged']:,} candidate pairs")
    for k, v in sorted(stats.items()):
        if k != "judged":
            print(f"  {k:30s} {v:5d}")
    print(f"survival rate: {stats['survived'] / max(stats['judged'], 1):.1%}")

    # ---- balance -------------------------------------------------------------
    by_q: dict[str, list] = collections.defaultdict(list)
    for s in survivors:
        by_q[s["item_id"]].append(s)
    for k in by_q:
        by_q[k].sort(key=lambda s: s["target_passage_id"])

    kept, per_q, per_t = [], collections.Counter(), collections.Counter()
    round_i = 0
    while True:
        added = False
        for q in sorted(by_q):
            if per_q[q] >= args.max_per_question or round_i >= len(by_q[q]):
                continue
            s = by_q[q][round_i]
            if per_t[s["target_passage_id"]] >= args.max_per_target:
                continue
            kept.append(s)
            per_q[q] += 1
            per_t[s["target_passage_id"]] += 1
            added = True
        if not added:
            break
        round_i += 1

    for k, s in enumerate(kept):
        s["transplant_id"] = f"udhr_t{k + 1:03d}"

    print(f"\nkept {len(kept)} of {len(survivors)} survivors after balancing "
          f"(<= {args.max_per_question}/question, <= {args.max_per_target}/target)")
    print(f"  distinct source questions: {len(per_q)}")
    print(f"  distinct target passages:  {len(per_t)}")
    if per_t:
        print(f"  per target: min {min(per_t.values())}, max {max(per_t.values())}")

    args.out.write_text(json.dumps({
        "n_transplants": len(kept),
        "n_survivors": len(survivors),
        "min_confidence": args.min_confidence,
        "caps": {"per_question": args.max_per_question, "per_target": args.max_per_target},
        "transplants": kept,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\nNext: python3 data/udhr/build_translation_inputs.py")


if __name__ == "__main__":
    main()
