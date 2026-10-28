#!/usr/bin/env python3
"""Prepare the question-authoring jobs: one per passage, with its English text and per-article
breakdown (each authored question must name its article), plus exemplars: the five test passages
that are UDHR with their questions and silver labels, shown to the authoring model for style.

Output: data/design_exemplars.json, data/question_jobs.json
"""

import argparse
import collections
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passages", type=Path, default=SCRIPT_DIR / "data/passages.json")
    ap.add_argument("--units", type=Path, default=SCRIPT_DIR / "data/udhr_units.json")
    ap.add_argument("--alignment", type=Path, default=SCRIPT_DIR / "data/udhr_alignment.json")
    ap.add_argument("--silver-items", type=Path,
                    default=REPO_ROOT / "silver/silver_items.json")
    ap.add_argument("--silver-labels", type=Path,
                    default=REPO_ROOT / "silver/silver_labels.json")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data")
    args = ap.parse_args()

    passages = json.loads(args.passages.read_text(encoding="utf-8"))
    units = json.loads(args.units.read_text(encoding="utf-8"))["eng"]
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    silver_items = {i["item_id"]: i for i in
                    json.loads(args.silver_items.read_text(encoding="utf-8"))}
    silver_labels = json.loads(args.silver_labels.read_text(encoding="utf-8"))["labels"]

    # ---- exemplars: the 5 UDHR test passages, with their real questions ------
    by_passage: dict[str, list] = collections.defaultdict(list)
    for item_id, a in alignment["items"].items():
        by_passage[tuple(a["units"])].append(item_id)

    exemplars = []
    for unit_ids, item_ids in sorted(by_passage.items(),
                                     key=lambda kv: (kv[0][0] != "P.1", kv[0][0])):
        item_ids.sort()
        questions = []
        for iid in item_ids:
            lab = silver_labels[iid]
            questions.append({
                "question": silver_items[iid]["english_question"],
                "answerable": lab["answerable"],
                "answer": lab["answer"],
                "silver_confidence": lab["confidence"],
            })
        exemplars.append({
            "source": "+".join(unit_ids),
            "passage": silver_items[item_ids[0]]["english_passage"],
            "n_questions": len(questions),
            "n_unanswerable": sum(1 for q in questions if not q["answerable"]),
            "questions": questions,
        })

    n_q = sum(e["n_questions"] for e in exemplars)
    n_un = sum(e["n_unanswerable"] for e in exemplars)
    print(f"exemplars: {len(exemplars)} UDHR test passages, {n_q} real questions, "
          f"{n_un} unanswerable ({n_un / n_q:.0%})")
    for e in exemplars:
        print(f"  {e['source']:22s} {e['n_questions']}q, "
              f"{e['n_unanswerable']} unanswerable")

    (args.out_dir / "design_exemplars.json").write_text(
        json.dumps({
            "_about": "The organisers' own questions on the 5 UDHR passages the test "
                      "set uses. Shown to the authoring agents as the design to "
                      "imitate. These passages are excluded from training.",
            "exemplars": exemplars,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # ---- jobs ---------------------------------------------------------------
    jobs = []
    for p in passages["passages"]:
        article_texts = {}
        for a in p["articles"]:
            article_texts[a] = " ".join(
                units[u] for u in p["unit_ids"] if u.split(".")[0] == a)
        jobs.append({
            "passage_id": p["passage_id"],
            "articles": p["articles"],
            "n_chars": p["n_chars_eng"],
            "passage": p["text"]["eng"],
            "article_texts": article_texts,
        })

    (args.out_dir / "question_jobs.json").write_text(
        json.dumps({"n_jobs": len(jobs), "jobs": jobs}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    print(f"\njobs: {len(jobs)} passages, "
          f"{sum(len(j['articles']) for j in jobs)} article slots")
    print(f"wrote {args.out_dir / 'design_exemplars.json'}")
    print(f"wrote {args.out_dir / 'question_jobs.json'}")
    print("\nNext: the question-authoring fan-out (see README.md, 'Generating the "
          "questions'), then merge_questions.py")


if __name__ == "__main__":
    main()
