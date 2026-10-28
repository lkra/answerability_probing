#!/usr/bin/env python3
"""Assemble one submission file over all 12,775 test ids: qa-context answers with the probe's
abstentions written over them (the refusal string from data/qa_context_templates.json), plus the
sum-sum and qa-oeg outputs. Refuses to write if any id is missing, empty or leaks a chat-role marker
(--allow-incomplete to override; --no-abstentions for raw answers).

    python3 probe/submission/build_full_submission.py --qa-context <file> --sum-sum <file> --qa-oeg <file> --out <file>
"""

import argparse
import collections
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# A leaked chat-role marker means the model kept talking past its turn; anything from the
# marker on is not part of the answer. The generation scripts already strip this, so a hit
# here means one of them was bypassed.
ROLE_LEAK_RE = re.compile(r"\n(?:user|assistant)\b")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", type=Path, default=REPO_ROOT / "data/test/tests.jsonl")
    ap.add_argument("--qa-context", type=Path, required=True)
    ap.add_argument("--sum-sum", type=Path, required=True)
    ap.add_argument("--qa-oeg", type=Path, required=True)
    ap.add_argument("--abstentions", type=Path,
                    default=REPO_ROOT / "probe/runs/qwen35-9b/bare/mist_test_predictions_tydi_udhr.jsonl")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "data/qa_context_templates.json")
    ap.add_argument("--threshold", type=float, default=0.42229130921144387,
                    help="p(answerable) below this abstains. The default is the shipped rule: the TyDi+UDHR "
                         "probe with one global bias (probe/runs/qwen35-9b/bare/*.summary.json), 1,768 abstentions.")
    ap.add_argument("--no-abstentions", action="store_true",
                    help="Submit raw qa-context generations with no abstention overlay.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Write the file even if ids are missing (prints them). Off by "
                         "default - an incomplete submission scores zero on what it omits.")
    args = ap.parse_args()

    tests = load_jsonl(args.tests)
    outputs = {}
    for task, path in [("qa-context", args.qa_context), ("sum-sum", args.sum_sum),
                       ("qa-oeg", args.qa_oeg)]:
        rows = load_jsonl(path)
        outputs.update({r["id"]: r["output"] for r in rows})
        print(f"{task:11s} {len(rows):6,d} rows from {path.name}")

    # ---- abstention overlay on qa-context ----
    n_abstain = 0
    if not args.no_abstentions:
        probs = {r["id"]: r["prob_answerable"] for r in load_jsonl(args.abstentions)}
        literals = {k: v["not_answerable"] for k, v in
                    json.loads(args.templates.read_text(encoding="utf-8"))["templates"].items()}
        no_score = [r["id"] for r in tests if r["task"] == "qa-context" and r["id"] not in probs]
        if no_score:
            raise SystemExit(f"{len(no_score)} qa-context rows have no probe score "
                             f"(e.g. {no_score[:3]}). Re-run the probe or pass --no-abstentions.")
        for row in tests:
            if row["task"] != "qa-context":
                continue
            if probs[row["id"]] < args.threshold:
                outputs[row["id"]] = literals[row["question_lang"]]
                n_abstain += 1
        print(f"abstentions  {n_abstain:6,d} qa-context rows at threshold {args.threshold}")

    # ---- completeness and sanity ----
    missing = [r["id"] for r in tests if r["id"] not in outputs]
    empty = [r["id"] for r in tests if not outputs.get(r["id"], "").strip()]
    leaked = [r["id"] for r in tests if ROLE_LEAK_RE.search(outputs.get(r["id"], ""))]
    extra = set(outputs) - {r["id"] for r in tests}

    print(f"\ntest ids {len(tests):,}   covered {len(tests)-len(missing):,}")
    for label, ids in [("MISSING", missing), ("EMPTY", empty), ("ROLE-LEAK", leaked)]:
        if ids:
            by_task = collections.Counter(
                r["task"] for r in tests if r["id"] in set(ids))
            print(f"  {label}: {len(ids):,}  {dict(by_task)}  e.g. {ids[:3]}")
    if extra:
        print(f"  note: {len(extra):,} prediction id(s) not in the test set (ignored)")
    if missing and not args.allow_incomplete:
        raise SystemExit("refusing to write an incomplete submission; "
                         "--allow-incomplete to override")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in tests:
            f.write(json.dumps({"id": row["id"], "output": outputs.get(row["id"], "")},
                               ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}  ({len(tests):,} lines)")


if __name__ == "__main__":
    main()
