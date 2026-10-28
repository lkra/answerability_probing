#!/usr/bin/env python3
"""Write the qa-context abstention mask: one {"id", "output"} line per test row, the language's
refusal string where the probe abstains and "" otherwise.

Rule: abstain when p(answerable) < --threshold (default 0.4223, the shipped TyDi+UDHR probe's
global bias of +0.3134), or with --offsets when logit p + b[question_lang] < 0, b from
calibrate_offsets.py (--offsets-mode centred keeps the global level and applies only per-language
deviations; fitted applies the raw offsets). The shipped mask was built with the global offsets
file. Refusal strings come from data/qa_context_templates.json.

    python3 probe/submission/build_submission.py --predictions <file> --offsets <offsets.json> --out <file>
"""

import argparse
import collections
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", type=Path, default=REPO_ROOT / "data/test/tests.jsonl",
                    help="the organizers' test release")
    ap.add_argument("--predictions", type=Path,
                    default=REPO_ROOT / "probe/runs/qwen35-9b/bare/mist_test_predictions_tydi_udhr.jsonl",
                    help="per-row probe scores from probing/eval_mist_test.py --out")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "data/qa_context_templates.json")
    ap.add_argument("--threshold", type=float, default=0.42229130921144387)
    ap.add_argument("--offsets", type=Path, default=None,
                    help="per-language log-odds offsets from probing/calibrate_offsets.py")
    ap.add_argument("--offsets-mode", choices=["centred", "fitted"], default="centred")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_path = args.out or SCRIPT_DIR / "qa-context-abstentions.jsonl"

    tests = [r for r in load_jsonl(args.tests) if r["task"] == "qa-context"]
    preds = {r["id"]: r for r in load_jsonl(args.predictions)}
    literals = {k: v["not_answerable"]
                for k, v in json.loads(args.templates.read_text(encoding="utf-8"))["templates"].items()}

    missing = [r["id"] for r in tests if r["id"] not in preds]
    if missing:
        raise SystemExit(f"{len(missing)} test ids have no probe score, e.g. {missing[:3]} -- "
                         f"the submission would be scored on a partial system. Re-run "
                         f"probing/eval_mist_test.py over the full test cache first.")

    decide = None
    if args.offsets:
        spec = json.loads(args.offsets.read_text(encoding="utf-8"))
        if spec.get("by") != "question":
            raise SystemExit(f"offsets are grouped by {spec.get('by')!r}; only 'question' can be "
                             f"applied here. Context-language offsets did not survive "
                             f"item-disjoint validation -- see calibrate_offsets.py.")
        b = dict(spec["question_offsets"])
        shift = spec["global_prior_shift"]
        if args.offsets_mode == "centred":
            mean_b = sum(b.values()) / len(b)
            b = {k: v - mean_b + shift for k, v in b.items()}
        missing = {r["question_lang"] for r in tests} - set(b)
        if missing:
            raise SystemExit(f"no offset for {sorted(missing)} -- refusing to silently fall back "
                             f"to the global shift for some languages and not others, which "
                             f"would make the per-language comparison meaningless.")

        def decide(row, pred):
            # log-odds space: the offsets are additive there, and the stored probabilities
            # are unsaturated (5e-06 .. 0.99999) so inverting the sigmoid is lossless enough.
            p = min(max(pred["prob_answerable"], 1e-12), 1 - 1e-12)
            return math.log(p / (1 - p)) + b[row["question_lang"]] < 0

    n_abstain = collections.Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tests:
            pred = preds[row["id"]]
            unanswerable = (decide(row, pred) if decide
                            else pred["prob_answerable"] < args.threshold)
            if unanswerable:
                n_abstain[row["question_lang"]] += 1
            output = literals[row["question_lang"]] if unanswerable else ""
            f.write(json.dumps({"id": row["id"], "output": output}, ensure_ascii=False) + "\n")

    total = sum(n_abstain.values())
    print(f"Wrote {out_path}  ({len(tests):,} lines)")
    rule = (f"per-language offsets ({args.offsets_mode}) from {args.offsets.name}"
            if args.offsets else f"global threshold {args.threshold}")
    print(f"  abstentions: {total:,} of {len(tests):,} rows ({total / len(tests):.1%}), {rule}")
    print(f"  blank (answer still to be generated): {len(tests) - total:,}")
    print(f"\n  {'lang':6s} {'abstain':>8s}")
    for lang in sorted({r['question_lang'] for r in tests}):
        print(f"  {lang:6s} {n_abstain[lang]:8,d}")


if __name__ == "__main__":
    main()
