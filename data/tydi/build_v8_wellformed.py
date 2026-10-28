#!/usr/bin/env python3
"""Make the adaptive-length targets well-formed (Adapter 1's training file): uppercase a lowercase
first letter; append the question language's sentence terminator where the completion does not
already end in a terminator or a closing bracket or quote. The terminator per language is the most
common final mark among that language's already-terminated completions (danda for ben and hin, the
ideographic full stop for jpn and zho, the period otherwise); nothing is read from the references.

    python3 data/tydi/build_v8_wellformed.py   # data/built/qa_context_tydi_adaptive_train.jsonl -> qa_context_v8_train.jsonl (+ stats)
"""

import argparse
import collections
import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # data/

# Sentence-final punctuation across the 24 question languages' scripts. Used both
# for "is this already terminated?" and as the candidate set when inferring each
# language's terminator.
TERMINATORS = set(".!?。！？।॥؟…")
# Not terminators, but a completion ending in one of these is surface-complete
# and must not get a period glued after it.
CLOSERS = set(")]}\"'”’»」』】〕〉》")
DEFAULT_TERMINATOR = "."


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_terminated(text: str) -> bool:
    return bool(text) and (text[-1] in TERMINATORS or text[-1] in CLOSERS)


def needs_capital(text: str) -> bool:
    """A leading lowercase letter that actually has an uppercase form. A few
    completions open with IPA (e.g. 'ʕ'), which str.islower() reports as
    lowercase but which has no case pair; those are left as they are."""
    return bool(text) and text[0].islower() and text[0].upper() != text[0]


def infer_terminators(rows: list[dict]) -> dict[str, str]:
    """Per question language, the most common final mark among completions that
    already end in one. Languages with no terminated completion at all (does
    not happen on any file this project has built) fall back to '.'."""
    tally: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        c = r["completion"]
        if c and c[-1] in TERMINATORS:
            tally[r["question_lang"]][c[-1]] += 1
    return {lang: counts.most_common(1)[0][0] for lang, counts in tally.items()}


def normalise(completion: str, terminator: str) -> tuple[str, bool, bool]:
    """Returns (new_completion, capitalised, terminated) flags for the stats."""
    capitalised = terminated = False
    if needs_capital(completion):
        completion = completion[0].upper() + completion[1:]
        capitalised = True
    if not is_terminated(completion):
        completion = completion + terminator
        terminated = True
    return completion, capitalised, terminated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "built/qa_context_tydi_adaptive_train.jsonl",
                    help="qa-context training file whose targets to normalise (the adaptive-length build).")
    ap.add_argument("--out", type=Path, default=ROOT / "built/qa_context_v8_train.jsonl")
    ap.add_argument("--stats-out", type=Path, default=ROOT / "built/qa_context_v8_stats.json")
    args = ap.parse_args()

    rows = load_jsonl(args.inp)
    print(f"Loaded {len(rows):,} rows from {args.inp}")

    terminators = infer_terminators(rows)
    langs = sorted({r["question_lang"] for r in rows})
    for lang in langs:
        terminators.setdefault(lang, DEFAULT_TERMINATOR)
    non_default = {l: t for l, t in sorted(terminators.items()) if t != DEFAULT_TERMINATOR}
    print(f"Inferred terminators: {non_default}; all other languages '{DEFAULT_TERMINATOR}'")

    before_term = sum(is_terminated(r["completion"]) for r in rows)
    before_cap = sum(not needs_capital(r["completion"]) for r in rows)

    n_changed = n_cap = n_term = 0
    changed_by_lang = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            new, capitalised, terminated = normalise(r["completion"], terminators[r["question_lang"]])
            if new != r["completion"]:
                n_changed += 1
                changed_by_lang[r["question_lang"]] += 1
            n_cap += capitalised
            n_term += terminated
            r["completion"] = new
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    after_term = sum(is_terminated(r["completion"]) for r in rows)
    after_cap = sum(not needs_capital(r["completion"]) for r in rows)
    assert after_term == len(rows) and after_cap == len(rows), "normalisation left an ill-formed row"

    print(f"\nWrote {len(rows):,} rows to {args.out}")
    print(f"Changed {n_changed:,} rows ({n_changed / len(rows):.1%}): "
          f"{n_cap:,} capitalised, {n_term:,} given a terminator")
    print(f"Terminated: {before_term / len(rows):.1%} -> {after_term / len(rows):.1%}   "
          f"capitalised: {before_cap / len(rows):.1%} -> {after_cap / len(rows):.1%}")

    if args.stats_out is not None:
        stats = {
            "in": os.path.relpath(args.inp, ROOT.parent),
            "out": os.path.relpath(args.out, ROOT.parent),
            "rows": len(rows),
            "changed": n_changed,
            "capitalised": n_cap,
            "terminator_appended": n_term,
            "terminated_share_before": round(before_term / len(rows), 4),
            "capitalised_share_before": round(before_cap / len(rows), 4),
            "terminators": dict(sorted(terminators.items())),
            "changed_by_question_lang": dict(sorted(changed_by_lang.items())),
        }
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats to {args.stats_out}")


if __name__ == "__main__":
    main()
