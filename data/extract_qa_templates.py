#!/usr/bin/env python3
r"""Recover the organisers' qa-context prompt template per question language from
data/test/tests.jsonl. Every prompt is {intro}\n\n{passage}\n\n{q_prefix}{question}\n\n{instruction}
joined by the literal two-character sequence backslash-n backslash-n (the prompts contain no real
newlines). Writes the templates and the exact "not answerable" string each instruction names.

Output: data/qa_context_templates.json
"""

import argparse
import collections
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# The literal two-character sequence, not a newline. See the module docstring.
SEP = "\\n\\n"

# The "not answerable" literal is always the quoted span in the instruction.
# The organizers use several quote conventions across languages, so match any.
QUOTE_PAIRS = [('"', '"'), ("«", "»"), ("“", "”"), ("„", "”"), ("‘", "’")]


def common_prefix(strings: list[str]) -> str:
    out = strings[0]
    for s in strings[1:]:
        i = 0
        while i < min(len(out), len(s)) and out[i] == s[i]:
            i += 1
        out = out[:i]
    return out


def extract_quoted(instruction: str) -> str | None:
    """The 'not answerable' literal: the quoted span inside the instruction.

    Returns None rather than guessing if the instruction is not exactly one
    quoted span -- a silent wrong answer here is worse than a visible gap.
    """
    hits = []
    for lq, rq in QUOTE_PAIRS:
        pat = re.escape(lq) + r"([^" + re.escape(lq + rq) + r"]+)" + re.escape(rq)
        hits += re.findall(pat, instruction)
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", type=Path, default=SCRIPT_DIR / "test/tests.jsonl")
    ap.add_argument("--task", default="qa-context")
    ap.add_argument("--out", type=Path,
                    default=SCRIPT_DIR / "qa_context_templates.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.tests, encoding="utf-8")]
    rows = [r for r in rows if r["task"] == args.task]
    if not rows:
        raise SystemExit(f"error: no rows with task={args.task!r} in {args.tests}")

    by_lang: dict[str, list[list[str]]] = collections.defaultdict(list)
    bad = 0
    for r in rows:
        parts = r["prompt"].split(SEP)
        if len(parts) != 4:
            bad += 1
            continue
        by_lang[r["question_lang"]].append(parts)
    if bad:
        print(f"warning: {bad} prompts did not split into 4 segments; skipped")

    templates, problems = {}, []
    for lang in sorted(by_lang):
        segs = by_lang[lang]
        intros = {s[0] for s in segs}
        instrs = {s[3] for s in segs}
        if len(intros) != 1:
            problems.append(f"{lang}: {len(intros)} distinct intros")
        if len(instrs) != 1:
            problems.append(f"{lang}: {len(instrs)} distinct instructions")
        instruction = sorted(instrs, key=len)[0]
        q_prefix = common_prefix([s[2] for s in segs])
        not_answerable = extract_quoted(instruction)
        if not_answerable is None:
            problems.append(f"{lang}: could not isolate the 'not answerable' literal")
        templates[lang] = {
            "question_lang": lang,
            "n_rows": len(segs),
            "intro": sorted(intros, key=len)[0],
            "q_prefix": q_prefix,
            "instruction": instruction,
            "not_answerable": not_answerable,
            "sep": SEP,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"task": args.task, "sep": SEP, "templates": templates},
                  f, ensure_ascii=False, indent=2)

    # Round-trip check: re-render every prompt from its template and compare.
    # Without this the template is a plausible-looking guess rather than a fact.
    rebuilt = mismatch = 0
    for r in rows:
        parts = r["prompt"].split(SEP)
        if len(parts) != 4:
            continue
        t = templates[r["question_lang"]]
        question = parts[2][len(t["q_prefix"]):]
        got = SEP.join([t["intro"], parts[1], t["q_prefix"] + question,
                        t["instruction"]])
        rebuilt += 1
        mismatch += got != r["prompt"]

    print(f"languages: {len(templates)}")
    print(f"round-trip: {rebuilt - mismatch}/{rebuilt} prompts rebuilt exactly"
          + (f"  ({mismatch} MISMATCH)" if mismatch else ""))
    for p in problems:
        print("  problem:", p)
    print(f"\n{'lang':6s} {'rows':>5s}  {'not_answerable':30s} q_prefix")
    print("-" * 88)
    for lang, t in templates.items():
        na = t["not_answerable"] or "<UNRESOLVED>"
        print(f"{lang:6s} {t['n_rows']:5d}  {na:30s} {t['q_prefix']!r}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
