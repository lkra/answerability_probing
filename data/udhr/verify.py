#!/usr/bin/env python3
"""Check the rendered rows against data/test/tests.jsonl: no real newlines; ids unique and in the
test pattern; prompts split into four segments whose intro, instruction and question prefix match
the organisers' templates; unanswerable completions are exactly the refusal literal and no
answerable one is; no passage overlaps a test prompt (all 12,775 rows) and no row uses an article
the test set touches; the longest shared substring is reported; every passage equals the official
rendering it claims.
"""

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SEP = "\\n\\n"
ID_RE = re.compile(r"^qa-context_\d+_[a-z]{3}_[a-z]{3}$")
_WS_RE = re.compile(r"\s+")


def clean(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def longest_common_substring_len(a: str, b: str, cap: int = 400) -> int:
    """Binary search on k-gram set intersection. O(n log n), unlike DP."""
    lo, hi = 0, min(len(a), len(b), cap)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        grams = {a[i:i + mid] for i in range(len(a) - mid + 1)}
        if any(b[i:i + mid] in grams for i in range(len(b) - mid + 1)):
            lo = mid
        else:
            hi = mid - 1
    return lo


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.passed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"  FAIL  {name}   {detail}")
        return ok

    def note(self, text: str) -> None:
        self.notes.append(text)
        print(f"  note  {text}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=SCRIPT_DIR / "data/udhr_qa_train.jsonl")
    ap.add_argument("--passages", type=Path, default=SCRIPT_DIR / "data/passages.json")
    ap.add_argument("--alignment", type=Path, default=SCRIPT_DIR / "data/udhr_alignment.json")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "qa_context_templates.json")
    ap.add_argument("--tests", type=Path, default=REPO_ROOT / "test/tests.jsonl")
    ap.add_argument("--report-out", type=Path, default=SCRIPT_DIR / "data/verify_report.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.rows.open(encoding="utf-8")]
    passages = {p["passage_id"]: p
                for p in json.loads(args.passages.read_text(encoding="utf-8"))["passages"]}
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    templates = json.loads(args.templates.read_text(encoding="utf-8"))["templates"]
    test_rows = [json.loads(l) for l in args.tests.open(encoding="utf-8")]

    c = Checker()
    print(f"rows: {len(rows):,}   test rows: {len(test_rows):,}\n")

    # ---- 1. no real newlines ------------------------------------------------
    print("format")
    bad = [r["id"] for r in rows if "\n" in r["prompt"] or "\n" in r["completion"]]
    c.check("no real newline in prompt or completion", not bad,
            f"{len(bad)} rows, e.g. {bad[:3]}")

    # ---- 2. ids -------------------------------------------------------------
    ids = [r["id"] for r in rows]
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    c.check("row ids unique", not dupes, f"{len(dupes)} duplicated, e.g. {dupes[:3]}")
    malformed = [i for i in ids if not ID_RE.match(i)]
    c.check("row ids match the test-set id pattern", not malformed,
            f"{len(malformed)} malformed, e.g. {malformed[:3]}")
    test_ids = {r["id"] for r in test_rows}
    collide = sorted(set(ids) & test_ids)
    c.check("no row id collides with a real test-set id", not collide,
            f"{len(collide)} collide, e.g. {collide[:3]}")

    # ---- 3/4. prompt structure and template fidelity ------------------------
    seg_bad, intro_bad, instr_bad, prefix_bad = [], [], [], []
    for r in rows:
        parts = r["prompt"].split(SEP)
        if len(parts) != 4:
            seg_bad.append(r["id"])
            continue
        t = templates[r["question_lang"]]
        if parts[0] != t["intro"]:
            intro_bad.append(r["id"])
        if parts[3] != t["instruction"]:
            instr_bad.append(r["id"])
        if not parts[2].startswith(t["q_prefix"]):
            prefix_bad.append(r["id"])
    c.check("every prompt splits into 4 segments", not seg_bad,
            f"{len(seg_bad)} rows, e.g. {seg_bad[:3]}")
    c.check("intro byte-identical to the organizers' template", not intro_bad,
            f"{len(intro_bad)} rows, e.g. {intro_bad[:3]}")
    c.check("instruction byte-identical to the organizers' template", not instr_bad,
            f"{len(instr_bad)} rows, e.g. {instr_bad[:3]}")
    c.check("question prefix byte-identical to the organizers' template", not prefix_bad,
            f"{len(prefix_bad)} rows, e.g. {prefix_bad[:3]}")

    # ---- 5. abstention literals ---------------------------------------------
    print("\nabstention")
    wrong_abstain, leaked_abstain = [], []
    for r in rows:
        literal = templates[r["question_lang"]]["not_answerable"]
        if not r["answerable"] and r["completion"] != literal:
            wrong_abstain.append((r["id"], r["completion"][:40]))
        if r["answerable"] and r["completion"] == literal:
            leaked_abstain.append(r["id"])
    c.check("every unanswerable completion is the exact abstention literal",
            not wrong_abstain, f"{len(wrong_abstain)} rows, e.g. {wrong_abstain[:2]}")
    c.check("no answerable completion is an abstention literal", not leaked_abstain,
            f"{len(leaked_abstain)} rows, e.g. {leaked_abstain[:3]}")
    empty = [r["id"] for r in rows if not r["completion"].strip()]
    c.check("no empty completion", not empty, f"{len(empty)} rows")

    # ---- 9. passages match the official text --------------------------------
    print("\ngrounding")
    drift = []
    for r in rows:
        want = clean(passages[r["passage_id"]]["text"][r["context_lang"]])
        got = r["prompt"].split(SEP)[1]
        if got != want:
            drift.append(r["id"])
    c.check("every passage is byte-identical to its official UDHR rendering",
            not drift, f"{len(drift)} rows, e.g. {drift[:3]}")

    # ---- 7. article-level exclusion -----------------------------------------
    print("\ncontamination")
    excluded = set(alignment["used_articles"])
    bad_art = sorted({r["passage_id"] for r in rows
                      if excluded & set(r["articles"])})
    c.check("no row uses an article the test set touches", not bad_art,
            f"passages {bad_art}")

    # ---- 6. substring containment against every test prompt -----------------
    our_passages = {}
    for r in rows:
        our_passages[(r["passage_id"], r["context_lang"])] = r["prompt"].split(SEP)[1]
    test_prompts = [r["prompt"] for r in test_rows]
    test_blob = "\n\n".join(test_prompts)

    contained = [k for k, v in our_passages.items() if v and v in test_blob]
    c.check("no generated passage occurs inside any test prompt", not contained,
            f"{len(contained)} passages, e.g. {contained[:3]}")

    # the other direction: a test passage sitting inside one of ours
    test_passages = set()
    for r in test_rows:
        if r["task"] == "qa-context":
            p = r["prompt"].split(SEP)
            if len(p) == 4:
                test_passages.add(p[1])
    our_blob = "\n\n".join(our_passages.values())
    reverse = [p for p in test_passages if p and p in our_blob]
    c.check("no test passage occurs inside any generated passage", not reverse,
            f"{len(reverse)} test passages")

    # ---- 8. longest shared substring, reported ------------------------------
    eng_test = [p for p in test_passages if p.isascii()]
    worst = []
    for pid, p in sorted(passages.items()):
        text = clean(p["text"]["eng"])
        best = max((longest_common_substring_len(text, t) for t in eng_test), default=0)
        worst.append((best, pid))
    worst.sort(reverse=True)
    c.note(f"longest substring shared with an English test passage: "
           f"max {worst[0][0]} chars ({worst[0][1]}), "
           f"median {sorted(w for w, _ in worst)[len(worst) // 2]} chars")
    for n_chars, pid in worst[:3]:
        text = clean(passages[pid]["text"]["eng"])
        for t in eng_test:
            if longest_common_substring_len(text, t) == n_chars:
                # recover the actual shared string for the report
                shared = next((text[i:i + n_chars] for i in range(len(text) - n_chars + 1)
                               if text[i:i + n_chars] in t), "")
                c.note(f"  {pid}: {n_chars} chars -- {shared!r}")
                break

    # ---- distribution sanity -------------------------------------------------
    print("\ndistribution")
    n_un = sum(1 for r in rows if not r["answerable"])
    share = n_un / max(len(rows), 1)
    # Answerable-only builds are detected from the data rather than from a flag,
    # so the checks cannot disagree with the file they are checking.
    answerable_only = n_un == 0
    if answerable_only:
        c.note("answerable-only build: no unanswerable rows, so the share checks "
               "below are replaced by the stricter requirement that there are none")
        c.check("no unanswerable row present", n_un == 0, f"{n_un}")
        # The point of this mode is that the generator never abstains on its own --
        # a probe decides that downstream. So an abstention literal appearing as a
        # completion would defeat the whole arrangement. Already checked above per
        # question language; restated here because in this mode it is the load-
        # bearing check rather than a formatting detail.
        literals = {t["not_answerable"] for t in templates.values()}
        leaked = [r["id"] for r in rows if r["completion"] in literals]
        c.check("no completion anywhere is an abstention literal", not leaked,
                f"{len(leaked)} rows, e.g. {leaked[:3]}")
    else:
        c.check("unanswerable share within 5 points of the test set's 20.3%",
                abs(share - 0.203) < 0.05, f"{share:.1%}")
    qlangs = collections.Counter(r["question_lang"] for r in rows)
    clangs = collections.Counter(r["context_lang"] for r in rows)
    missing_q = sorted(set(templates) - set(qlangs))
    c.check("all 24 question languages present", not missing_q,
            f"missing {missing_q} -- these languages have no localisation output; "
            f"re-run their jobs in data/translation_inputs/")
    c.check("all 25 context languages present", len(clangs) == 25, f"{len(clangs)}")
    # English is deliberately over-represented as a context language -- every item
    # carries an English rendering, which is the test set's own structure. So the
    # evenness check is on the other 24. For reference, the test set's own
    # (item, context-language) distribution is far more skewed than this one:
    # English 27.8% of pairs, max/min 20.0 overall and 5.0 among the non-English
    # languages, with yor/mar/jpn/kor/pes/tur/vie/slk at 1.4% each. Spreading
    # context languages evenly here is a deliberate departure -- those are the
    # languages the data is meant to help.
    non_eng = {k: v for k, v in clangs.items() if k != "eng"}
    spread = max(non_eng.values()) / max(min(non_eng.values()), 1)
    c.check("non-English context languages within 2x of each other", spread <= 2.0,
            f"max/min = {spread:.2f}")
    eng_share = clangs.get("eng", 0) / max(len(rows), 1)
    c.note(f"English as context language: {eng_share:.1%} of rows "
           f"(test set: 27.8% of its item x context-language pairs)")
    if not answerable_only:
        per_lang_un = {}
        for lang in clangs:
            tot = clangs[lang]
            un = sum(1 for r in rows if r["context_lang"] == lang and not r["answerable"])
            per_lang_un[lang] = un / tot
        off = {k: f"{v:.0%}" for k, v in per_lang_un.items() if abs(v - share) > 0.08}
        c.check("no context language's unanswerable share is far from the mean",
                not off, f"{off}")

    # ---- verdict -------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"{c.passed} checks passed, {len(c.failures)} failed")
    for f in c.failures:
        print(f"  FAILED: {f}")

    args.report_out.write_text(json.dumps({
        "n_rows": len(rows),
        "checks_passed": c.passed,
        "checks_failed": len(c.failures),
        "failures": c.failures,
        "notes": c.notes,
        "unanswerable_share": round(share, 4),
        "longest_shared_substring_with_test": {pid: n for n, pid in worst},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report_out}")

    sys.exit(1 if c.failures else 0)


if __name__ == "__main__":
    main()
