#!/usr/bin/env python3
"""Join generate_answers.py output with per-row probe scores on id and report: how often the model
answers rows the probe scores as unanswerable, by probe-confidence bin; whether abstentions use the
exact per-language literal (matched after normalising case, quotes and trailing punctuation;
unlisted paraphrases count as answers, so abstention rates are lower bounds; --dump-unmatched to
inspect); and, cross-lingually, whether the output is in the question's language, the passage's, or
neither.
"""

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Atomic character blocks. Han is deliberately its own bucket rather than being
# folded into either CJK language: kanji and hanzi share the block, so a
# kanji-heavy Japanese sentence is indistinguishable from Chinese character by
# character. Treating Han as "Chinese" scored 38.8% of real Japanese answers as
# wrong-script; the fix is to ask whether each character is *compatible* with the
# target script, not which single script wins a majority vote.
CHAR_BLOCKS = {
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Beng": [(0x0980, 0x09FF)],
    "Cyrl": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Deva": [(0x0900, 0x097F)],
    "Hang": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "Kana": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
    "Han":  [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "Telu": [(0x0C00, 0x0C7F)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Latn": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}

# Which blocks legitimately appear when writing each FLORES script.
COMPATIBLE = {
    "Arab": {"Arab"}, "Beng": {"Beng"}, "Cyrl": {"Cyrl"}, "Deva": {"Deva"},
    "Telu": {"Telu"}, "Thai": {"Thai"}, "Latn": {"Latn"},
    "Jpan": {"Kana", "Han"},        # kana and kanji both, in any proportion
    "Hans": {"Han"},
    "Hang": {"Hang", "Han"},        # hanja still turn up in Korean text
}

_TRAILING = " \t\r\n.。!！?？؟۔।,、:;\"'«»“”„‘’()[]{}<>*_-–—"

# Renderings that mean the right thing but are not the string the organizers
# score. Curated, not heuristic: each one was seen in `--dump-unmatched` output
# and added deliberately, because a fuzzy matcher across 24 languages would
# quietly reclassify real answers as abstentions.
#
# NFKC does not fold traditional to simplified Chinese, so a model answering in
# traditional produces a correct abstention that scores zero. That is a genuine
# finding, not a matching artefact, so it gets its own category rather than
# being folded into `abstain_exact`.
KNOWN_VARIANTS = {
    "zho": ["無法回答"],          # traditional form of 无法回答
}

# Free-form refusals -- "the passage does not say" rather than the literal the
# organizers score. Used only to *estimate how much* exact matching undercounts
# abstention, never to reclassify: it is English-only and would misfire on a
# genuine answer that happens to note a gap ("the passage does not give the area,
# but the region is 30,370 km2"). English-only means it is a lower bound on the
# undercount, which is the safe direction for a caveat.
_PARAPHRASE_RE = re.compile(
    r"passage (does not|doesn't|did not)|not (provide|mention|state|specify|contain)"
    r"|no information|cannot be answered|can't be answered|unable to answer"
    r"|does not say", re.I)


def normalise(text: str) -> str:
    """Fold away everything the organizers would not score on."""
    text = unicodedata.normalize("NFKC", text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(_TRAILING).strip()
    return text.casefold()


def script_of(text: str) -> collections.Counter:
    """Count alphabetic characters per atomic block; ignores digits and punctuation."""
    counts = collections.Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for tag, ranges in CHAR_BLOCKS.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[tag] += 1
                break
    return counts


def wrong_script(text: str, target_script: str) -> bool | None:
    """Is most of `text` written in a block incompatible with `target_script`?

    None when there is nothing to judge (no alphabetic characters -- a bare date
    or number is a legitimate answer in any script).
    """
    counts = script_of(text)
    total = sum(counts.values())
    if not total:
        return None
    ok = sum(n for tag, n in counts.items() if tag in COMPATIBLE.get(target_script, set()))
    return ok / total < 0.5


def script_share(text: str, script: str) -> float | None:
    """Fraction of alphabetic characters compatible with `script`. None if none to judge."""
    counts = script_of(text)
    total = sum(counts.values())
    if not total:
        return None
    return sum(n for tag, n in counts.items()
               if tag in COMPATIBLE.get(script, set())) / total


def output_side(text: str, question_script: str, passage_script: str) -> str | None:
    """Which language the output is in -> "question" | "passage" | "either" | "neither" | None.
    "either": the two languages share a script and the check cannot separate them; exclude it from
    rates.
    """
    q = script_share(text, question_script)
    if q is None:
        return None                      # a bare number or date is valid in any script
    p = script_share(text, passage_script)
    if question_script == passage_script or (q >= 0.5 and p >= 0.5):
        return "either"
    if q >= 0.5:
        return "question"
    if p >= 0.5:
        return "passage"
    return "neither"


def classify(generation: str, question_lang: str, literals: dict[str, str],
             variants: dict[str, list[str]] | None = None):
    """-> (category, matched_language or None)

    Order matters: an exact match in the right language must win over a
    containment match in another, or a language whose literal is a substring of
    another's would be misattributed.
    """
    variants = variants or {}
    g = normalise(generation)
    if not g:
        return "empty", None

    target = normalise(literals[question_lang])
    if g == target:
        return "abstain_exact", question_lang

    others = {l: normalise(v) for l, v in literals.items() if l != question_lang}
    for lang, lit in others.items():
        if lit and g == lit:
            return "abstain_wrong_language", lang

    for v in variants.get(question_lang, []):
        if g == normalise(v):
            return "abstain_variant", question_lang

    if target and target in g:
        return "abstain_extra_text", question_lang
    for lang, lit in others.items():
        if lit and lit in g:
            return "abstain_wrong_language_extra", lang
    for v in variants.get(question_lang, []):
        nv = normalise(v)
        if nv and nv in g:
            return "abstain_variant", question_lang

    return "answer", None


ABSTAIN = {"abstain_exact", "abstain_extra_text", "abstain_variant",
           "abstain_wrong_language", "abstain_wrong_language_extra"}
SCORED = {"abstain_exact"}          # what the organizers would actually credit


def pct(n, d):
    return f"{n / d:6.1%}" if d else "     -"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=Path,
                    default=REPO_ROOT / "probe/generation/validation_qa_context.jsonl")
    ap.add_argument("--probe", type=Path,
                    default=REPO_ROOT / "probe/generation/probe_predictions.jsonl")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "data/qa_context_templates.json")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="prob_answerable below this = the probe says unanswerable.")
    ap.add_argument("--dump-unmatched", type=int, default=0,
                    help="Print this many generations classified `answer` on unanswerable rows, "
                         "for hand-checking how many are unlisted abstention paraphrases.")
    ap.add_argument("--variants", type=Path, default=None,
                    help="JSON {lang: [extra abstention strings]} merged into the built-in "
                         "KNOWN_VARIANTS. Populate it from --dump-unmatched output.")
    ap.add_argument("--out", type=Path, default=None, help="Write the full report as JSON.")
    args = ap.parse_args()

    spec = json.loads(args.templates.read_text(encoding="utf-8"))
    literals = {k: v["not_answerable"] for k, v in spec["templates"].items()
                if v.get("not_answerable")}

    variants = {k: list(v) for k, v in KNOWN_VARIANTS.items()}
    if args.variants:
        for k, v in json.loads(args.variants.read_text(encoding="utf-8")).items():
            variants.setdefault(k, []).extend(v)

    gens = [json.loads(l) for l in open(args.generations, encoding="utf-8")]
    probe = {json.loads(l)["id"]: json.loads(l)
             for l in open(args.probe, encoding="utf-8")}

    rows, unjoined = [], 0
    for g in gens:
        p = probe.get(g["id"])
        if p is None:
            unjoined += 1
            continue
        cat, matched = classify(g["generation"], g["question_lang"], literals, variants)
        rows.append({**g, "prob_answerable": p["prob_answerable"],
                     "category": cat, "matched_lang": matched})
    print(f"{len(gens):,} generations, {len(rows):,} joined to a probe score"
          + (f", {unjoined:,} dropped (no probe score -- likely excluded as contaminated)"
             if unjoined else ""))
    if not rows:
        raise SystemExit("error: nothing joined; do the two files cover the same rows?")

    unans = [r for r in rows if r["answerable"] == 0]
    ans = [r for r in rows if r["answerable"] == 1]

    # ---- 1. does the model answer what the probe knows is unanswerable? ----
    print(f"\n{'=' * 78}\n1. THE MODEL ANSWERS WHAT THE PROBE DETECTS\n{'=' * 78}")
    print(f"\nTruly unanswerable rows: {len(unans):,}   answerable: {len(ans):,}")

    detected = [r for r in unans if r["prob_answerable"] < args.threshold]
    missed = [r for r in unans if r["prob_answerable"] >= args.threshold]
    answered_anyway = [r for r in detected if r["category"] not in ABSTAIN]
    print(f"\nOf the {len(unans):,} unanswerable rows, the probe detects "
          f"{len(detected):,} ({pct(len(detected), len(unans))}).")
    print(f"On those {len(detected):,}, the model still produced an answer "
          f"{len(answered_anyway):,} times = {pct(len(answered_anyway), len(detected))}")
    print("  ^ the headline: the signal is present and linearly readable, but the "
          "model does not act on it.")

    print(f"\n{'lang':10s} {'unans':>6s} {'probe finds':>11s} {'model abstains':>14s} "
          f"{'answers despite probe':>21s}")
    by = collections.defaultdict(list)
    for r in unans:
        by[r["lang_code"]].append(r)
    for lang in sorted(by):
        rs = by[lang]
        det = [r for r in rs if r["prob_answerable"] < args.threshold]
        abst = [r for r in rs if r["category"] in ABSTAIN]
        bad = [r for r in det if r["category"] not in ABSTAIN]
        print(f"{lang:10s} {len(rs):6,d} {pct(len(det), len(rs)):>11s} "
              f"{pct(len(abst), len(rs)):>14s} {pct(len(bad), len(det)):>21s}")

    # Stratifying by probe confidence separates two very different stories:
    # "the model ignores a signal it has" versus "probe and model disagree only
    # where the probe is unsure anyway".
    print(f"\nBy probe confidence (unanswerable rows only):")
    print(f"  {'p(answerable)':16s} {'n':>7s} {'model answered':>15s}")
    for lo, hi in [(0, .01), (.01, .05), (.05, .1), (.1, .25), (.25, .5)]:
        s = [r for r in unans if lo <= r["prob_answerable"] < hi]
        if not s:
            continue
        bad = sum(r["category"] not in ABSTAIN for r in s)
        print(f"  [{lo:.2f},{hi:.2f})     {len(s):7,d} {pct(bad, len(s)):>15s}")

    # False abstention on answerable rows is the mirror-image error.
    false_abst = [r for r in ans if r["category"] in ABSTAIN]
    print(f"\nMirror error -- abstained on an ANSWERABLE row: {len(false_abst):,} "
          f"of {len(ans):,} = {pct(len(false_abst), len(ans))}")

    # How badly does exact matching undercount abstention? Estimated by looking
    # for free-form refusals among the rows called `answer`. English-only by
    # construction, so it bounds the undercount rather than measuring it in full.
    residual = [r for r in unans
                if r["category"] == "answer" and _PARAPHRASE_RE.search(r["generation"])]
    print(f"\nUnlisted abstention paraphrases still counted as `answer`: "
          f"{len(residual):,} of {len(unans):,} unanswerable rows "
          f"({pct(len(residual), len(unans))}) -- the amount by which every "
          f"abstention rate here is understated.")

    # ---- 2. when it abstains, does it abstain correctly? ----
    print(f"\n{'=' * 78}\n2. INSTRUCTION-FOLLOWING WHEN ABSTAINING\n{'=' * 78}")
    abstained = [r for r in rows if r["category"] in ABSTAIN]
    print(f"\n{len(abstained):,} abstentions detected across all rows.")
    if abstained:
        cats = collections.Counter(r["category"] for r in abstained)
        print(f"\n{'category':30s} {'n':>7s} {'share':>8s}   scored?")
        for cat, n in cats.most_common():
            print(f"  {cat:28s} {n:7,d} {pct(n, len(abstained)):>8s}   "
                  f"{'yes' if cat in SCORED else 'NO'}")
        wrong = [r for r in abstained if r["category"] not in SCORED]
        print(f"\n{len(wrong):,} of {len(abstained):,} abstentions "
              f"({pct(len(wrong), len(abstained))}) would score zero despite being "
              f"the right decision.")

        leak = collections.Counter(r["matched_lang"] for r in abstained
                                   if r["category"].startswith("abstain_wrong_language"))
        if leak:
            print(f"\nWrong-language abstentions, by the language actually produced:")
            for lang, n in leak.most_common(10):
                print(f"  {lang:10s} {n:6,d}")

        print(f"\n{'lang':10s} {'abstentions':>11s} {'exact':>8s} {'wrong-lang':>11s} "
              f"{'extra text':>11s}")
        byl = collections.defaultdict(list)
        for r in abstained:
            byl[r["question_lang"]].append(r)
        for lang in sorted(byl):
            rs = byl[lang]
            ex = sum(r["category"] == "abstain_exact" for r in rs)
            wl = sum(r["category"].startswith("abstain_wrong_language") for r in rs)
            et = sum(r["category"] == "abstain_extra_text" for r in rs)
            print(f"{lang:10s} {len(rs):11,d} {pct(ex, len(rs)):>8s} "
                  f"{pct(wl, len(rs)):>11s} {pct(et, len(rs)):>11s}")

    # ---- 3. answering in the wrong script ----
    print(f"\n{'=' * 78}\n3. ANSWERS IN THE WRONG SCRIPT (non-Latin targets only)\n{'=' * 78}")
    print("Latin-script targets are omitted: the check cannot tell eng from fin from ind.")
    attempts = [r for r in rows if r["category"] == "answer"]
    print(f"\n{'lang':10s} {'answers':>8s} {'judged':>7s} {'wrong script':>13s}")
    bys = collections.defaultdict(list)
    for r in attempts:
        bys[r["lang_code"]].append(r)
    for lang in sorted(bys):
        script = lang.split("_")[1]
        if script == "Latn":
            continue
        verdicts = [wrong_script(r["generation"], script) for r in bys[lang]]
        judged = [v for v in verdicts if v is not None]
        print(f"{lang:10s} {len(verdicts):8,d} {len(judged):7,d} "
              f"{pct(sum(judged), len(judged)):>13s}")

    # ---- 4. cross-lingual: question's language or the passage's? ----
    # Monolingually this section is empty by construction, and that is the point:
    # zero wrong-language abstentions out of 4,068 says nothing about the case where
    # the passage is pulling the other way. Here the two candidates are separated.
    xling = [r for r in rows
             if r.get("passage_lang") and r["passage_lang"] != r.get("question_lang_flores",
                                                                     r["lang_code"])]
    print(f"\n{'=' * 78}\n4. CROSS-LINGUAL: WHICH LANGUAGE DID IT ANSWER IN?\n{'=' * 78}")
    if not xling:
        print("\nNo cross-lingual rows in this file (passage and question language always "
              "agree), so there is nothing to separate. Run with --source translation or "
              "--source belebele generations to populate this section.")
    else:
        print(f"\n{len(xling):,} of {len(rows):,} rows have passage != question language.")
        print("`either` = the two languages share a script and the check cannot separate them; "
              "it is excluded from the rates below rather than counted as a success.")

        def side_of(r):
            qs = r.get("question_lang_flores", r["lang_code"]).split("_")[1]
            return output_side(r["generation"], qs, r["passage_lang"].split("_")[1])

        # Split by what the model was *trying* to do: an abstention in the passage's
        # language is a different failure from an answer in the passage's language --
        # the first got the answerability decision right and the output language wrong.
        for label, subset in (("ANSWER ATTEMPTS", [r for r in xling if r["category"] == "answer"]),
                              ("ABSTENTIONS", [r for r in xling if r["category"] in ABSTAIN])):
            sides = collections.Counter()
            for r in subset:
                sides[side_of(r)] += 1
            judged = sum(n for s, n in sides.items() if s in ("question", "passage", "neither"))
            print(f"\n{label}: {len(subset):,} rows, {judged:,} separable")
            for s in ("question", "passage", "neither", "either", None):
                if sides.get(s):
                    tag = "unjudgeable (no letters)" if s is None else s
                    share = pct(sides[s], judged) if s in ("question", "passage", "neither") else "     -"
                    print(f"  answered in the {tag:26s} {sides[s]:6,d} {share}")
            if judged and sides.get("passage"):
                print(f"  -> {pct(sides['passage'], judged)} of separable {label.lower()} "
                      f"followed the PASSAGE's language against the instruction.")

        # Per question language, restricted to answer attempts: the hypothesis is that
        # the pull toward the passage is not uniform across languages.
        print(f"\n{'question':10s} {'separable':>9s} {'in question lang':>17s} "
              f"{'in passage lang':>16s} {'neither':>9s}")
        byq = collections.defaultdict(list)
        for r in xling:
            if r["category"] == "answer":
                byq[r.get("question_lang_flores", r["lang_code"])].append(r)
        for lang in sorted(byq):
            sides = collections.Counter(side_of(r) for r in byq[lang])
            judged = sum(sides[s] for s in ("question", "passage", "neither"))
            if not judged:
                continue
            print(f"{lang:10s} {judged:9,d} {pct(sides['question'], judged):>17s} "
                  f"{pct(sides['passage'], judged):>16s} {pct(sides['neither'], judged):>9s}")

    if args.dump_unmatched:
        print(f"\n{'=' * 78}\nSAMPLE OF `answer` ON UNANSWERABLE ROWS "
              f"(check for unlisted abstention paraphrases)\n{'=' * 78}")
        pool = [r for r in unans if r["category"] == "answer"]
        for r in pool[:args.dump_unmatched]:
            print(f"\n[{r['lang_code']}] p(answerable)={r['prob_answerable']:.3f}")
            print(f"  expected literal: {literals[r['question_lang']]!r}")
            print(f"  generated:        {r['generation'][:200]!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "n_joined": len(rows), "n_unjoined": unjoined,
            "threshold": args.threshold,
            "rows": [{k: r[k] for k in ("id", "lang_code", "answerable",
                                        "prob_answerable", "category", "matched_lang")}
                     for r in rows],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
