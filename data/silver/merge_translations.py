"""Fill each row's target from translation_outputs/<lang>.json and record its provenance in
target_source: exact (refusal string from the prompt), silver_english, silver_grounded (official
passage wording, same-language rows), silver_translated. Rows left without a target are reported.

Output: silver_test_labels_final.jsonl
"""

import argparse
import collections
import json
import pathlib


def span_coverage(text, passage, n=4):
    """Fraction of the reference's character n-grams that occur in the passage.

    Script-agnostic (whitespace tokenisation is useless for Chinese and Japanese) and
    tolerant of reordering, which matters because several references concatenate a list
    of separate spans -- their longest *contiguous* match is short even though nearly
    every word is lifted.
    """
    text = " ".join(text.strip(" .。۔؟!").split())
    passage = " ".join((passage or "").split())
    if not text or not passage:
        return 0.0
    if len(text) < n:
        return 1.0 if text in passage else 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return sum(1 for g in grams if g in passage) / len(grams)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/silver/silver_test_labels.jsonl")
    ap.add_argument("--translations", default="data/silver/translation_outputs")
    ap.add_argument("--items", default="data/silver/silver_items.json")
    ap.add_argument("--uncertainty", default="data/silver/translation_uncertainty.json")
    ap.add_argument("--output", default="data/silver/silver_test_labels_final.jsonl")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.labels, encoding="utf-8")]
    items = {it["item_id"]: it for it in json.load(open(args.items, encoding="utf-8"))}

    # (lang, item_id) -> {translation, grounded}
    table = {}
    GROUND_THRESHOLD = 0.8
    tdir = pathlib.Path(args.translations)
    found_langs = []
    for path in sorted(tdir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        lang = data["target_lang"]
        found_langs.append(lang)
        for u in data["units"]:
            if u.get("translation"):
                # Measure grounding rather than trusting the input flag: `same_language`
                # only says an official passage was *available*, not that the reference
                # actually reuses its wording. Some answers cannot be spans at all --
                # "Three." is a count, and the Czech/Portuguese/Slovak AI-Act text has no
                # phrase corresponding to "The European Union".
                coverage = (span_coverage(u["translation"], u["target_passage"])
                            if u.get("same_language") else 0.0)
                table[(lang, u["item_id"])] = {
                    "text": u["translation"],
                    "coverage": coverage,
                    "grounded": coverage >= GROUND_THRESHOLD,
                }
    print(f"loaded {len(found_langs)} language files, {len(table)} translations")

    # Per-language self-reported uncertainty from the translation pass, plus a
    # cross-language tally. An item flagged in many languages usually is not a
    # translation failure but an ill-posed reference: item_056 asks for "the
    # third-to-last listed activity" of a four-item list that the official Russian and
    # Finnish UDHR do not preserve, so the ordinal has no target-language answer at all.
    flagged = set()
    per_item = collections.Counter()
    unc_path = pathlib.Path(args.uncertainty)
    if unc_path.exists():
        unc = json.loads(unc_path.read_text(encoding="utf-8"))
        for lang, data in unc.items():
            for u in data.get("uncertain", []):
                flagged.add((lang, u["item_id"]))
                per_item[u["item_id"]] += 1
        print(f"loaded {len(flagged)} self-flagged (language, item) pairs")

    UNRELIABLE_AT = 8  # flagged in this many languages -> the reference itself is suspect
    unreliable = {i for i, n in per_item.items() if n >= UNRELIABLE_AT}
    if unreliable:
        print(f"references unreliable across languages: {sorted(unreliable)}")

    filled = collections.Counter()
    missing = collections.Counter()
    for r in recs:
        if r["item_id"] in unreliable:
            r["reference_unreliable"] = True
        if (r["question_lang"], r["item_id"]) in flagged:
            r["translation_uncertain"] = True
        if not r.pop("needs_translation", False):
            filled[r["target_source"]] += 1
            continue
        hit = table.get((r["question_lang"], r["item_id"]))
        if hit is None:
            missing[r["question_lang"]] += 1
            r["target"] = None
            r["target_source"] = "MISSING"
            continue
        r["target"] = hit["text"]
        # Grounded only where this row's own context language matches its answer
        # language -- that is the case where the model actually saw the passage it is
        # expected to quote. Elsewhere the official wording exists but the model never
        # read it, so the reference is a translation regardless of its provenance.
        same = r["question_lang"] == r["context_lang"]
        r["target_source"] = "silver_grounded" if (hit["grounded"] and same) else "silver_translated"
        if same:
            r["grounding_coverage"] = round(hit["coverage"], 2)
        r.pop("reference_english", None)
        filled[r["target_source"]] += 1

    with open(args.output, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(recs)} rows -> {args.output}")
    for source, n in filled.most_common():
        print(f"  {source:<20} {n:>5}")
    n_unc = sum(1 for r in recs if r.get("translation_uncertain"))
    n_unrel = sum(1 for r in recs if r.get("reference_unreliable"))
    print(f"\n  flagged: {n_unc} rows translation_uncertain, {n_unrel} reference_unreliable")
    if missing:
        print(f"  MISSING targets in {len(missing)} languages: {dict(missing)}")
    else:
        print("  every row has a reference")


if __name__ == "__main__":
    main()
