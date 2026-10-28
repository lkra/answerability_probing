"""Expand the 100 item labels to the 8,640 test rows. Answerability is copied from the item to every
rendering. Unanswerable rows get the refusal string quoted in their own prompt; answerable
non-English rows are emitted with needs_translation=true and a null target.

Output: silver_test_labels.jsonl
"""

import argparse
import collections
import json
import pathlib
import re

QUOTE_PAIRS = r'["“«「‘]([^"”»」’]{2,40})["”»」’]'


def extract_abstention_strings(rows):
    """Recover each language's exact abstention string from the prompts themselves.

    Every qa-context prompt ends with an instruction quoting the localised equivalent
    of "not answerable". Within a language the string is identical across all 360
    prompts, so the majority (in practice, unanimous) value is exact, not approximate.
    """
    per_lang = collections.defaultdict(list)
    for r in rows:
        per_lang[r["question_lang"]].append(set(re.findall(QUOTE_PAIRS, r["prompt"][-400:])))

    strings = {}
    for lang, prompt_quotes in per_lang.items():
        # Some prompts quote a second string (a term lifted from the question), so take
        # the strings common to every prompt in the language rather than the most frequent.
        common = set.intersection(*prompt_quotes)
        if len(common) != 1:
            raise ValueError(
                f"{lang}: expected exactly one string common to all {len(prompt_quotes)} "
                f"prompts, got {sorted(common)}"
            )
        strings[lang] = common.pop()
    return strings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests-path", default="data/test/tests.jsonl")
    ap.add_argument("--items", default="data/silver/silver_items.json")
    ap.add_argument("--labels", default="data/silver/silver_labels.json")
    ap.add_argument("--output", default="data/silver/silver_test_labels.jsonl")
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default="low",
                    help="drop rows whose item label is below this confidence")
    args = ap.parse_args()

    rank = {"low": 0, "medium": 1, "high": 2}
    rows = [json.loads(line) for line in open(args.tests_path, encoding="utf-8")]
    qa = [r for r in rows if r["task"] == "qa-context"]
    items = json.load(open(args.items, encoding="utf-8"))
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]

    missing = [it["item_id"] for it in items if it["item_id"] not in labels]
    if missing:
        raise SystemExit(f"no label for {len(missing)} items: {missing[:5]}")

    abstention = extract_abstention_strings(qa)
    print(f"recovered abstention strings for {len(abstention)} languages")

    row_to_item = {rid: it["item_id"] for it in items for rid in it["row_ids"]}
    by_id = {r["id"]: r for r in qa}

    out, skipped = [], 0
    for it in items:
        lab = labels[it["item_id"]]
        if rank[lab["confidence"]] < rank[args.min_confidence]:
            skipped += len(it["row_ids"])
            continue
        for rid in it["row_ids"]:
            qlang = by_id[rid]["question_lang"]
            rec = {
                "id": rid,
                "item_id": it["item_id"],
                "question_lang": qlang,
                "context_lang": rid.split("_")[3],
                "answerable": lab["answerable"],
                "confidence": lab["confidence"],
            }
            if not lab["answerable"]:
                # Exact, extracted from the prompt -- not a translation.
                rec["target"] = abstention[qlang]
                rec["target_source"] = "exact"
                rec["needs_translation"] = False
            elif qlang == "eng":
                rec["target"] = lab["answer"]
                rec["target_source"] = "silver_english"
                rec["needs_translation"] = False
            else:
                rec["target"] = None
                rec["target_source"] = "silver_english_untranslated"
                rec["reference_english"] = lab["answer"]
                rec["needs_translation"] = True
            out.append(rec)

    out.sort(key=lambda r: r["id"])
    with open(args.output, "w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_unans = sum(1 for r in out if not r["answerable"])
    n_trans = sum(1 for r in out if r["needs_translation"])
    items_unans = sum(1 for it in items if not labels[it["item_id"]]["answerable"])
    conf = collections.Counter(labels[it["item_id"]]["confidence"] for it in items)

    print(f"wrote {len(out)} rows -> {args.output}" + (f" ({skipped} skipped)" if skipped else ""))
    print(f"  items:  {items_unans}/{len(items)} unanswerable  |  confidence: {dict(conf)}")
    print(f"  rows:   {n_unans} unanswerable ({n_unans / len(out):.1%}), "
          f"{len(out) - n_unans} answerable")
    print(f"  targets: {len(out) - n_trans} ready ({n_unans} exact), {n_trans} need translation")


if __name__ == "__main__":
    main()
