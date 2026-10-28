"""Build the translation units for the silver references: 80 answerable items x 23 non-English
question languages. Each unit carries the English passage and, where the same source text exists in
the test set in the target language, that official passage (target_passage) so spans can be reused
verbatim. same_language marks units serving rows whose context language equals their question
language.

Output: translation_inputs/<lang>.json
"""

import argparse
import collections
import json
import pathlib

QUESTION_MARKER = "The question is:"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests-path", default="data/test/tests.jsonl")
    ap.add_argument("--items", default="data/silver/silver_items.json")
    ap.add_argument("--labels", default="data/silver/silver_labels.json")
    ap.add_argument("--output-dir", default="data/silver/translation_inputs")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.tests_path, encoding="utf-8")]
    qa = [r for r in rows if r["task"] == "qa-context"]
    items = json.load(open(args.items, encoding="utf-8"))
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]

    # passage text for every (integer id, context language), read off any row
    passages = {}
    for r in qa:
        _, n, _, clang = r["id"].split("_")
        passages.setdefault((int(n), clang), r["prompt"].split("\\n\\n")[1])

    # for an item, the passage in a given language (if that language is a context lang)
    def passage_in(item, lang):
        for rid in item["row_ids"]:
            _, n, _, clang = rid.split("_")
            if clang == lang:
                return passages[(int(n), clang)]
        return None

    qlangs = sorted({r["question_lang"] for r in qa} - {"eng"})
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = collections.Counter()
    for lang in qlangs:
        units = []
        for it in items:
            lab = labels[it["item_id"]]
            if not lab["answerable"]:
                continue  # abstention strings are exact, never translated
            target_passage = passage_in(it, lang)
            units.append({
                "item_id": it["item_id"],
                "english_question": it["english_question"],
                "english_answer": lab["answer"],
                "confidence": lab["confidence"],
                "english_passage": it["english_passage"],
                "target_passage": target_passage,
                "same_language": target_passage is not None,
                "translation": None,
            })
        counts[lang] = sum(1 for u in units if u["same_language"])
        (out_dir / f"{lang}.json").write_text(
            json.dumps({"target_lang": lang, "units": units},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(qlangs) * sum(1 for it in items if labels[it["item_id"]]["answerable"])
    print(f"wrote {len(qlangs)} language files ({total} units) -> {out_dir}")
    print(f"units with an authoritative target passage: {sum(counts.values())}")
    print("  by language:", dict(counts.most_common()))


if __name__ == "__main__":
    main()
