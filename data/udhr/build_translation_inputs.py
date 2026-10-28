#!/usr/bin/env python3
"""Build the localisation jobs for questions and answers; passages need none, the official UDHR
text exists in every language. Each unit carries the official target-language passage so that a
span answer is located in it rather than translated; free translation is the fallback for non-span
answers. Unanswerable items only need their question localised.

Output: data/translation_inputs/<lang>.<half>.json
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Endonym-free English names, used only to address the translator. The
# authoritative identifier is the code; these exist so a prompt can say
# "Bhojpuri" instead of "bho".
LANG_NAMES = {
    "arb": "Modern Standard Arabic", "ben": "Bengali", "bho": "Bhojpuri",
    "ces": "Czech", "ckb": "Kurdish (Kurmanji, Latin script -- the variety the "
                            "organisers' 'ckb' passages are actually written in)",
    "deu": "German", "eng": "English", "fin": "Finnish", "fra": "French",
    "hat": "Haitian Creole", "hin": "Hindi", "ind": "Indonesian", "ita": "Italian",
    "jpn": "Japanese", "kor": "Korean", "mar": "Marathi", "pes": "Persian (Farsi)",
    "por": "Portuguese", "rus": "Russian", "slk": "Slovak", "spa": "Spanish",
    "tur": "Turkish", "vie": "Vietnamese", "yor": "Yoruba", "zho": "Chinese (Simplified)",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=SCRIPT_DIR / "data/items.json")
    ap.add_argument("--passages", type=Path, default=SCRIPT_DIR / "data/passages.json")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "qa_context_templates.json")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data/translation_inputs")
    ap.add_argument("--halves", type=int, default=2,
                    help="Split each language's units into this many job files, so one "
                         "agent's job stays small enough to do carefully.")
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))["items"]
    passages = {p["passage_id"]: p
                for p in json.loads(args.passages.read_text(encoding="utf-8"))["passages"]}
    templates = json.loads(args.templates.read_text(encoding="utf-8"))["templates"]

    # Question languages: the 24 the organisers ask questions in, minus English
    # (the source). fra is a context language only -- it never carries a
    # question -- so it needs no localisation.
    target_langs = sorted(set(templates) - {"eng"})
    print(f"target question languages: {len(target_langs)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*.json"):
        old.unlink()

    n_files = n_units = 0
    for lang in target_langs:
        units = []
        for it in items:
            p = passages[it["passage_id"]]
            units.append({
                "item_id": it["item_id"],
                "passage_id": it["passage_id"],
                "articles": it["articles"],
                "answerable": it["answerable"],
                "question_eng": it["question"],
                "answer_eng": it["answer"],
                "answer_is_span": it["answer_is_span"],
                "question_type": it["question_type"],
                "passage_eng": p["text"]["eng"],
                "passage_tgt": p["text"][lang],
                # To be filled by the localisation pass:
                "question_tgt": None,
                "answer_tgt": None,
                "answer_grounded": None,
                "uncertain": False,
                "note": "",
            })

        size = (len(units) + args.halves - 1) // args.halves
        for h in range(args.halves):
            chunk = units[h * size:(h + 1) * size]
            if not chunk:
                continue
            path = args.out_dir / f"{lang}.{h + 1}.json"
            path.write_text(json.dumps({
                "lang": lang,
                "lang_name": LANG_NAMES.get(lang, lang),
                "half": h + 1,
                "of": args.halves,
                "not_answerable_literal": templates[lang]["not_answerable"],
                "n_units": len(chunk),
                "units": chunk,
            }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            n_files += 1
            n_units += len(chunk)

    n_ans = sum(1 for i in items if i["answerable"])
    print(f"items: {len(items)} ({n_ans} answerable, {len(items) - n_ans} unanswerable)")
    print(f"per language: {len(items)} questions + {n_ans} answers = "
          f"{len(items) + n_ans} localisation decisions")
    print(f"wrote {n_files} job files ({n_units:,} units) to {args.out_dir}")
    print(f"\ntotal localisation decisions across all languages: "
          f"{(len(items) + n_ans) * len(target_langs):,}")
    print("\nNext: the localisation fan-out (see README.md), then merge_translations.py")


if __name__ == "__main__":
    main()
