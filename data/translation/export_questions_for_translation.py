#!/usr/bin/env python3
"""Export the unique TyDi validation questions for translation (each question appears twice in the
source, once answerable and once not). Passages and answers are not exported. Columns: qid,
src_lang (FLORES), iso639_1, iso639_3, question.

Output: <out>.jsonl and <out>.tsv
"""

import argparse
import collections
import csv
import json
from pathlib import Path

# The 28 WMT26 MIST languages -- the target set. All 28 are NLLB-200 language
# tokens, and the codes below are exactly the strings NLLB expects.
#
# bho_Deva (Bhojpuri) is the one test-only language. It is absent from Belebele
# and from TyDi, so translation is the *only* route that reaches it -- which
# makes it the target it would be most costly to omit, despite never appearing
# as a source language here.
MIST_LANGUAGES = [
    "arb_Arab", "ben_Beng", "bho_Deva", "ces_Latn", "ckb_Arab", "deu_Latn",
    "eng_Latn", "fin_Latn", "fra_Latn", "hat_Latn", "hin_Deva", "ind_Latn",
    "ita_Latn", "jpn_Jpan", "kor_Hang", "mar_Deva", "pes_Arab", "por_Latn",
    "rus_Cyrl", "slk_Latn", "spa_Latn", "swh_Latn", "tel_Telu", "tha_Thai",
    "tur_Latn", "vie_Latn", "yor_Latn", "zho_Hans",
]

# FLORES-200 code -> (ISO 639-1, ISO 639-3)
ISO = {
    "arb_Arab": ("ar", "ara"), "ben_Beng": ("bn", "ben"),
    "eng_Latn": ("en", "eng"), "fin_Latn": ("fi", "fin"),
    "ind_Latn": ("id", "ind"), "jpn_Jpan": ("ja", "jpn"),
    "kor_Hang": ("ko", "kor"), "rus_Cyrl": ("ru", "rus"),
    "swh_Latn": ("sw", "swh"), "tel_Telu": ("te", "tel"),
    "tha_Thai": ("th", "tha"),
}


def load_language_names(path: Path) -> dict[str, str]:
    """code -> plain-English name, from the project language table."""
    if not path.exists():
        raise SystemExit(
            f"error: {path} not found. Pass --languages-csv, or the "
            "human-readable language columns cannot be written."
        )
    names = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            name = (row.get("Language") or "").strip()
            if code and name:
                names[code] = name
    return names


def label(code: str, names: dict[str, str]) -> str:
    """'Standard Arabic (arb_Arab)'. Falls back to the bare code so a language
    missing from the table is visible rather than silently blank."""
    name = names.get(code)
    return f"{name} ({code})" if name else code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/built/tydiqa_answerability"))
    ap.add_argument("--splits", default="validation",
                    help="Comma-separated: validation, train, or both.")
    ap.add_argument("--out", type=Path,
                    default=Path("data/translated/questions_to_translate"))
    ap.add_argument("--expand-targets", action="store_true",
                    help="Also emit one row per (question, target language) "
                    "pair -- the actual translation job list.")
    ap.add_argument("--targets", default=",".join(MIST_LANGUAGES),
                    help="Target languages for --expand-targets. A question is "
                    "never paired with its own language.")
    ap.add_argument("--languages-csv", type=Path,
                    default=Path("../../languages.csv"),
                    help="Project language table, used for the plain-English "
                    "names. Columns 'code' and 'Language'.")
    args = ap.parse_args()

    names = load_language_names(args.languages_csv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seen: dict[tuple[str, str], dict] = {}
    per_split = collections.Counter()

    for split in splits:
        path = args.data_dir / f"{split}.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r["lang_code"], r["question"])
                per_split[split] += 1
                if key in seen:
                    seen[key]["n_instances"] += 1
                    seen[key]["splits"].add(split)
                    continue
                iso1, iso3 = ISO.get(r["lang_code"], ("", ""))
                seen[key] = {
                    "lang_code": r["lang_code"],
                    "iso639_1": iso1,
                    "iso639_3": iso3,
                    "question": r["question"],
                    "n_instances": 1,
                    "splits": {split},
                }

    rows = []
    for i, ((lang, _q), rec) in enumerate(sorted(seen.items())):
        rec = dict(rec)
        rec["qid"] = f"tydiq_{i:06d}"
        rec["splits"] = ",".join(sorted(rec["splits"]))
        rows.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "qid": r["qid"], "src_lang": r["lang_code"],
                "src_language": label(r["lang_code"], names),
                "iso639_1": r["iso639_1"], "iso639_3": r["iso639_3"],
                "question": r["question"], "n_instances": r["n_instances"],
                "splits": r["splits"],
            }, ensure_ascii=False) + "\n")

    tsv_path = args.out.with_suffix(".tsv")
    unsafe = [r for r in rows if any(c in r["question"] for c in "\t\r\n")]
    if unsafe:
        raise RuntimeError(f"{len(unsafe)} questions contain tab/newline; TSV unsafe")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        w.writerow(["qid", "src_lang", "src_language", "iso639_1", "iso639_3",
                    "question"])
        for r in rows:
            w.writerow([r["qid"], r["lang_code"], label(r["lang_code"], names),
                        r["iso639_1"], r["iso639_3"], r["question"]])

    per_lang = collections.Counter(r["lang_code"] for r in rows)
    chars = collections.Counter()
    for r in rows:
        chars[r["lang_code"]] += len(r["question"])

    print(f"Splits: {', '.join(splits)} "
          f"({sum(per_split.values()):,} dataset rows)")
    print(f"Unique questions to translate: {len(rows):,}")
    print(f"\n{'lang_code':10s} {'questions':>10s} {'chars':>10s} {'mean len':>9s}")
    print("-" * 43)
    for lang in sorted(per_lang):
        n = per_lang[lang]
        print(f"{lang:10s} {n:10,d} {chars[lang]:10,d} {chars[lang] / n:9.1f}")
    print("-" * 43)
    print(f"{'TOTAL':10s} {len(rows):10,d} {sum(chars.values()):10,d}")
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {tsv_path}")

    if args.expand_targets:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        pair_path = args.out.parent / (args.out.name + "_pairs.tsv")
        n_pairs = 0
        pair_chars = 0
        with open(pair_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
            w.writerow(["job_id", "qid", "src_lang", "tgt_lang",
                        "src_language", "tgt_language", "question"])
            for r in rows:
                for tgt in targets:
                    if tgt == r["lang_code"]:
                        continue  # never translate a question into its own language
                    w.writerow([f"{r['qid']}__{tgt}", r["qid"], r["lang_code"],
                                tgt, label(r["lang_code"], names),
                                label(tgt, names), r["question"]])
                    n_pairs += 1
                    pair_chars += len(r["question"])
        print(f"\nExpanded to {n_pairs:,} (question, target) pairs across "
              f"{len(targets)} target languages, {pair_chars:,} source chars.")
        print(f"Wrote {pair_path}")


if __name__ == "__main__":
    main()
