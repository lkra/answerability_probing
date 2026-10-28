#!/usr/bin/env python3
r"""Build the TyDiQA answerability set from copenlu/answerable_tydiqa (11 languages, ~50/50);
annotations.answer_start[0] == -1 marks unanswerable. Train and validation stay separate splits;
rows whose passage appears in the organisers' sample or in the other split are flagged
(--exclude-contaminated drops them).

Output rows: id, split, source, language, lang_code, answerable, context, question, input
("{context}\n\n{question}"), output (gold span, or --unanswerable-target), passage_id,
in_organizer_sample, passage_in_other_split.

    python3 data/tydi/build_tydiqa.py     # -> data/built/tydiqa_answerability/{train,validation}.jsonl
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

SOURCE = "copenlu/answerable_tydiqa"

# TyDi QA language names -> MIST "lang_Script" codes. All 11 are in the WMT26
# MIST language list.
LANG_CODES = {
    "arabic": "arb_Arab",
    "bengali": "ben_Beng",
    "english": "eng_Latn",
    "finnish": "fin_Latn",
    "indonesian": "ind_Latn",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "russian": "rus_Cyrl",
    "swahili": "swh_Latn",
    "telugu": "tel_Telu",
    "thai": "tha_Thai",
}

# Abstention strings verified in the organizer data (MCIF). Only English
# overlaps the TyDi languages -- the other 10 have no organizer-sanctioned
# phrasing, which is exactly why --unanswerable-target defaults to "empty".
VERIFIED_ABSTENTION = {
    "eng_Latn": "Not answerable.",
    "deu_Latn": "Nicht zu beantworten.",
    "ita_Latn": "Non è possibile rispondere.",
    "zho_Hans": "无法回答。",
}


def content_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def is_unanswerable(annotations) -> bool:
    return int(annotations["answer_start"][0]) == -1


def gold_answer(annotations) -> str:
    return str(annotations["answer_text"][0])


def load_split(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(df["language"]) - set(LANG_CODES)
    if missing:
        raise RuntimeError(f"Unmapped languages in {path}: {sorted(missing)}")
    df = df.copy()
    df["split"] = split
    df["answerable"] = (~df["annotations"].map(is_unanswerable)).astype(int)
    df["gold"] = df["annotations"].map(gold_answer)
    # The organizers render TyDi rows as `document_plaintext.lstrip()` + "\n\n"
    # + `question_text` verbatim. Reproduced exactly (verified: all 3,112 of
    # their TyDi inputs are byte-identical under this rendering, and none under
    # strip/rstrip/no-strip variants), so rows built here are drop-in
    # compatible with the organizer mix.
    df["context"] = df["document_plaintext"].str.lstrip()
    df["question"] = df["question_text"]
    return df


def organizer_passages(samples_path: Path) -> set[str]:
    """Passages of the organizer sample's TyDi rows. Their `input` is rendered
    as `context\\n\\nquestion`, so the passage is everything before the final
    blank line."""
    passages = set()
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["source"] != SOURCE:
                continue
            parts = row["input"].rsplit("\n\n", 1)
            passages.add(content_hash(parts[0]))
    return passages


def abstention_target(lang_code: str, mode: str) -> str:
    if mode == "empty":
        return ""
    if mode == "english":
        return VERIFIED_ABSTENTION["eng_Latn"]
    raise ValueError(f"unknown --unanswerable-target: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tydiqa-dir", type=Path, default=Path("data/upstream/answerable_tydiqa")
    )
    parser.add_argument(
        "--samples-path",
        type=Path,
        default=Path("data/upstream/wmt26-mist-sample/samples.jsonl"),
        help="Organizer sample, used only to flag passage overlap.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/built/tydiqa_answerability")
    )
    parser.add_argument(
        "--unanswerable-target",
        choices=["empty", "english"],
        default="empty",
        help="What an unanswerable instance's target output should be. "
        "'empty' (default) writes '' and leaves the decision to downstream "
        "code -- correct for probing setups that never need the string. "
        "'english' writes the verified MCIF string 'Not answerable.' in every "
        "language, i.e. a language-invariant abstention target. There is "
        "deliberately no option that writes guessed per-language translations.",
    )
    parser.add_argument(
        "--exclude-contaminated",
        action="store_true",
        help="Drop rows whose passage appears in the organizer sample or in "
        "the other split.",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Also write the per-language breakdown as CSV.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Drop rows with a longer passage (0 = keep all).",
    )
    args = parser.parse_args()

    frames = []
    for split in ("train", "validation"):
        path = args.tydiqa_dir / f"{split}.parquet"
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Reading {path} ...")
        frames.append(load_split(path, split))
    train_df, val_df = frames

    print(f"Reading organizer sample from {args.samples_path} ...")
    org = organizer_passages(args.samples_path)
    print(f"{len(org)} distinct TyDi passages in the organizer mix.")

    train_passages = set(train_df["context"].map(content_hash))
    val_passages = set(val_df["context"].map(content_hash))

    records = []
    for df, other_passages in ((train_df, val_passages), (val_df, train_passages)):
        split = df["split"].iloc[0]
        for i, row in enumerate(df.itertuples(index=False)):
            pid = content_hash(row.context)
            lang_code = LANG_CODES[row.language]
            output = (
                row.gold
                if row.answerable
                else abstention_target(lang_code, args.unanswerable_target)
            )
            records.append(
                {
                    "id": f"tydiqa-{split}_{i:06d}",
                    "split": split,
                    "source": SOURCE,
                    "language": row.language,
                    "lang_code": lang_code,
                    "answerable": int(row.answerable),
                    "context": row.context,
                    "question": row.question,
                    "input": f"{row.context}\n\n{row.question}",
                    "output": output,
                    "passage_id": pid,
                    "in_organizer_sample": pid in org,
                    "passage_in_other_split": pid in other_passages,
                }
            )

    n_before = len(records)
    if args.max_context_chars:
        records = [
            r for r in records if len(r["context"]) <= args.max_context_chars
        ]
    if args.exclude_contaminated:
        records = [
            r
            for r in records
            if not r["in_organizer_sample"] and not r["passage_in_other_split"]
        ]
    if len(records) != n_before:
        print(f"Filtered {n_before - len(records)} rows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        rows = [r for r in records if r["split"] == split]
        path = args.output_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as out:
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {path}")

    print_stats(records)
    if args.stats_csv:
        write_stats_csv(records, args.stats_csv)
        print(f"Wrote per-language stats to {args.stats_csv}")


def write_stats_csv(records: list[dict], path: Path) -> None:
    per = collections.defaultdict(collections.Counter)
    for r in records:
        per[(r["lang_code"], r["language"])][(r["split"], r["answerable"])] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "language",
                "lang_code",
                "train_total",
                "train_unanswerable",
                "train_answerable",
                "validation_total",
                "validation_unanswerable",
                "validation_answerable",
                "total",
                "total_unanswerable",
                "pct_unanswerable",
            ]
        )
        for (lang_code, language), c in sorted(
            per.items(), key=lambda kv: -sum(kv[1].values())
        ):
            tr_u, tr_a = c[("train", 0)], c[("train", 1)]
            va_u, va_a = c[("validation", 0)], c[("validation", 1)]
            total = tr_u + tr_a + va_u + va_a
            total_u = tr_u + va_u
            writer.writerow(
                [
                    language,
                    lang_code,
                    tr_u + tr_a,
                    tr_u,
                    tr_a,
                    va_u + va_a,
                    va_u,
                    va_a,
                    total,
                    total_u,
                    f"{100 * total_u / total:.2f}",
                ]
            )


def print_stats(records: list[dict]) -> None:
    per = collections.defaultdict(collections.Counter)
    for r in records:
        per[(r["split"], r["lang_code"])][r["answerable"]] += 1

    print(f"\n{'split':11s} {'lang':10s} {'total':>7s} {'unans':>7s} {'%unans':>7s}")
    print("-" * 46)
    for (split, lang), c in sorted(per.items()):
        total = c[0] + c[1]
        print(f"{split:11s} {lang:10s} {total:7d} {c[0]:7d} {100*c[0]/total:6.1f}%")
    print("-" * 46)
    tot = collections.Counter()
    for c in per.values():
        tot += c
    grand = tot[0] + tot[1]
    print(f"{'ALL':11s} {'':10s} {grand:7d} {tot[0]:7d} {100*tot[0]/grand:6.1f}%")

    flagged = sum(
        1 for r in records if r["in_organizer_sample"] or r["passage_in_other_split"]
    )
    print(f"\n{flagged} rows flagged as passage-overlapping (not dropped unless "
          f"--exclude-contaminated).")


if __name__ == "__main__":
    main()
