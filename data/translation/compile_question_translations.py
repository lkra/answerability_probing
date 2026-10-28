#!/usr/bin/env python3
"""Compile the per-language question translations ({id, lang, question} per file) into one
long-format file with source and target language per row, joined on the export spec
(questions_to_translate.tsv). Coverage is checked against the spec minus self-pairs. Quality flags
(self_pair, empty, identical_to_source, degenerate) are recorded; --exclude-flagged drops them. Every
row carries the model name.

    python3 data/translation/compile_question_translations.py [--translations-dir <dir>] [--output <file>] [--coverage-csv <csv>]
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TRANSLATIONS_DIR = REPO_ROOT / "data" / "translated"
DEFAULT_SPEC = DEFAULT_TRANSLATIONS_DIR / "questions_to_translate.tsv"
# The model behind the 17-language run: translate_questions.py's MODELS["hymt2"], i.e.
# tencent/Hy-MT2-7B-FP8. Pass --model-name when compiling the output of a different model
# (e.g. the MADLAD-400 run that covers the six languages Hy-MT2 cannot reach).
DEFAULT_MODEL_NAME = "Hy-MT2-7B-FP8"

# The 28 WMT26 MIST languages, keyed by the English name `translate_questions.py` uses for the
# output filename (lowercased). Names must match TARGET_LANGUAGES / MADLAD_EXTRA_LANGUAGES there.
# French/Swahili/Telugu/Thai are in neither of that script's language lists, so no run can
# currently produce them; they are listed here so the coverage report names them as absent rather
# than silently omitting them. English was in the same position until it was added as a target --
# the Hy-MT2 run predates that, so `english.jsonl` is absent from it and present only in runs made
# afterwards.
NAME_TO_FLORES = {
    "arabic": "arb_Arab", "bengali": "ben_Beng", "bhojpuri": "bho_Deva", "czech": "ces_Latn",
    "central kurdish": "ckb_Arab", "german": "deu_Latn", "english": "eng_Latn",
    "finnish": "fin_Latn", "french": "fra_Latn", "haitian creole": "hat_Latn",
    "hindi": "hin_Deva", "indonesian": "ind_Latn", "italian": "ita_Latn", "japanese": "jpn_Jpan",
    "korean": "kor_Hang", "marathi": "mar_Deva", "persian": "pes_Arab", "portuguese": "por_Latn",
    "russian": "rus_Cyrl", "slovak": "slk_Latn", "spanish": "spa_Latn", "swahili": "swh_Latn",
    "telugu": "tel_Telu", "thai": "tha_Thai", "turkish": "tur_Latn", "vietnamese": "vie_Latn",
    "yoruba": "yor_Latn", "chinese": "zho_Hans",
}

# Files in the translations dir that are inputs, not per-language translation output. The compiled
# file this run writes is excluded separately (see discover_files); "translated_questions" is the
# name an earlier version of this script used, kept so a stale one is never re-ingested.
NON_TRANSLATION_STEMS = {"questions_to_translate", "translated_questions"}

# A chunk of 6+ characters repeated 3+ times back-to-back. Catches the decoder getting stuck in a
# loop ("barvy barvy barvy ..."), which at max_new_tokens=256 also truncates the real translation.
DEGENERATE_RE = re.compile(r"(.{6,}?)\1{2,}")


def load_spec(path):
    """qid -> {src_lang, question}. This is the authority on what jobs exist. The spec is the
    TSV `export_questions_for_translation.py` writes; a JSONL with the same fields also works."""
    spec = {}
    with open(path, encoding="utf-8", newline="") as f:
        if str(path).endswith(".tsv"):
            import csv
            for o in csv.DictReader(f, delimiter="\t"):
                spec[o["qid"]] = {"src_lang": o["src_lang"], "question": o["question"]}
        else:
            for line in f:
                o = json.loads(line)
                spec[o["qid"]] = {"src_lang": o["src_lang"], "question": o["question"]}
    return spec


def expected_counts(spec):
    """target FLORES code -> number of jobs, i.e. every question not already in that language."""
    per_src = collections.Counter(v["src_lang"] for v in spec.values())
    total = len(spec)
    return {code: total - per_src.get(code, 0) for code in NAME_TO_FLORES.values()}


def discover_files(translations_dir, exclude=()):
    """[(flores_code, path)] for each recognised per-language file, sorted by code.

    `exclude` holds paths this run writes itself (the compiled output), so re-running in place
    does not re-ingest its own product or warn about it.
    """
    excluded = {p.resolve() for p in exclude}
    found, unknown = [], []
    for path in sorted(translations_dir.glob("*.jsonl")):
        if path.stem in NON_TRANSLATION_STEMS or path.resolve() in excluded:
            continue
        code = NAME_TO_FLORES.get(path.stem.lower())
        if code is None:
            unknown.append(path)
        else:
            found.append((code, path))
    return sorted(found), unknown


def read_language(path, code, spec, model_name):
    """Parse one language file into rows plus a per-language problem tally.

    Rows are returned even when flagged; `flags` on a row names every check it tripped.
    """
    rows, seen = [], set()
    stats = collections.Counter()
    for lineno, line in enumerate(open(path), 1):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            stats["malformed_json"] += 1
            continue

        qid, question = o.get("id"), o.get("question", "")
        if qid not in spec:
            stats["unknown_qid"] += 1
            continue
        if qid in seen:
            stats["duplicate_qid"] += 1
            continue
        seen.add(qid)

        src_lang = spec[qid]["src_lang"]
        flags = []
        if src_lang == code:
            # The export emits no self-pairs, so this means the run translated a question into the
            # language it was already in -- possible when a source language is missing from
            # translate_questions.py's ISO_TO_TARGET_LANG skip map.
            flags.append("self_pair")
        if not question.strip():
            flags.append("empty")
        if question.strip() and question.strip() == spec[qid]["question"].strip():
            flags.append("identical_to_source")
        if DEGENERATE_RE.search(question):
            flags.append("degenerate")

        for flag in flags:
            stats[flag] += 1
        rows.append({
            "qid": qid,
            "src_lang": src_lang,
            "tgt_lang": code,
            "src_question": spec[qid]["question"],
            "question": question,
            "model": model_name,
            "flags": flags,
        })
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translations-dir", type=Path, default=DEFAULT_TRANSLATIONS_DIR)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC,
                        help="questions_to_translate.tsv -- the qid -> src_lang authority")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help=f"MT model these translations came from; names the output file and is "
                             f"stamped on every row (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--output", type=Path, default=None,
                        help="compiled long-format JSONL (default: <translations-dir>/<model-name>.jsonl)")
    parser.add_argument("--coverage-csv", type=Path, default=None,
                        help="also write the per-language coverage table as CSV")
    parser.add_argument("--exclude-flagged", action="store_true",
                        help="drop rows with any quality flag (default: keep, flags are recorded)")
    parser.add_argument("--no-write", action="store_true", help="report coverage only")
    args = parser.parse_args()
    output = args.output or (args.translations_dir / f"{args.model_name}.jsonl")

    if not args.spec.exists():
        sys.exit(f"Spec not found: {args.spec}")
    spec = load_spec(args.spec)
    expected = expected_counts(spec)
    files, unknown = discover_files(args.translations_dir, exclude=[output])

    print(f"Model: {args.model_name}")
    print(f"Spec: {len(spec)} unique questions across "
          f"{len(set(v['src_lang'] for v in spec.values()))} source languages")
    print(f"Target languages: {len(NAME_TO_FLORES)} in MIST, {len(files)} with a translation file\n")
    for path in unknown:
        print(f"WARNING: unrecognised file, not compiled: {path.name} "
              f"(add its language to NAME_TO_FLORES if it is a translation)")

    all_rows, per_lang, all_stats = [], {}, collections.Counter()
    for code, path in files:
        rows, stats = read_language(path, code, spec, args.model_name)
        per_lang[code] = (len(rows), stats)
        all_stats.update(stats)
        all_rows.extend(rows)

    # --- coverage table -------------------------------------------------------------------
    header = f"{'target':10s} {'file':18s} {'got':>6s} {'expect':>7s} {'missing':>8s} {'cov%':>7s}  flags"
    print(header)
    print("-" * len(header))
    table = []
    for code in sorted(NAME_TO_FLORES.values()):
        exp = expected[code]
        path = dict((c, p) for c, p in files).get(code)
        if path is None:
            print(f"{code:10s} {'-- no file --':18s} {0:6d} {exp:7d} {exp:8d} {0.0:6.1f}%")
            table.append((code, "", 0, exp, exp, 0.0, ""))
            continue
        got, stats = per_lang[code]
        missing = exp - got
        cov = 100.0 * got / exp if exp else 0.0
        flagstr = ", ".join(f"{k}={v}" for k, v in sorted(stats.items())) if stats else ""
        print(f"{code:10s} {path.name:18s} {got:6d} {exp:7d} {missing:8d} {cov:6.1f}%  {flagstr}")
        table.append((code, path.name, got, exp, missing, round(cov, 2), flagstr))

    done = [c for c in per_lang]
    total_got = sum(per_lang[c][0] for c in done)
    total_exp_done = sum(expected[c] for c in done)
    total_exp_all = sum(expected.values())
    print("-" * len(header))
    print(f"{'TOTAL':10s} {'(' + str(len(done)) + ' langs)':18s} {total_got:6d} {total_exp_done:7d} "
          f"{total_exp_done - total_got:8d} {100.0 * total_got / total_exp_done if total_exp_done else 0:6.1f}%")
    print(f"\nAgainst all {len(NAME_TO_FLORES)} MIST target languages: "
          f"{total_got}/{total_exp_all} pairs ({100.0 * total_got / total_exp_all:.1f}%)")

    if all_stats:
        print("\nQuality flags (rows are kept unless --exclude-flagged):")
        for k, v in sorted(all_stats.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v}")

    # --- write ----------------------------------------------------------------------------
    if args.coverage_csv:
        import csv
        args.coverage_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.coverage_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tgt_lang", "file", "got", "expected", "missing", "coverage_pct", "flags"])
            w.writerows(table)
        print(f"\nWrote coverage table to {args.coverage_csv}")

    if args.no_write:
        return
    out_rows = [r for r in all_rows if not r["flags"]] if args.exclude_flagged else all_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dropped = len(all_rows) - len(out_rows)
    print(f"\nWrote {len(out_rows)} pairs to {output}" + (f" ({dropped} flagged rows excluded)" if dropped else ""))


if __name__ == "__main__":
    main()
