#!/usr/bin/env python3
"""Append one target language's shard to the compiled translation file (Hy-MT2-7B-FP8.jsonl)
without recompiling: the per-language shards behind the compiled file no longer exist, so a
recompile would replace it. Rows are built with compile_question_translations.read_language (same
fields and quality flags); the existing bytes are hashed before and after and the run aborts if they
changed. Keep shards in a subdirectory, not beside the compiled file.

    python3 data/translation/append_language_to_compiled.py --shard <dir>/english.jsonl --lang eng_Latn
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from compile_question_translations import load_spec, read_language  # noqa: E402

DEFAULT_COMPILED = REPO_ROOT / "data/translated/Hy-MT2-7B-FP8.jsonl"
DEFAULT_SPEC = REPO_ROOT / "data/translated/questions_to_translate.tsv"


def digest(path, nbytes):
    """SHA-256 of the first `nbytes` of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = nbytes
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=Path, required=True,
                    help="per-language file from translate_questions.py: {id, lang, question}")
    ap.add_argument("--lang", required=True, help="FLORES code of the target, e.g. eng_Latn")
    ap.add_argument("--compiled", type=Path, default=DEFAULT_COMPILED)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--model-name", default=None,
                    help="default: read from the compiled file's existing rows, so the new rows "
                         "cannot be labelled with a different model than the file they join")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    size_before = args.compiled.stat().st_size
    hash_before = digest(args.compiled, size_before)

    existing = collections.Counter()
    models = set()
    with open(args.compiled, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            existing[o["tgt_lang"]] += 1
            models.add(o["model"])
    print(f"compiled: {sum(existing.values()):,} rows, {len(existing)} targets, "
          f"model(s) {sorted(models)}")

    # Appending a language that is already in there would double it, and the duplicate would be
    # invisible downstream because every consumer groups by (qid, tgt_lang) without asserting
    # uniqueness.
    if args.lang in existing:
        raise SystemExit(f"{args.lang} is ALREADY in {args.compiled.name} "
                         f"({existing[args.lang]:,} rows). Refusing to append a duplicate.")
    if args.model_name is None:
        if len(models) != 1:
            raise SystemExit(f"compiled file mixes models {sorted(models)}; pass --model-name")
        args.model_name = models.pop()

    spec = load_spec(args.spec)
    rows, stats = read_language(args.shard, args.lang, spec, args.model_name)
    print(f"shard: {len(rows):,} rows for {args.lang}, model={args.model_name}")
    if stats:
        print("  flags: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    expected = sum(1 for q in spec.values() if q["src_lang"] != args.lang)
    print(f"  expected {expected:,} jobs (every question not already in {args.lang}); "
          f"got {len(rows):,}"
          + ("" if len(rows) == expected else "  <-- MISMATCH, check the shard"))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    with open(args.compiled, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The point of the whole script: the pre-existing bytes must be exactly as they were.
    if digest(args.compiled, size_before) != hash_before:
        raise SystemExit("FATAL: the pre-existing rows changed. Restore from git and investigate.")
    print(f"\nappended {len(rows):,} rows; first {size_before:,} bytes verified unchanged")
    print(f"{args.compiled}: {sum(existing.values()) + len(rows):,} rows, "
          f"{len(existing) + 1} targets")


if __name__ == "__main__":
    main()
