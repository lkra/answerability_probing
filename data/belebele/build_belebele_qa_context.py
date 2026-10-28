#!/usr/bin/env python3
"""Render BELEBELE as qa-context fine-tuning rows (answerable only).

Passage, question and answer (mc_answer{correct_answer_num}) are BELEBELE's own text in the target
language. Items whose question refers to the options (303 of 900, matched on the English question
with the patterns in probe/probing/_sources.py) are dropped. Each item is rendered in every question
language with the passage in English and --ctx-per-item - 1 other languages. Answers get the
language's sentence terminator (--no-punctuate to disable). Ids start at 700001. Bhojpuri is not in
BELEBELE.

Output: data/built/belebele_qa_context_train.jsonl, belebele_qa_context_stats.json
"""

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent  # data/

SEP = "\\n\\n"  # the organizers' literal backslash-n-backslash-n, not a newline
ID_RE = re.compile(r"^qa-context_\d+_[a-z]{3}_[a-z]{3}$")
_WS_RE = re.compile(r"\s+")

# The 25 context languages the qa-context test set uses. Belebele also ships
# swh/tel/tha, which are MIST languages but never carry a qa-context passage, so
# they are left out rather than adding off-distribution passage languages.
MIST_CONTEXT_LANGS = [
    "arb", "ben", "bho", "ces", "ckb", "deu", "eng", "fin", "fra", "hat", "hin",
    "ind", "ita", "jpn", "kor", "mar", "pes", "por", "rus", "slk", "spa", "tur",
    "vie", "yor", "zho",
]

# Kept byte-identical to probing/_sources.py's OPTION_DEPENDENT.
OPTION_DEPENDENT = [
    r"which of the following", r"which of these", r"all of the following",
    r"\bof the following\b", r"none of the (above|following)",
    r"which statement", r"which one",
]

# Sentence terminator per question language, measured from
# data/silver/silver_test_labels_final.jsonl (100% agreement within each language).
# Note `mar`: Marathi is written in Devanagari but the references use a full
# stop, not the danda that ben/bho/hin use. Measured, not inferred from script.
TERMINATOR = collections.defaultdict(lambda: ".", {
    "ben": "।", "bho": "।", "hin": "।",   # danda
    "jpn": "。", "zho": "。",                     # ideographic full stop
})
_SENT_END = ".!?。！？؟।"


def clean(s: str) -> str:
    """Same rule as build_qa_context_train.py's clean_text().

    The whitespace collapse is the load-bearing part: the organizers' prompts
    contain zero real newline characters, so a Belebele passage carrying one
    through would be a train/test format mismatch.
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def norm(s: str) -> str:
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip().lower()


def is_option_dependent(question: str) -> bool:
    return any(re.search(p, question, re.I) for p in OPTION_DEPENDENT)


def rotate(seq: list, start: int, k: int) -> list:
    return [seq[(start + i) % len(seq)] for i in range(k)]


def load_belebele(data_dir: Path) -> tuple[dict, dict]:
    """-> ({lang3: {item_key: row}}, {lang3: flores_code})"""
    by_lang, codes = {}, {}
    for path in sorted(data_dir.glob("*.jsonl")):
        flores = path.stem
        lang3 = flores.split("_")[0]
        rows = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[(r["link"], int(r["question_number"]))] = r
        by_lang[lang3] = rows
        codes[lang3] = flores
    return by_lang, codes


def organizer_sample_items(sample_path: Path, by_lang: dict) -> dict[str, set]:
    """{lang3: item keys released in the organisers' sample in that language}. Kept per (item,
    language) because the per-language subsets differ and their union is all 900 items; matched on
    the question string, since a passage is shared by ~1.8 questions.
    """
    out: dict[str, set] = collections.defaultdict(set)
    if not sample_path.exists():
        return out
    inputs: dict[str, list] = collections.defaultdict(list)
    with sample_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "elebele" not in r.get("source", ""):
                continue
            inputs[r["lang_code"].split("_")[0]].append(norm(r["input"]))
    for lang3, texts in inputs.items():
        blob = " || ".join(texts)
        for key, row in by_lang.get(lang3, {}).items():
            q = norm(row["question"])
            if q and q in blob:
                out[lang3].add(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--belebele-dir", type=Path, default=DATA_DIR / "upstream/belebele")
    ap.add_argument("--templates", type=Path,
                    default=DATA_DIR / "qa_context_templates.json")
    ap.add_argument("--tests", type=Path, default=DATA_DIR / "test/tests.jsonl")
    ap.add_argument("--sample", type=Path,
                    default=DATA_DIR / "upstream/wmt26-mist-sample/samples.jsonl")
    ap.add_argument("--out", type=Path,
                    default=DATA_DIR / "built/belebele_qa_context_train.jsonl")
    ap.add_argument("--stats-out", type=Path,
                    default=DATA_DIR / "built/belebele_qa_context_stats.json")
    ap.add_argument("--ctx-per-item", type=int, default=4,
                    help="Context languages per item (English + N-1 rotating others). "
                         "597 items x N x 23 question languages rows.")
    ap.add_argument("--start-index", type=int, default=700001,
                    help="Row-id counter start. Default sits above both "
                         "qa_context_train.jsonl (from 1) and udhr_qa (from 500001) "
                         "so all three concatenate without id collisions.")
    ap.add_argument("--no-punctuate", action="store_true",
                    help="Leave Belebele option text exactly as it ships, unpunctuated.")
    ap.add_argument("--drop-organizer-sample", action="store_true",
                    help="Exclude items that appear in the organizers' sample release "
                         "instead of only flagging them.")
    args = ap.parse_args()

    templates = json.loads(args.templates.read_text(encoding="utf-8"))["templates"]
    by_lang, codes = load_belebele(args.belebele_dir)
    if "eng" not in by_lang:
        raise SystemExit(f"error: no eng_Latn.jsonl in {args.belebele_dir}")
    print(f"Belebele: {len(by_lang)} languages, {len(by_lang['eng'])} items each")

    question_langs = sorted(set(templates) & set(by_lang))
    context_langs = [l for l in MIST_CONTEXT_LANGS if l in by_lang]
    missing_q = sorted(set(templates) - set(by_lang))
    print(f"question languages: {len(question_langs)}/24"
          + (f"   missing: {missing_q} (not in Belebele)" if missing_q else ""))
    print(f"context languages : {len(context_langs)}/25")

    # ---- items ---------------------------------------------------------------
    eng = by_lang["eng"]
    all_keys = sorted(eng)
    parallel = [k for k in all_keys if all(k in by_lang[l] for l in question_langs + context_langs)]
    if len(parallel) != len(all_keys):
        print(f"  {len(all_keys) - len(parallel)} items are not present in every "
              f"language and were dropped")

    option_dependent = [k for k in parallel if is_option_dependent(eng[k]["question"])]
    usable = [k for k in parallel if k not in set(option_dependent)]
    print(f"\nitems: {len(parallel)} parallel, {len(option_dependent)} option-dependent "
          f"dropped ({len(option_dependent) / max(len(parallel), 1):.1%}), "
          f"{len(usable)} usable")

    in_sample = organizer_sample_items(args.sample, by_lang)
    union = set().union(*in_sample.values()) if in_sample else set()
    per_lang_n = {l: len(v & set(usable)) for l, v in sorted(in_sample.items())}
    print(f"\norganizers' sample overlap: {len(union & set(usable))}/{len(usable)} usable "
          f"items appear in SOME language; per language "
          f"{min(per_lang_n.values()) if per_lang_n else 0}-"
          f"{max(per_lang_n.values()) if per_lang_n else 0} of {len(usable)}")
    print(f"  flagged per (row, question language) as `in_organizer_sample`; an "
          f"item-level flag would mark everything")
    if args.drop_organizer_sample:
        before = len(usable)
        usable = [k for k in usable if k not in union]
        print(f"  --drop-organizer-sample: {before} -> {len(usable)} items "
              f"(the union covers every item, so this empties the set)")

    # ---- render --------------------------------------------------------------
    n = args.start_index
    written = 0
    stats = {
        "by_question_lang": collections.Counter(),
        "by_context_lang": collections.Counter(),
        "by_pair": collections.Counter(),
    }
    non_eng_ctx = [c for c in context_langs if c != "eng"]
    n_ctx = max(1, min(args.ctx_per_item, len(context_langs)))
    if args.ctx_per_item > len(context_langs):
        raise SystemExit(f"error: --ctx-per-item={args.ctx_per_item} exceeds the "
                         f"{len(context_langs)} context languages available.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, key in enumerate(usable):
            # English always present, mirroring the test set (each of its 100
            # items has exactly one English-context rendering), plus a rotating
            # slice so every language gets its turn.
            ctxs = (["eng"] + rotate(non_eng_ctx, i * (n_ctx - 1), n_ctx - 1)
                    if n_ctx > 1 else [context_langs[i % len(context_langs)]])

            for qlang in question_langs:
                t = templates[qlang]
                row_q = by_lang[qlang][key]
                question = clean(row_q["question"])
                answer = clean(row_q[f"mc_answer{int(row_q['correct_answer_num'])}"])
                if not question or not answer:
                    continue
                if not args.no_punctuate and answer[-1] not in _SENT_END:
                    answer += TERMINATOR[qlang]

                for clang in ctxs:
                    passage = clean(by_lang[clang][key]["flores_passage"])
                    prompt = SEP.join([t["intro"], passage,
                                       t["q_prefix"] + question, t["instruction"]])
                    f.write(json.dumps({
                        "id": f"qa-context_{n}_{qlang}_{clang}",
                        "prompt": prompt,
                        "completion": answer,
                        "task": "qa-context",
                        "question_lang": qlang,
                        # provenance
                        "source": "belebele",
                        "item_id": f"belebele_qa_{i:04d}",
                        "belebele_link": key[0],
                        "question_number": key[1],
                        "context_lang": clang,
                        "answerable": True,
                        "in_organizer_sample": key in in_sample.get(qlang, ()),
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    written += 1
                    stats["by_question_lang"][qlang] += 1
                    stats["by_context_lang"][clang] += 1
                    stats["by_pair"][f"{clang}->{qlang}"] += 1

    print(f"\nwrote {written:,} rows to {args.out}")
    print(f"  {len(usable)} items x {n_ctx} context langs x {len(question_langs)} "
          f"question langs")
    print(f"  language-pair grid: {len(stats['by_pair'])} of "
          f"{len(context_langs) * len(question_langs)} cells populated")
    print(f"  rows per question language: {min(stats['by_question_lang'].values()):,}")

    # ---- verify --------------------------------------------------------------
    print("\nverifying")
    rows = [json.loads(l) for l in args.out.open(encoding="utf-8")]
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"   {detail}"))
        if not ok:
            failures.append(f"{name}: {detail}")

    bad = [r["id"] for r in rows if "\n" in r["prompt"] or "\n" in r["completion"]]
    check("no real newline in prompt or completion", not bad, f"{len(bad)} rows")
    ids = [r["id"] for r in rows]
    check("row ids unique", len(set(ids)) == len(ids),
          f"{len(ids) - len(set(ids))} duplicates")
    check("row ids match the test-set pattern", all(ID_RE.match(i) for i in ids))

    test_rows = [json.loads(l) for l in args.tests.open(encoding="utf-8")]
    test_ids = {r["id"] for r in test_rows}
    check("no row id collides with a real test id", not (set(ids) & test_ids))

    seg = intro = instr = pref = 0
    for r in rows:
        p = r["prompt"].split(SEP)
        if len(p) != 4:
            seg += 1
            continue
        t = templates[r["question_lang"]]
        intro += p[0] != t["intro"]
        instr += p[3] != t["instruction"]
        pref += not p[2].startswith(t["q_prefix"])
    check("every prompt splits into 4 segments", not seg, f"{seg} rows")
    check("intro byte-identical to the organizers' template", not intro, f"{intro} rows")
    check("instruction byte-identical to the organizers' template", not instr, f"{instr} rows")
    check("question prefix byte-identical to the organizers' template", not pref, f"{pref} rows")

    literals = {t["not_answerable"] for t in templates.values()}
    leaked = [r["id"] for r in rows if r["completion"] in literals]
    check("no completion is an abstention literal", not leaked, f"{len(leaked)} rows")
    check("no empty completion", all(r["completion"].strip() for r in rows))

    # Contamination, both directions, against every test prompt.
    test_blob = "\n\n".join(r["prompt"] for r in test_rows)
    ours = {(r["item_id"], r["context_lang"]): r["prompt"].split(SEP)[1] for r in rows}
    inside = [k for k, v in ours.items() if v and v in test_blob]
    check("no generated passage occurs inside any test prompt", not inside,
          f"{len(inside)} passages")
    test_passages = {p[1] for r in test_rows if r["task"] == "qa-context"
                     for p in [r["prompt"].split(SEP)] if len(p) == 4}
    our_blob = "\n\n".join(ours.values())
    rev = [p for p in test_passages if p and p in our_blob]
    check("no test passage occurs inside any generated passage", not rev,
          f"{len(rev)} test passages")

    check("all available question languages present",
          len(stats["by_question_lang"]) == len(question_langs))
    ctx_counts = {k: v for k, v in stats["by_context_lang"].items() if k != "eng"}
    spread = max(ctx_counts.values()) / max(min(ctx_counts.values()), 1)
    check("non-English context languages within 2x of each other", spread <= 2.0,
          f"max/min = {spread:.2f}")

    print(f"\n{len(rows) and ''}{'=' * 56}")
    print(f"{sum(1 for _ in range(0))}{len(failures)} checks failed" if failures
          else "all checks passed")
    for f_ in failures:
        print(f"  FAILED: {f_}")

    args.stats_out.write_text(json.dumps({
        "n_rows": written,
        "n_items": len(usable),
        "n_option_dependent_dropped": len(option_dependent),
        "n_items_in_organizer_sample_any_lang": len(union & set(usable)),
        "organizer_sample_items_per_lang": per_lang_n,
        "ctx_per_item": n_ctx,
        "question_langs": question_langs,
        "context_langs": context_langs,
        "missing_question_langs": missing_q,
        "punctuated": not args.no_punctuate,
        "checks_failed": len(failures),
        **{k: dict(v) for k, v in stats.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.stats_out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
