#!/usr/bin/env python3
r"""Render TyDiQA/cross fine-tuning rows from the translated jobs.

Joins data/translated/qa_translation_items.jsonl and the translation dirs (later dirs win on id
collision: hymt2, hymt2_pes_fix, madlad400/kept) onto the TyDi passages. The passage stays in its
source language; question and completion are the translations; the prompt wrapper is the organisers'
template for the question language. Rows: {id, prompt, completion, task, question_lang},
id = qa-context_<n>_<question_lang>_<passage_lang>.

--answer-form sentence (default): the translated answer sentence, falling back to the span
(--no-fallback drops instead). span: the gold span. adaptive: the sentence if at most
--max-sentence-chars, else the span.

    python3 data/tydi/build_qa_context_train.py \
        --out data/built/qa_context_tydi_train.jsonl --stats-out data/built/qa_context_tydi_stats.json
    python3 data/tydi/build_qa_context_train.py --answer-form adaptive --max-sentence-chars 120 \
        --out data/built/qa_context_tydi_adaptive_train.jsonl --stats-out data/built/qa_context_tydi_adaptive_stats.json
"""

import argparse
import collections
import json
import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # data/
# Merged in this order: a later dir wins on an id collision, so the Persian re-translation
# (hymt2_pes_fix) overrides hymt2's buggy arb->pes rows. madlad400_kept is the
# back-translation-filtered MADLAD output for bho/ckb/fin/hat/slk.
DEFAULT_TRANSLATED_DIRS = [
    ROOT / "translated/finetune/hymt2",
    ROOT / "translated/finetune/hymt2_pes_fix",
    ROOT / "translated/madlad400/kept",
]

# Kept in sync with build_translation_jobs.py's clean_text() - passages need
# identical cleanup so what the model reads at train time matches what
# question/answer cleaning already assumed.
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_HTML_TAG_RE = re.compile(r"<[^<>]{0,300}?>")
# \s+ (any whitespace run, including newlines), not build_translation_jobs.py's [ \t ]+
# (spaces/tabs only): that version is applied to question/answer text, which is short
# and effectively never contains a real newline, so the gap never showed up there. A
# multi-paragraph Wikipedia passage routinely does - real qa-context test prompts have
# zero actual newline characters anywhere (confirmed against data/test/tests.jsonl), so a
# passage with an embedded line break here would be a real train/test format mismatch.
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return text
    text = _HTML_TAG_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# FLORES-200 code -> ISO 639-3, for the 28 MIST languages. Mostly kept in sync
# with build_translation_jobs.py's LANGUAGES table (that dict's second tuple
# element) - duplicated rather than imported, matching this project's
# standalone-script convention - EXCEPT arb_Arab: that table has "ara" (the
# general Arabic macrolanguage code), but the organizers' actual question_lang
# in tests.jsonl / qa_context_templates.json uses "arb" (Standard Arabic
# specifically, matching the FLORES tag directly). build_translation_jobs.py
# never notices because it only uses this code for cosmetic labeling; template
# lookup here needs it to match the organizers' scheme exactly, so this one
# entry deliberately diverges from that table - confirmed against the real
# generated qa_context_templates.json, not guessed.
FLORES_TO_ISO3 = {
    "arb_Arab": "arb", "ben_Beng": "ben", "bho_Deva": "bho", "ces_Latn": "ces",
    "ckb_Arab": "ckb", "deu_Latn": "deu", "eng_Latn": "eng", "fin_Latn": "fin",
    "fra_Latn": "fra", "hat_Latn": "hat", "hin_Deva": "hin", "ind_Latn": "ind",
    "ita_Latn": "ita", "jpn_Jpan": "jpn", "kor_Hang": "kor", "mar_Deva": "mar",
    "pes_Arab": "pes", "por_Latn": "por", "rus_Cyrl": "rus", "slk_Latn": "slk",
    "spa_Latn": "spa", "swh_Latn": "swh", "tel_Telu": "tel", "tha_Thai": "tha",
    "tur_Latn": "tur", "vie_Latn": "vie", "yor_Latn": "yor", "zho_Hans": "zho",
}


def load_templates(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"error: {path} not found - run extract_qa_templates.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["templates"]


def load_items(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"error: {path} not found - run build_translation_jobs.py first")
    items = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            items[it["item_id"]] = it
    return items


def load_passages(data_dir: Path, splits: set[str], row_ids: set[str]) -> dict[str, str]:
    """Raw row id -> raw context, for exactly the rows referenced by row_ids
    (the *first* row_id per item, matching how build_translation_jobs.py's
    answer_sentence was extracted from that same row)."""
    passages = {}
    for split in sorted(splits):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"error: {path} not found (--tydi-data-dir)")
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in row_ids:
                    passages[r["id"]] = r["context"]
    return passages


def load_translations(dirs: list[Path]) -> dict[str, dict]:
    """job_id -> translated row, merged across dirs (e.g. one dir per model
    used, since no single model here reaches all 24 target languages). A
    later dir's row wins on a genuine id collision, which should not happen
    if each model covered a disjoint language set. Rows without item_id/field
    are silently skipped - not data/translation/translate_questions.py --input-dir
    output (e.g. its --input single-TSV mode output has no such columns)."""
    out = {}
    for d in dirs:
        for path in sorted(d.glob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if "item_id" in row and "field" in row:
                        out[row["id"]] = row
    return out


def render_prompt(template: dict, passage: str, question: str) -> str:
    sep = template["sep"]
    return sep.join([template["intro"], passage, template["q_prefix"] + question, template["instruction"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path,
                     default=ROOT / "translated/qa_translation_items.jsonl")
    ap.add_argument("--templates", type=Path,
                     default=ROOT / "qa_context_templates.json")
    ap.add_argument("--tydi-data-dir", type=Path,
                     default=ROOT / "built/tydiqa_answerability")
    ap.add_argument("--translated-dirs", type=Path, nargs="+", default=DEFAULT_TRANSLATED_DIRS,
                     help="One or more data/translation/translate_questions.py --input-dir "
                          "output dirs to merge (e.g. one per model used, since "
                          "reaching all 24 target languages needs hymt2 + madlad400).")
    ap.add_argument("--out", type=Path, default=ROOT / "built/qa_context_tydi_train.jsonl")
    ap.add_argument("--answer-form", choices=["span", "sentence", "adaptive"], default="sentence",
                     help="sentence (default) satisfies the instruction's \"one "
                          "sentence\" requirement more literally; span is shorter/"
                          "more precise but not phrased as a sentence. Falls back "
                          "to the other form if the chosen one is missing for a "
                          "given (item, target language), unless --no-fallback. "
                          "adaptive (ablation V6) takes the sentence when it is at most "
                          "--max-sentence-chars long and the span otherwise, so targets "
                          "stay close to the reference length without losing rows.")
    ap.add_argument("--max-sentence-chars", type=int, default=120,
                     help="adaptive only: longest sentence-form completion kept before "
                          "falling back to the span (T=120 in the paper).")
    ap.add_argument("--stats-out", type=Path, default=None,
                     help="optional JSON sidecar with row counts by answer form and "
                          "question language, so a build can be audited without the data.")
    ap.add_argument("--no-fallback", action="store_true",
                     help="drop a row instead of falling back to the other answer "
                          "form when --answer-form's chosen form is missing -- e.g. "
                          "with --answer-form sentence, this keeps only rows whose "
                          "completion is a genuine one-sentence answer, dropping the "
                          "cases (notably tha_Thai) where sentence extraction failed "
                          "and the row would otherwise carry a raw span instead.")
    ap.add_argument("--start-index", type=int, default=1)
    args = ap.parse_args()

    templates = load_templates(args.templates)
    items = load_items(args.items)
    print(f"Loaded {len(items):,} items, {len(templates)} question-language templates")

    translations = load_translations(args.translated_dirs)
    print(f"Loaded {len(translations):,} translated rows from {len(args.translated_dirs)} dir(s)")

    splits_needed = {s for it in items.values() for s in it["splits"].split(",")}
    row_ids_needed = {it["row_ids"][0] for it in items.values()}
    passages = load_passages(args.tydi_data_dir, splits_needed, row_ids_needed)
    print(f"Loaded {len(passages):,} passages for {len(row_ids_needed):,} needed rows")

    answer_field = "answer" if args.answer_form == "span" else "answer_sentence"
    fallback_field = "answer_sentence" if answer_field == "answer" else "answer"

    n = args.start_index
    written = 0
    skipped = collections.Counter()
    forms_used = collections.Counter()
    langs_written = collections.Counter()
    fallback_used = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f:
        for item in items.values():
            passage_raw = passages.get(item["row_ids"][0])
            if passage_raw is None:
                skipped["no_passage"] += len(item["targets"])
                continue
            passage = clean_text(passage_raw)
            src_iso3 = FLORES_TO_ISO3.get(item["src_lang"])

            for tgt in item["targets"]:
                tgt_iso3 = FLORES_TO_ISO3.get(tgt)
                if tgt_iso3 is None or src_iso3 is None:
                    skipped["unknown_language_code"] += 1
                    continue

                template = templates.get(tgt_iso3)
                if template is None:
                    skipped["no_template"] += 1
                    continue

                q_row = translations.get(f"{item['item_id']}__{tgt}__question")
                sent_row = translations.get(f"{item['item_id']}__{tgt}__answer_sentence")
                span_row = translations.get(f"{item['item_id']}__{tgt}__answer")
                a_row, form, used_fallback = None, None, False
                if args.answer_form == "adaptive":
                    # Ablation V6: a concise sentence is the best target (natural *and*
                    # near reference length); past the threshold the bare span is closer
                    # to what the references look like than a long sentence is. The
                    # threshold is measured on the cleaned text, which is what gets written.
                    sent = clean_text(sent_row["question"]) if sent_row else None
                    if sent is not None and len(sent) <= args.max_sentence_chars:
                        a_row, form = sent_row, "sentence"
                    elif span_row is not None:
                        a_row, form = span_row, "span"
                    elif sent_row is not None and not args.no_fallback:
                        a_row, form, used_fallback = sent_row, "sentence_over_limit", True
                else:
                    primary, other = (sent_row, span_row) if answer_field == "answer_sentence" else (span_row, sent_row)
                    a_row, form = primary, answer_field
                    if a_row is None and not args.no_fallback:
                        a_row, form = other, fallback_field
                        used_fallback = a_row is not None
                if q_row is None or a_row is None:
                    skipped["missing_translation"] += 1
                    continue
                forms_used[form] += 1
                langs_written[tgt_iso3] += 1

                # MT output isn't guaranteed clean either - same real-newline risk as the
                # raw passage (confirmed in practice: translated answers occasionally carry
                # one straight through), so run translated text through the same cleanup.
                question = clean_text(q_row["question"])
                completion = clean_text(a_row["question"])

                prompt = render_prompt(template, passage, question)
                out_f.write(json.dumps({
                    "id": f"qa-context_{n}_{tgt_iso3}_{src_iso3}",
                    "prompt": prompt,
                    "completion": completion,
                    "task": "qa-context",
                    "question_lang": tgt_iso3,
                }, ensure_ascii=False) + "\n")
                n += 1
                written += 1
                fallback_used += used_fallback

    print(f"\nWrote {written:,} training rows to {args.out}")
    print("By answer form: " + ", ".join(f"{k}={v:,}" for k, v in forms_used.most_common()))
    if fallback_used and args.answer_form != "adaptive":
        print(f"  ({fallback_used:,} used --answer-form {fallback_field} as a fallback)")
    if skipped:
        print("Skipped:")
        for k, v in skipped.most_common():
            print(f"  {k:22s} {v:,}")

    if args.stats_out is not None:
        stats = {
            "out": os.path.relpath(args.out, ROOT.parent),
            "rows": written,
            "answer_form": args.answer_form,
            "max_sentence_chars": args.max_sentence_chars if args.answer_form == "adaptive" else None,
            "no_fallback": args.no_fallback,
            "translated_dirs": [os.path.relpath(d, ROOT.parent) for d in args.translated_dirs],
            "by_answer_form": dict(forms_used.most_common()),
            "fallback_used": fallback_used,
            "skipped": dict(skipped.most_common()),
            "by_question_lang": dict(sorted(langs_written.items())),
        }
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote stats to {args.stats_out}")


if __name__ == "__main__":
    main()
