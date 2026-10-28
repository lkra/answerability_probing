"""Row loaders shared by extraction and generation: TyDi splits, the translated cross-lingual set,
UDHR, BELEBELE (option-dependent questions dropped) and the test set with silver labels. Every row
carries passage_lang and question_lang; _meta is what the cache's meta.jsonl records.
"""

import collections
import json
import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "built" / "tydiqa_answerability"
TRANSLATION_DIR = REPO_ROOT / "data" / "translated"
DEFAULT_TRANSLATIONS = TRANSLATION_DIR / "Hy-MT2-7B-FP8.jsonl"
DEFAULT_SPEC = TRANSLATION_DIR / "questions_to_translate.tsv"
DEFAULT_BELEBELE = (REPO_ROOT / "data" / "built" /
                    "belebele_answerability_delete_strict.jsonl")

# Belebele questions that cannot be answered from the passage alone because they refer to the four
# options -- which the `tydi` rendering deliberately drops. Kept byte-identical to
# `build_belebele.OPTION_DEPENDENT`; see `drop_option_dependent` for why they have to go.
OPTION_DEPENDENT = [
    r"which of the following", r"which of these", r"all of the following",
    r"\bof the following\b", r"none of the (above|following)",
    r"which statement", r"which one",
]


def is_option_dependent(question: str) -> bool:
    return any(re.search(p, question, re.I) for p in OPTION_DEPENDENT)


def drop_option_dependent(rows, english_by_item, verbose=True):
    """Drop items whose question refers to the answer options, which the `tydi` rendering hides (20% of
    belebele8 items, 6% of delete_strict). Matched on the item's English question so the same items
    go in every language.
    """
    bad = {item for item, q in english_by_item.items() if is_option_dependent(q)}
    kept = [r for r in rows if r["_meta"].get("item_id") not in bad]
    if verbose:
        n = len(rows) - len(kept)
        print(f"option-dependent filter: dropped {n} rows from {len(bad)} of "
              f"{len(english_by_item)} items ({100 * n / max(len(rows), 1):.0f}% of rows)")
    return kept


def english_questions_by_item(path):
    """item_id -> the English question, for `drop_option_dependent`."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("question_lang") == "eng_Latn":
                parts = o["input"].split("\n")
                if len(parts) == 6:
                    out.setdefault(o["item_id"], parts[1])
    return out


DEFAULT_TESTS = REPO_ROOT / "data" / "test" / "tests.jsonl"
DEFAULT_SILVER = REPO_ROOT / "data" / "silver" / "silver_test_labels_final.jsonl"


def load_rows_mist_test(tests_path=None, templates_path=None, silver_path=None,
                        languages=None, limit_per_lang=0, seed=0, min_confidence=None):
    """The 8,640 qa-context test rows split back into passage and question against the recovered
    templates; a prompt that does not match its template is skipped (all 8,640 split cleanly). Labels
    are the silver labels, attached as `answerable`; min_confidence="high" drops the 912 medium and
    192 low-confidence rows.
    """
    tests_path = Path(tests_path or DEFAULT_TESTS)
    templates_path = Path(templates_path or
                          REPO_ROOT / "data" / "qa_context_templates.json")
    silver_path = Path(silver_path or DEFAULT_SILVER)

    spec = json.loads(templates_path.read_text(encoding="utf-8"))
    templates, sep = spec["templates"], spec["sep"]
    silver = {}
    with open(silver_path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            silver[s["id"]] = s

    order = {"low": 0, "medium": 1, "high": 2}
    rows, skipped = [], collections.Counter()
    with open(tests_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("task") != "qa-context":
                continue
            s = silver.get(o["id"])
            if s is None:
                skipped["no silver label"] += 1
                continue
            if min_confidence and order[s["confidence"]] < order[min_confidence]:
                skipped[f"confidence < {min_confidence}"] += 1
                continue
            qlang = o["question_lang"]
            if languages and qlang not in languages:
                continue
            t = templates.get(qlang)
            if t is None:
                skipped["no qa-context template"] += 1
                continue
            parts = o["prompt"].split(sep)
            if (len(parts) != 4 or parts[0] != t["intro"]
                    or not parts[2].startswith(t["q_prefix"]) or parts[3] != t["instruction"]):
                # Not recoverable by the documented layout. Refuse rather than guess: a
                # mis-split silently moves instruction text into the passage and the probe
                # then scores a prompt nobody designed.
                skipped["prompt does not match its template"] += 1
                continue
            passage, question = parts[1], parts[2][len(t["q_prefix"]):]
            rows.append({
                "id": o["id"],
                "context": passage,
                "question": question,
                "lang_code": qlang,
                "answerable": int(bool(s["answerable"])),
                "_meta": {
                    "id": o["id"], "item_id": s.get("item_id"),
                    "lang_code": qlang,
                    "passage_lang": s["context_lang"], "question_lang": qlang,
                    "answerable": int(bool(s["answerable"])),
                    "label_source": "silver",
                    "confidence": s["confidence"],
                    "split": "mist-test", "question": question,
                    "in_organizer_sample": False,   # this *is* the organizer set
                    "passage_in_other_split": False,
                },
            })
    if skipped:
        print("mist-test rows skipped: "
              + ", ".join(f"{k} ({v:,})" for k, v in sorted(skipped.items())))
    if limit_per_lang:
        rows = stratified_subsample(
            rows, lambda o: (o["_meta"]["passage_lang"], o["_meta"]["question_lang"]),
            limit_per_lang, seed)
    return rows


def load_rows_udhr(path=None, templates_path=None, languages=None, limit_per_lang=0, seed=0):
    """data/udhr/data/udhr_qa_train.jsonl: 24 question x 25 context languages, legal register, 20.3%
    unanswerable, items disjoint from the test set. Prompts are split back into passage and question
    against the same templates (17,280 of 17,280 split cleanly).
    """
    path = Path(path or REPO_ROOT / "data" / "udhr" / "data" / "udhr_qa_train.jsonl")
    templates_path = Path(templates_path or
                          REPO_ROOT / "data" / "qa_context_templates.json")
    spec = json.loads(templates_path.read_text(encoding="utf-8"))
    templates, sep = spec["templates"], spec["sep"]

    rows, skipped = [], collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qlang = o["question_lang"]
            if languages and qlang not in languages:
                continue
            t = templates.get(qlang)
            if t is None:
                skipped["no qa-context template"] += 1
                continue
            parts = o["prompt"].split(sep)
            if (len(parts) != 4 or parts[0] != t["intro"]
                    or not parts[2].startswith(t["q_prefix"]) or parts[3] != t["instruction"]):
                # Same refusal as mist-test: a mis-split moves instruction text into the
                # passage and the probe scores a prompt nobody designed.
                skipped["prompt does not match its template"] += 1
                continue
            passage, question = parts[1], parts[2][len(t["q_prefix"]):]
            rows.append({
                "id": o["id"],
                "context": passage,
                "question": question,
                "lang_code": qlang,
                "answerable": int(bool(o["answerable"])),
                "_meta": {
                    "id": o["id"], "item_id": o.get("item_id"),
                    "lang_code": qlang,
                    "passage_lang": o["context_lang"], "question_lang": qlang,
                    "answerable": int(bool(o["answerable"])),
                    "label_source": "udhr_qa",
                    "origin": o.get("origin"), "question_type": o.get("question_type"),
                    "passage_id": o.get("passage_id"),
                    "split": "udhr", "question": question,
                    "in_organizer_sample": False,
                    "passage_in_other_split": False,
                },
            })
    if skipped:
        print("udhr rows skipped: "
              + ", ".join(f"{k} ({v:,})" for k, v in sorted(skipped.items())))
    if limit_per_lang:
        rows = stratified_subsample(
            rows, lambda o: (o["_meta"]["passage_lang"], o["_meta"]["question_lang"]),
            limit_per_lang, seed)
    return rows


def stratified_subsample(rows, key_of, limit, seed=0):
    """Keep ~`limit` rows per group, label-balanced within each group.

    Stratified by label, not a prefix: the TyDi build emits its rows grouped by label, so taking
    the first N per language yields a single-class pilot on which every probe and every baseline
    scores 1.000 and AUROC is undefined.
    """
    rng = np.random.default_rng(seed)
    buckets = {}
    for i, o in enumerate(rows):
        buckets.setdefault((key_of(o), o["answerable"]), []).append(i)
    keep = []
    # sorted(), not a set: set iteration order over strings varies with PYTHONHASHSEED, so drawing
    # from `rng` in set order makes the "seeded" sample differ between processes. That silently
    # breaks any comparison between two pilot runs that are supposed to be identical.
    for key in sorted({key_of(o) for o in rows}):
        labels = [l for l in (0, 1) if (key, l) in buckets]
        per_label = max(limit // max(len(labels), 1), 1)
        for l in labels:
            pool = buckets[(key, l)]
            keep.extend(rng.choice(pool, size=min(per_label, len(pool)), replace=False).tolist())
    return [rows[i] for i in sorted(keep)]


def load_rows(data_dir, split, languages, limit_per_lang, seed=0):
    """Monolingual TyDi rows: passage and question in the same language."""
    rows = []
    with open(Path(data_dir) / f"{split}.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if languages and o["lang_code"] not in languages:
                continue
            o["_meta"] = {
                "id": o["id"], "lang_code": o["lang_code"], "passage_lang": o["lang_code"],
                "question_lang": o["lang_code"], "answerable": o["answerable"],
                "split": o["split"], "question": o["question"],
                "in_organizer_sample": o["in_organizer_sample"],
                "passage_in_other_split": o["passage_in_other_split"],
            }
            rows.append(o)
    if limit_per_lang:
        rows = stratified_subsample(rows, lambda o: o["lang_code"], limit_per_lang, seed)
    return rows


def load_rows_translation(data_dir, translations, spec_path, languages, limit_per_lang, seed=0):
    """Cross-lingual rows: a TyDi validation passage with its question machine-translated. qid -> spec ->
    (src_lang, question) -> the two TyDi validation rows with that question (one answerable, one not),
    so the label stays balanced. `languages` filters on the question language.
    """
    # The export spec is the TSV `export_questions_for_translation.py` writes (qid, src_lang,
    # src_language, iso639_1, iso639_3, question); a JSONL spec with the same fields also works.
    spec = {}
    with open(spec_path, encoding="utf-8", newline="") as f:
        if str(spec_path).endswith(".tsv"):
            import csv
            for s in csv.DictReader(f, delimiter="\t"):
                spec[s["qid"]] = s
        else:
            for line in f:
                s = json.loads(line)
                spec[s["qid"]] = s

    by_question = {}
    with open(Path(data_dir) / "validation.jsonl") as f:
        for line in f:
            o = json.loads(line)
            by_question.setdefault((o["lang_code"], o["question"]), []).append(o)

    rows, unmatched = [], 0
    with open(translations) as f:
        for line in f:
            t = json.loads(line)
            if languages and t["tgt_lang"] not in languages:
                continue
            s = spec.get(t["qid"])
            hits = by_question.get((s["src_lang"], s["question"]), []) if s else []
            if not hits:
                unmatched += 1
                continue
            for o in hits:
                # The passage is the TyDi one, untouched; only the question is replaced. Keeping
                # `context` under its TyDi name lets build_prompt stay source-agnostic.
                rows.append({
                    "id": f"xl_{o['id']}_{t['tgt_lang']}",
                    "context": o["context"],
                    "question": t["question"],
                    "lang_code": t["tgt_lang"],
                    "answerable": o["answerable"],
                    "_meta": {
                        "id": f"xl_{o['id']}_{t['tgt_lang']}",
                        "lang_code": t["tgt_lang"],          # = question_lang, for grouping code
                        "passage_lang": o["lang_code"],
                        "question_lang": t["tgt_lang"],
                        "answerable": o["answerable"],
                        "split": "validation",
                        "question": t["question"],
                        "src_question": s["question"],
                        "qid": t["qid"],
                        "tydi_id": o["id"],
                        "mt_model": t.get("model"),
                        "mt_flags": t.get("flags", []),
                        "in_organizer_sample": o["in_organizer_sample"],
                        "passage_in_other_split": o["passage_in_other_split"],
                    },
                })
    if unmatched:
        print(f"WARNING: {unmatched} translation rows had no matching TyDi validation question")
    if limit_per_lang:
        rows = stratified_subsample(
            rows, lambda o: (o["_meta"]["passage_lang"], o["_meta"]["question_lang"]),
            limit_per_lang, seed)
    return rows


def load_rows_belebele(path, fmt, languages, limit_per_lang, seed=0, subset=None,
                       drop_option_dep=True):
    """BELEBELE rows, 26 passage x 27 question languages, constructed negatives. fmt="tydi" (default)
    re-renders the six-line native input as passage, blank line, question without the options,
    matching the probe's training format; "native" keeps BELEBELE's rendering. drop_option_dep
    applies drop_option_dependent (a no-op under native).
    """
    raw = [json.loads(l) for l in open(path)]

    # `_perlang` keeps each item in whichever languages segment it the same way as English, so the
    # item sets differ per language and a raw cross-lingual comparison partly measures *which*
    # items each language was scored on. Restricting to the intersection over the selected passage
    # languages is what makes the comparison mean something -- and on a subset of languages that
    # intersection is far larger than the all-language one (`_strict`, 16 items). For the eight
    # languages the low-resource finding rests on it is 160.
    if subset:
        by_lang = {}
        for o in raw:
            by_lang.setdefault(o["passage_lang"], set()).add(o["item_id"])
        present = [l for l in sorted(subset) if l in by_lang]
        keep_items = set.intersection(*(by_lang[l] for l in present)) if present else set()
        print(f"restricting to {len(present)} languages and their common "
              f"{len(keep_items)} items (from {len(raw)} rows)")
        raw = [o for o in raw
               if o["item_id"] in keep_items
               and o["passage_lang"] in subset and o["question_lang"] in subset]

    rows = []
    for o in raw:
            if languages and o["question_lang"] not in languages:
                continue
            parts = o["input"].split("\n")
            if len(parts) != 6:
                continue
            passage, question, options = parts[0], parts[1], parts[2:]
            if fmt == "native":
                question = "\n".join([question] + options)
            rows.append({
                "id": o["id"],
                "context": passage,
                "question": question,
                "lang_code": o["question_lang"],
                "answerable": o["answerable"],
                "_meta": {
                    "id": o["id"], "item_id": o["item_id"],
                    "lang_code": o["question_lang"],
                    "passage_lang": o["passage_lang"], "question_lang": o["question_lang"],
                    "answerable": o["answerable"], "split": "belebele",
                    "question": question,
                    "answer_in_distractor": o.get("answer_in_distractor"),
                    "in_organizer_sample": False,   # Belebele is not in the organizer mix
                    "passage_in_other_split": False,
                },
            })
    if drop_option_dep and fmt == "tydi":
        rows = drop_option_dependent(rows, english_questions_by_item(path))
    if limit_per_lang:
        rows = stratified_subsample(
            rows, lambda o: (o["_meta"]["passage_lang"], o["_meta"]["question_lang"]),
            limit_per_lang, seed)
    return rows
