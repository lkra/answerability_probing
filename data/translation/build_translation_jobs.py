#!/usr/bin/env python3
"""Build the translation job spec for TyDiQA/cross (answerable rows only).

Each of the 64,257 TyDi items is assigned --targets-per-item of the 24 test question languages by a
balanced greedy that equalises targets and spreads (source, target) pairs. Per (item, target) three
jobs are emitted: the question, the gold span (answer) and the passage sentence containing it
(answer_sentence; absent where extraction fails, notably Thai). Passages are not translated. Job id
= <item_id>__<tgt_lang>__<field>; the text column is named `question` for every field because
translate_questions.py expects it.

Output (data/translated/): qa_translation_items.jsonl, to-translate/<tgt>.tsv (one per target
language, CSV-quoted), qa_translation_jobs_manifest.tsv; qa_translation_jobs.jsonl only with
--write-jobs-jsonl.
"""

import argparse
import collections
import contextlib
import csv
import json
import random
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# FLORES-200 code -> (ISO 639-1, ISO 639-3, plain-English name), for the 28 MIST
# languages. The English names are the strings `data/translation/translate_questions.py`
# accepts as `--languages`, so a manifest row can be passed to it verbatim.
LANGUAGES = {
    "arb_Arab": ("ar", "ara", "Arabic"),
    "ben_Beng": ("bn", "ben", "Bengali"),
    "bho_Deva": ("", "bho", "Bhojpuri"),
    "ces_Latn": ("cs", "ces", "Czech"),
    "ckb_Arab": ("", "ckb", "Central Kurdish"),
    "deu_Latn": ("de", "deu", "German"),
    "eng_Latn": ("en", "eng", "English"),
    "fin_Latn": ("fi", "fin", "Finnish"),
    "fra_Latn": ("fr", "fra", "French"),
    "hat_Latn": ("ht", "hat", "Haitian Creole"),
    "hin_Deva": ("hi", "hin", "Hindi"),
    "ind_Latn": ("id", "ind", "Indonesian"),
    "ita_Latn": ("it", "ita", "Italian"),
    "jpn_Jpan": ("ja", "jpn", "Japanese"),
    "kor_Hang": ("ko", "kor", "Korean"),
    "mar_Deva": ("mr", "mar", "Marathi"),
    "pes_Arab": ("fa", "pes", "Persian"),
    "por_Latn": ("pt", "por", "Portuguese"),
    "rus_Cyrl": ("ru", "rus", "Russian"),
    "slk_Latn": ("sk", "slk", "Slovak"),
    "spa_Latn": ("es", "spa", "Spanish"),
    "swh_Latn": ("sw", "swh", "Swahili"),
    "tel_Telu": ("te", "tel", "Telugu"),
    "tha_Thai": ("th", "tha", "Thai"),
    "tur_Latn": ("tr", "tur", "Turkish"),
    "vie_Latn": ("vi", "vie", "Vietnamese"),
    "yor_Latn": ("yo", "yor", "Yoruba"),
    "zho_Hans": ("zh", "zho", "Chinese"),
}

# `bho_Deva` and `ckb_Arab` have no ISO 639-1 code and carry "" there. Neither is
# ever a *source* language in this spec (TyDi has no Bhojpuri or Sorani, and the
# Belebele gap-fill is `eng`-sourced), and `iso639_1` is only ever read for the
# source, so the blank never reaches a consumer.


def lang_name(code: str) -> str:
    return LANGUAGES[code][2] if code in LANGUAGES else code


def rel(path: Path) -> str:
    """Path relative to the invocation directory when it is below it.

    The default output dir is derived from this file's location, so it is
    absolute; emitting that into the manifest would break the moment the repo is
    copied to a cluster scratch dir.
    """
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()

# The 24 question languages of the test set, each exactly 360 of 8,640 rows.
# Note this is *not* all 28 MIST languages: fra appears only as a passage
# language, and swh/tel/tha appear as neither. Translating into those four buys
# nothing for `qa-context`.
TEST_QUESTION_LANGS = [
    "arb_Arab", "ben_Beng", "bho_Deva", "ces_Latn", "ckb_Arab", "deu_Latn",
    "eng_Latn", "fin_Latn", "hat_Latn", "hin_Deva", "ind_Latn", "ita_Latn",
    "jpn_Jpan", "kor_Hang", "mar_Deva", "pes_Arab", "por_Latn", "rus_Cyrl",
    "slk_Latn", "spa_Latn", "tur_Latn", "vie_Latn", "yor_Latn", "zho_Hans",
]

# Test-set passage-language distribution, restricted to languages TyDi can
# supply. Used by `--source-mix test`. TyDi reaches only 8 of the 25 test
# passage languages; Belebele covers the other 17 natively, so this weighting
# shapes only the TyDi half of the eventual mix.
TEST_PASSAGE_WEIGHTS = {
    "eng_Latn": 2400, "arb_Arab": 600, "rus_Cyrl": 480, "ben_Beng": 240,
    "fin_Latn": 240, "ind_Latn": 240, "jpn_Jpan": 120, "kor_Hang": 120,
}

# Sentence terminators across the TyDi scripts. Thai has none of these -- it
# marks sentence breaks with spaces -- so tha_Thai sentence extraction is
# expected to fail and is reported rather than silently approximated.
TERMINATORS = ".!?。！？؟۔।\n"
_TERM_RE = re.compile(f"[{re.escape(TERMINATORS)}]")

# TyDi's contexts are scraped Wikipedia and still carry Parsoid debris. Two
# distinct problems hide in there, and they need opposite treatment:
#
#   * Citation markers like `[12]` are genuinely part of the text. Cosmetic --
#     strip them and keep the item.
#   * A *structural* fragment inside a gold answer (an HTML tag, a `data-parsoid`
#     attribute, a bare `<` or `>`) means the answer's character offsets were
#     computed against a marked-up string. Those spans are silently truncated
#     mid-word -- e.g. `='{"dsr":[255,265,3,3]}'>نواة في الخلية عُض` -- and no
#     amount of cleaning recovers the intended answer. Drop the item.
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_HTML_TAG_RE = re.compile(r"<[^<>]{0,300}?>")
_WS_RE = re.compile(r"[ \t ]+")
_STRUCTURAL_RE = re.compile(
    r"""<|>|data-parsoid|\{"dsr"|\[\[|\]\]|\{\{|\}\}|=\s*['"]""")


def clean_text(text: str) -> str:
    """Cosmetic cleanup: drop HTML tags and citation markers, tidy whitespace.

    The LoRA build step should apply this to passages too, so that what the
    model reads at train time matches what it is trained to answer from.
    """
    if not text:
        return text
    text = _HTML_TAG_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def is_structurally_broken(text: str) -> bool:
    """True if markup survives cleaning, i.e. the offsets cannot be trusted."""
    return bool(text) and bool(_STRUCTURAL_RE.search(clean_text(text)))


def containing_sentence(context: str, span: str) -> str | None:
    """The sentence of `context` that holds `span`, or None if not isolable.

    Returns None when no terminator is found on either side, which would
    otherwise hand back the whole passage dressed up as a sentence.
    """
    i = context.find(span)
    if i < 0:
        return None
    left = max((m.end() for m in _TERM_RE.finditer(context, 0, i)), default=None)
    m = _TERM_RE.search(context, i + len(span))
    right = m.end() if m else None
    if left is None and right is None:
        return None
    sent = context[(left or 0):(right or len(context))].strip()
    return sent or None


def load_tydi_items(data_dir: Path, splits: list[str],
                    clean: bool, drop_broken: bool) -> list[dict]:
    """Unique (lang, question, answer) items over the answerable TyDi rows (every question appears
    twice and recurs across paragraphs). The answer sentence is extracted from the raw context before
    cleaning, then cleaned.
    """
    items: dict[tuple, dict] = {}
    n_rows = 0
    dropped = collections.Counter()
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"error: {path} not found")
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("answerable") != 1:
                    continue
                n_rows += 1
                question, answer = r["question"], r["output"]
                sentence = containing_sentence(r["context"], answer)
                if drop_broken and (is_structurally_broken(answer)
                                    or is_structurally_broken(question)):
                    dropped[r["lang_code"]] += 1
                    continue
                if clean:
                    question = clean_text(question)
                    answer = clean_text(answer)
                    sentence = clean_text(sentence) if sentence else None
                # A sentence is optional, so residual markup costs only the
                # sentence form rather than the whole item.
                if drop_broken and sentence and is_structurally_broken(sentence):
                    sentence = None
                if not question or not answer:
                    dropped[r["lang_code"]] += 1
                    continue
                key = (r["lang_code"], question, answer)
                it = items.get(key)
                if it is None:
                    it = items[key] = {
                        "src_lang": r["lang_code"],
                        "question": question,
                        "answer": answer,
                        "answer_sentence": sentence,
                        "splits": set(),
                        "row_ids": [],
                        "in_organizer_sample": False,
                    }
                it["splits"].add(split)
                it["row_ids"].append(r["id"])
                it["in_organizer_sample"] |= bool(r.get("in_organizer_sample"))
    out = []
    for i, (_k, it) in enumerate(sorted(items.items())):
        it = dict(it)
        it["item_id"] = f"tydi_{i:06d}"
        it["splits"] = ",".join(sorted(it["splits"]))
        out.append(it)
    print(f"TyDi: {n_rows:,} answerable rows -> {len(out):,} unique "
          f"(lang, question, answer) items")
    if dropped:
        print(f"  dropped {sum(dropped.values()):,} rows with unrecoverable "
              f"markup in the question or gold span "
              f"({sum(dropped.values()) / n_rows:.2%}):")
        for lang, n in dropped.most_common():
            print(f"    {lang:10s} {n:5,d}")
    return out


def load_belebele_bho(belebele_dir: Path, src: str = "eng_Latn",
                      tgt: str = "bho_Deva") -> list[dict]:
    """Texts to add Bhojpuri, absent from both corpora, as a passage language via BELEBELE, translated
    from English (the pivot the other BELEBELE languages were translated from).
    """
    path = belebele_dir / f"{src}.jsonl"
    if not path.exists():
        raise SystemExit(f"error: {path} not found (need --belebele-dir)")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    texts: dict[tuple[str, str], str] = {}
    for r in rows:
        texts[("passage", r["flores_passage"])] = r["flores_passage"]
        texts[("question", r["question"])] = r["question"]
        for i in (1, 2, 3, 4):
            texts[("option", r[f"mc_answer{i}"])] = r[f"mc_answer{i}"]
    out = []
    for i, ((field, _t), text) in enumerate(sorted(texts.items())):
        out.append({
            "item_id": f"belebele_{field}_{i:05d}",
            "field": field,
            "src_lang": src,
            "targets": [tgt],
            "text": text,
        })
    n = collections.Counter(i["field"] for i in out)
    print(f"Belebele {src} -> {tgt}: {len(out):,} unique texts "
          f"({n['passage']:,} passages, {n['question']:,} questions, "
          f"{n['option']:,} options)")
    return out


def apply_source_mix(items: list[dict], mix: str, cap: int | None,
                     rng: random.Random) -> list[dict]:
    """Subsample per source language. TyDi is skewed ~8:1 (arb 15.8k, kor 1.9k)."""
    by_lang = collections.defaultdict(list)
    for it in items:
        by_lang[it["src_lang"]].append(it)

    if mix == "natural":
        quota = {k: len(v) for k, v in by_lang.items()}
    elif mix == "uniform":
        n = min(len(v) for v in by_lang.values())
        quota = {k: n for k in by_lang}
    elif mix == "test":
        # Scale the test passage-language proportions up until some language
        # runs out. Languages absent from the test set get nothing.
        w = {k: TEST_PASSAGE_WEIGHTS.get(k, 0) for k in by_lang}
        scale = min((len(by_lang[k]) / w[k] for k in w if w[k] > 0), default=0)
        quota = {k: int(w[k] * scale) for k in by_lang}
    else:
        raise SystemExit(f"error: unknown --source-mix {mix!r}")

    if cap is not None:
        quota = {k: min(v, cap) for k, v in quota.items()}

    kept = []
    for lang in sorted(by_lang):
        pool = sorted(by_lang[lang], key=lambda x: x["item_id"])
        q = quota.get(lang, 0)
        if q < len(pool):
            pool = rng.sample(pool, q)
        kept.extend(pool)
    kept.sort(key=lambda x: x["item_id"])
    return kept


def assign_targets(items: list[dict], targets: list[str], k: int,
                   excluded: set[tuple[str, str]], rng: random.Random) -> None:
    """Balanced greedy assignment of k target languages per item, in place.

    Primary key is the target's current load, so target languages end up
    equalised -- the axis the test set balances. Secondary key is the (source,
    target) pair count, which spreads the pair matrix instead of letting a large
    source language monopolise whichever target is momentarily least loaded.
    """
    load = collections.Counter({t: 0 for t in targets})
    pair = collections.Counter()
    order = list(items)
    rng.shuffle(order)
    for it in order:
        src = it["src_lang"]
        eligible = [t for t in targets
                    if t != src and (src, t) not in excluded]
        if not eligible:
            it["targets"] = []
            continue
        chosen = []
        for _ in range(min(k, len(eligible))):
            t = min((t for t in eligible if t not in chosen),
                    key=lambda t: (load[t], pair[(src, t)], t))
            chosen.append(t)
            load[t] += 1
            pair[(src, t)] += 1
        it["targets"] = sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path,
                    default=SCRIPT_DIR.parent / "built/tydiqa_answerability")
    ap.add_argument("--splits", default="train,validation",
                    help="Default is both. The earlier run exported only "
                         "validation, which capped item diversity at 6,665.")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR.parent / "translated")
    ap.add_argument("--prefix", default="qa_translation")
    ap.add_argument("--targets-per-item", type=int, default=2)
    ap.add_argument("--source-mix", choices=["natural", "uniform", "test"],
                    default="natural")
    ap.add_argument("--per-source-cap", type=int, default=None)
    ap.add_argument("--answer-forms", default="span,sentence",
                    help="Comma-separated subset of: span, sentence.")
    ap.add_argument("--targets", default=",".join(TEST_QUESTION_LANGS))
    ap.add_argument("--exclude-pairs", default="",
                    help="e.g. 'arb_Arab>pes_Arab'. Empty by default -- see the "
                         "warning printed at the end.")
    ap.add_argument("--drop-organizer-sample", action="store_true",
                    help="Drop items appearing in the organizers' sample mix.")
    ap.add_argument("--belebele-dir", type=Path,
                    default=SCRIPT_DIR.parent / "upstream/belebele")
    ap.add_argument("--no-belebele-bho", action="store_true",
                    help="Skip the Bhojpuri gap-fill. Without it, bho has no "
                         "passage-language coverage from any source.")
    ap.add_argument("--write-jobs-jsonl", action="store_true",
                    help="Also write the combined <prefix>_jobs.jsonl. It "
                         "duplicates the per-language TSVs, which are what the "
                         "runner reads, and costs ~130 MB.")
    ap.add_argument("--no-clean", action="store_true",
                    help="Keep raw TyDi text (HTML tags, [12] citation markers).")
    ap.add_argument("--keep-broken", action="store_true",
                    help="Keep items whose gold span carries structural markup, "
                         "which means its offsets are wrong and the span is "
                         "usually truncated mid-word.")
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    forms = [f.strip() for f in args.answer_forms.split(",") if f.strip()]
    bad_forms = set(forms) - {"span", "sentence"}
    if bad_forms:
        raise SystemExit(f"error: unknown --answer-forms {sorted(bad_forms)}")
    unknown = [t for t in targets if t not in LANGUAGES]
    if unknown:
        raise SystemExit(f"error: not a MIST language code: {unknown}")

    excluded = set()
    for p in args.exclude_pairs.split(","):
        p = p.strip()
        if p:
            s, _, t = p.partition(">")
            excluded.add((s.strip(), t.strip()))

    items = load_tydi_items(args.data_dir, splits,
                            clean=not args.no_clean,
                            drop_broken=not args.keep_broken)
    if args.drop_organizer_sample:
        before = len(items)
        items = [i for i in items if not i["in_organizer_sample"]]
        print(f"dropped {before - len(items):,} items in the organizer sample")

    sent_ok = collections.Counter()
    sent_tot = collections.Counter()
    for it in items:
        sent_tot[it["src_lang"]] += 1
        sent_ok[it["src_lang"]] += it["answer_sentence"] is not None

    items = apply_source_mix(items, args.source_mix, args.per_source_cap, rng)
    print(f"after --source-mix {args.source_mix}"
          + (f" --per-source-cap {args.per_source_cap}" if args.per_source_cap else "")
          + f": {len(items):,} items")

    assign_targets(items, targets, args.targets_per_item, excluded, rng)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.out_dir / f"{args.prefix}_items.jsonl"
    jobs_path = args.out_dir / f"{args.prefix}_jobs.jsonl"
    tsv_dir = args.out_dir / f"{args.prefix}_jobs"
    manifest_path = args.out_dir / f"{args.prefix}_jobs_manifest.tsv"
    tsv_dir.mkdir(parents=True, exist_ok=True)
    # A rerun with fewer --targets would otherwise leave the dropped languages'
    # files behind, and a stale spec next to a fresh one is worse than neither.
    for stale in tsv_dir.glob("*.tsv"):
        stale.unlink()

    bho_items = ([] if args.no_belebele_bho
                 else load_belebele_bho(args.belebele_dir))
    if bho_items:
        bho_path = args.out_dir / f"{args.prefix}_bho_items.jsonl"
        with open(bho_path, "w", encoding="utf-8") as f:
            for it in bho_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(items_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "item_id": it["item_id"], "src_lang": it["src_lang"],
                "splits": it["splits"], "question": it["question"],
                "answer": it["answer"], "answer_sentence": it["answer_sentence"],
                "targets": it["targets"], "row_ids": it["row_ids"],
                "in_organizer_sample": it["in_organizer_sample"],
            }, ensure_ascii=False) + "\n")

    n_jobs = 0
    chars = 0
    per_field = collections.Counter()
    per_target = collections.Counter()
    per_pair = collections.Counter()
    field_texts = [("question", "question")]
    if "span" in forms:
        field_texts.append(("answer", "answer"))
    if "sentence" in forms:
        field_texts.append(("answer_sentence", "answer_sentence"))

    # One TSV per target language, because `translate_questions.py` translates a
    # whole input file into every `--languages` it is handed; splitting here is
    # what preserves the per-item target assignment made above. Column names are
    # that script's, not this one's -- notably `qid` and `question`, the latter
    # carrying answer text too.
    TSV_HEADER = ["qid", "item_id", "field", "src_lang", "src_language",
                  "iso639_1", "iso639_3", "tgt_lang", "tgt_language", "question"]
    per_target_jobs = collections.Counter()

    with contextlib.ExitStack() as stack:
        # The combined jsonl is a strict duplicate of the per-language TSVs (same
        # ten fields, one row per job) and no runner reads it -- the TSVs are the
        # input format. Off by default so a rebuild does not lay down another
        # 130 MB; --write-jobs-jsonl brings it back for ad-hoc analysis.
        jf = (stack.enter_context(open(jobs_path, "w", encoding="utf-8"))
              if args.write_jobs_jsonl else None)
        writers = {}

        def writer_for(tgt: str):
            w = writers.get(tgt)
            if w is None:
                f = stack.enter_context(
                    open(tsv_dir / f"{tgt}.tsv", "w", encoding="utf-8", newline=""))
                # Standard quoting, so a quote/tab/newline inside a passage is
                # escaped and survives the round-trip instead of costing the job.
                w = writers[tgt] = csv.writer(f, delimiter="\t", lineterminator="\n")
                w.writerow(TSV_HEADER)
            return w

        def emit(item_id, field, src_lang, tgt, text):
            nonlocal n_jobs, chars
            job_id = f"{item_id}__{tgt}__{field}"
            iso1, iso3, _name = LANGUAGES[src_lang]
            if jf is not None:
                jf.write(json.dumps({
                    "job_id": job_id, "item_id": item_id, "field": field,
                    "src_lang": src_lang, "src_language": lang_name(src_lang),
                    "iso639_1": iso1, "iso639_3": iso3,
                    "tgt_lang": tgt, "tgt_language": lang_name(tgt),
                    "text": text,
                }, ensure_ascii=False) + "\n")
            n_jobs += 1
            chars += len(text)
            per_field[field] += 1
            per_target_jobs[tgt] += 1
            if not field.startswith("belebele_"):
                per_target[tgt] += 1
                per_pair[(src_lang, tgt)] += 1
            writer_for(tgt).writerow(
                [job_id, item_id, field, src_lang, lang_name(src_lang),
                 iso1, iso3, tgt, lang_name(tgt), text])

        for it in items:
            for tgt in it["targets"]:
                for field, key in field_texts:
                    if it[key]:
                        emit(it["item_id"], field, it["src_lang"], tgt, it[key])

        for it in bho_items:
            for tgt in it["targets"]:
                emit(it["item_id"], f"belebele_{it['field']}", it["src_lang"],
                     tgt, it["text"])

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["tgt_lang", "tgt_language", "file", "n_jobs"])
        for tgt in sorted(per_target_jobs):
            w.writerow([tgt, lang_name(tgt), rel(tsv_dir / f"{tgt}.tsv"),
                        per_target_jobs[tgt]])

    # ---- manifest -------------------------------------------------------
    print(f"\n{'=' * 72}\nREALISED MIX\n{'=' * 72}")
    src_counts = collections.Counter(i["src_lang"] for i in items)
    print(f"\nSource (= passage) languages, {len(src_counts)} of the 25 test "
          f"passage languages TyDi can reach:")
    print(f"  {'lang':10s} {'items':>8s}  {'sentence extraction':>20s}")
    for lang in sorted(src_counts, key=lambda x: -src_counts[x]):
        rate = sent_ok[lang] / sent_tot[lang] if sent_tot[lang] else 0.0
        print(f"  {lang:10s} {src_counts[lang]:8,d}  {rate:19.1%}")

    print(f"\nTarget (= question) languages, {len(per_target)} of 24, TyDi "
          f"family only (test set gives each exactly 1/24):")
    lo, hi = min(per_target.values()), max(per_target.values())
    for lang in sorted(per_target):
        print(f"  {lang:10s} {per_target[lang]:8,d}")
    print(f"  spread: min {lo:,} max {hi:,} "
          f"({(hi - lo) / hi if hi else 0:.2%} imbalance)")

    print(f"\nLanguage pairs used: {len(per_pair)} "
          f"of {len(src_counts) * len(targets)} possible")
    print(f"\nJobs by field:")
    for k, v in per_field.most_common():
        print(f"  {k:16s} {v:9,d}")
    print(f"\nTOTAL JOBS  {n_jobs:,}")
    print(f"source chars {chars:,}  (mean {chars / max(n_jobs, 1):.1f}/job)")
    print(f"\nWrote {rel(items_path)}")
    if args.write_jobs_jsonl:
        print(f"Wrote {rel(jobs_path)}")
    print(f"Wrote {rel(tsv_dir)}/<tgt_lang>.tsv   ({len(per_target_jobs)} files)")
    print(f"Wrote {rel(manifest_path)}")

    print(f"\nRun the jobs, one target language per invocation -- the input\n"
          f"file is already restricted to that target, so --languages must\n"
          f"name exactly one:\n\n"
          f"  tail -n +2 {rel(manifest_path)} | while IFS=$'\\t' read -r tgt name file n; do\n"
          f"      python data/translation/translate_questions.py \\\n"
          f"          --input \"$file\" --languages \"$name\" \\\n"
          f"          --output-dir {rel(args.out_dir)}/translated_qa\n"
          f"  done\n\n"
          f"Language names must be ones the chosen model's list in\n"
          f"data/translation/translate_questions.py accepts. Its madlad400 list reaches\n"
          f"all 24 targets; its hymt2 list is 6 short (bho, ckb, fin, hat, slk,\n"
          f"yor), so an Hy-MT2 run needs a second model for those.")

    if ("arb_Arab", "pes_Arab") in per_pair:
        print(f"\n{'!' * 72}")
        print("arb_Arab -> pes_Arab is INCLUDED. The 59.5% first-person failure\n"
              "in data/translation/README.md was found in Hy-MT2-7B-FP8; whether the\n"
              "model you translate with shares it is unknown. Spot-check that\n"
              "pair after the run before trusting it, or re-run with\n"
              "  --exclude-pairs 'arb_Arab>pes_Arab'")
        print("!" * 72)


if __name__ == "__main__":
    main()
