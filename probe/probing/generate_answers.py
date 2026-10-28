#!/usr/bin/env python3
"""Generate answers from the base model under the organisers' qa-context prompt (template chosen by
question language), greedy, thinking disabled, with the same passage truncation as extraction.
--source tydi | translation | belebele; --ids-from restricts and orders rows; resumable.

Output rows: id, lang_code, question_lang, question_lang_flores, passage_lang, answerable,
truncated, generation
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch

from _device import pick_device
from _loading import load_generator, load_tokenizer
from _prompts import build_prompt, load_templates
from _sources import (DEFAULT_BELEBELE, DEFAULT_DATA, DEFAULT_SPEC, DEFAULT_TRANSLATIONS,
                      load_rows, load_rows_belebele, load_rows_translation, stratified_subsample)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
MAX_LEN_DEFAULT = 1200          # matches extract_activations.MAX_LEN_BY_STYLE["qa-context"]


def select_rows(rows, templates, per_language, seed, balanced):
    """Drop rows with no template for their question language (the template keys on the question
    language, which on cross-lingual rows differs from lang_code), then sample per_language rows per
    language, label-balanced if `balanced`.
    """
    have = set(templates)
    kept, skipped = [], collections.Counter()
    for r in rows:
        qlang = r["_meta"]["question_lang"].split("_")[0]
        if qlang not in have:
            skipped[r["_meta"]["question_lang"]] += 1
            continue
        r["question_lang"] = qlang
        kept.append(r)
    if skipped:
        print("no qa-context template for these question languages, skipped: "
              + ", ".join(f"{k} ({v:,})" for k, v in sorted(skipped.items())))

    if per_language:
        # Group by the (passage, question) language *pair*, not by one language: cross-lingually a
        # per-language cap would take N rows spread over 17 passage languages for one question
        # language, leaving some cells with a handful of rows and no usable per-cell rate.
        key = lambda r: (r["_meta"]["passage_lang"], r["_meta"]["question_lang"])
        if balanced:
            kept = stratified_subsample(kept, key, per_language, seed)
        else:
            rng = random.Random(seed)
            by = collections.defaultdict(list)
            for r in kept:
                by[key(r)].append(r)
            out = []
            for k in sorted(by):
                pool = sorted(by[k], key=lambda x: x["id"])
                out.extend(pool if len(pool) <= per_language
                           else rng.sample(pool, per_language))
            kept = out
        kept = sorted(kept, key=lambda x: x["id"])
    return kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="tydi", choices=["tydi", "translation", "belebele"],
                    help="tydi = monolingual. translation = cross-lingual (TyDi validation passage "
                         "+ machine-translated question). belebele = constructed negatives, both "
                         "passage and question language varying. The last two are the point: they "
                         "are the only sets where the instruction language and the passage "
                         "language disagree.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", default="validation", choices=["train", "validation"],
                    help="--source tydi only")
    ap.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--belebele", type=Path, default=DEFAULT_BELEBELE)
    ap.add_argument("--belebele-languages", default=None,
                    help="comma-separated; kept as BOTH passage and question language, then "
                         "restricted to their common item_ids")
    ap.add_argument("--belebele-format", default="tydi", choices=["tydi", "native"])
    ap.add_argument("--keep-option-dependent", action="store_true",
                    help="Keep Belebele items whose question refers to the four options. The "
                         "`tydi` rendering hides those options, so such a question is unanswerable "
                         "whatever the passage says -- the model abstains correctly and is scored "
                         "as failing on an 'answerable' row. 20%% of belebele8 items, 6%% of "
                         "delete_strict. Off by default; set this only to measure the effect.")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "data/qa_context_templates.json")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "probe/generation/validation_qa_context.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--languages", nargs="+", default=None,
                    help="FLORES codes, e.g. eng_Latn arb_Arab (default: all with a template). "
                         "For the cross-lingual sources this filters the *question* language.")
    ap.add_argument("--ids-from", type=Path, default=None,
                    help="Restrict to the ids in this JSONL (e.g. probe_predictions.jsonl), so "
                         "generations and probe scores cover exactly the same rows. Without it "
                         "the join later drops whatever the probe excluded as contaminated.")
    ap.add_argument("--per-language", type=int, default=None,
                    help="Sample this many rows per (passage-lang, question-lang) cell.")
    ap.add_argument("--balanced", action="store_true",
                    help="With --per-language, take half answerable / half not.")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT,
                    help="Prompt token budget. Must match the probe cache's max_len, or probe and "
                         "generation saw differently-truncated passages.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32, help="upper bound on rows per batch")
    ap.add_argument("--token-budget", type=int, default=8192,
                    help="rows x padded prompt length per batch, governing peak prefill memory. "
                         "A fixed row count does not: 16 rows of 1,200 tokens is 19,200 tokens of "
                         "prefill, and Qwen3.5's gated-delta-rule layers OOM a 24 GB card well "
                         "below that on the torch fallback path. Batches are formed shortest-first "
                         "so a long prompt cannot drag a wide batch up to its own length.")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Leave Qwen3's <think> block on. Off by default: the probe run had no "
                         "thinking, and it makes the abstention string unparseable.")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--device", default=None,
                    help="Override the device, e.g. cuda:0. Default picks the largest CUDA "
                         "device this torch build can actually run kernels on.")
    ap.add_argument("--require-gpu", action="store_true",
                    help="Fail rather than silently falling back to CPU.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve sources, --ids-from and sampling, then stop before loading the "
                         "model. Checks the expensive-to-discover things (does the id join land? "
                         "do all question languages have templates?) in seconds rather than after "
                         "a model load at the front of a multi-hour queue.")
    args = ap.parse_args()

    templates, sep = load_templates(args.templates)
    languages = set(args.languages or []) or None

    if args.source == "tydi":
        rows = load_rows(args.data_dir, args.split, languages, 0)
    elif args.source == "translation":
        rows = load_rows_translation(args.data_dir, args.translations, args.spec, languages, 0)
    else:
        subset = set(args.belebele_languages.split(",")) if args.belebele_languages else None
        rows = load_rows_belebele(args.belebele, args.belebele_format, languages, 0,
                                  subset=subset,
                                  drop_option_dep=not args.keep_option_dependent)
    print(f"{len(rows):,} rows from --source {args.source}")
    rows = select_rows(rows, templates, args.per_language, args.seed, args.balanced)
    if args.ids_from:
        keep = {json.loads(l)["id"] for l in open(args.ids_from, encoding="utf-8")}
        before = len(rows)
        rows = [r for r in rows if r["id"] in keep]
        print(f"--ids-from {args.ids_from.name}: {before:,} -> {len(rows):,} rows")
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done]
    # Report the language *pair*: on a cross-lingual source a single "lang_code" column hides
    # which passage language each question was asked against, and the cells are what get compared.
    n_by_cell = collections.Counter(
        (r["_meta"]["passage_lang"], r["_meta"]["question_lang"]) for r in todo)
    print(f"{len(rows):,} rows selected, {len(done):,} already generated, {len(todo):,} to do")
    xling = sum(n for (p, q), n in n_by_cell.items() if p != q)
    print(f"  {len(n_by_cell)} (passage, question) cells; {xling:,} rows cross-lingual "
          f"({100 * xling / max(len(todo), 1):.0f}%)")
    for (plang, qlang), n in sorted(n_by_cell.items())[:40]:
        print(f"  {plang:10s} -> {qlang:10s} {n:6,d}")
    if len(n_by_cell) > 40:
        print(f"  ... and {len(n_by_cell) - 40} more cells")
    if args.ids_from and not todo and not done:
        raise SystemExit(
            "error: --ids-from left zero rows. The id sets do not overlap, which usually means "
            "the cache and this --source were built from different data. Generating nothing here "
            "would look like success and fail silently at the join.")
    if args.dry_run:
        print("\n--dry-run: stopping before model load.")
        return
    if not todo:
        return

    # Not a bare "cuda": this box enumerates an unusable GT 1030 alongside the
    # 3090, and which one lands at index 0 is not stable. See _device.py.
    device = pick_device(args.device, require_gpu=args.require_gpu)
    if device == "cpu":
        print("WARNING: no usable CUDA device, running on CPU -- this will be very slow")

    tokenizer = load_tokenizer(args.model, "left")   # generation needs left padding
    # Qwen3.5 is a multimodal checkpoint (`Qwen3_5ForConditionalGeneration`), so the causal-LM
    # class is not guaranteed to accept it; _loading falls back to the image-text-to-text head.
    model = load_generator(args.model, device)

    # Render up front so batches can be grouped by length. Same renderer the probe cache uses --
    # identical rendering is what lets probe and generation be read off one prompt instead of two.
    print(f"rendering {len(todo):,} prompts ...", flush=True)
    rendered = []
    for j, r in enumerate(todo):
        text, ids, trunc = build_prompt(
            tokenizer, r["context"], r["question"], args.max_len,
            style="qa-context", templates=templates,
            question_lang=r["question_lang"], sep=sep,
            enable_thinking=args.enable_thinking)
        rendered.append((len(ids), text, trunc, r))
        if j and j % 20000 == 0:
            print(f"  {j:,}/{len(todo):,}", flush=True)
    rendered.sort(key=lambda e: e[0])

    # Group by a token budget rather than a row count, so peak prefill memory is constant:
    # short prompts batch wide, long prompts batch narrow.
    batches, cur = [], []
    for e in rendered:
        width = max([c[0] for c in cur] + [e[0]])
        if cur and ((len(cur) + 1) * width > args.token_budget or len(cur) >= args.batch_size):
            batches.append(cur); cur = []
        cur.append(e)
    if cur:
        batches.append(cur)
    print(f"{len(batches):,} batches (token budget {args.token_budget}, max {args.batch_size} rows)",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    n_done = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for group in batches:
            batch = [e[3] for e in group]
            texts = [e[1] for e in group]
            truncs = [e[2] for e in group]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[-1]:]
            texts_out = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for r, trunc, g in zip(batch, truncs, texts_out):
                # passage_lang is written as the full FLORES code and question_lang as the bare
                # three-letter template key, because that is what each is used as downstream:
                # the script check needs the passage's script, the abstention literal needs the
                # template key. analyze_abstention.py relies on both being present.
                f.write(json.dumps({
                    "id": r["id"], "lang_code": r["lang_code"],
                    "question_lang": r["question_lang"],
                    "question_lang_flores": r["_meta"]["question_lang"],
                    "passage_lang": r["_meta"]["passage_lang"],
                    "answerable": r["answerable"], "truncated": trunc,
                    "generation": g.strip(),
                }, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            rate = n_done / max(time.monotonic() - start, 1e-9)
            eta = (len(todo) - n_done) / rate if rate else 0
            # flush=True: stdout is block-buffered when redirected to a file, so
            # without this a backgrounded run shows no progress until it exits.
            print(f"  {n_done}/{len(todo)}  {rate:.1f} rows/s  "
                  f"ETA {int(eta // 60)}m{int(eta % 60):02d}s", flush=True)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
