#!/usr/bin/env python3
r"""Cache hidden states: one forward pass per row, keeping the state at one position from every layer
(or from --save-layers). --pooling last (default) reads the final token of the rendered prompt, the
newline after `assistant`; --pooling mean averages the prompt. --prompt-style bare|qa-context (see
_prompts.py). --source tydi (--split train|validation) | translation | udhr | belebele (--belebele
<file>, --belebele-languages) | mist-test. Resumable; refuses to resume into a cache built with
different settings.

Output: <out>/acts.npy (float16, rows x layers x hidden), meta.jsonl, config.json

    python3 probe/probing/extract_activations.py --model Qwen/Qwen3.5-9B --prompt-style bare \
        --split train --limit-per-lang 3000 --token-budget 4096 \
        --save-layers 0,4,8,12,16,19,22,26,29,32 --out probe/caches/qwen35-9b/bare/train
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

from _device import pick_device
from _loading import load_backbone, load_tokenizer, verify_hidden_states
from _prompts import PROMPT_STYLES, build_prompt, load_templates
from _sources import (DEFAULT_BELEBELE, DEFAULT_DATA, DEFAULT_SILVER, DEFAULT_SPEC,
                      DEFAULT_TESTS, DEFAULT_TRANSLATIONS, load_rows, load_rows_belebele,
                      load_rows_mist_test, load_rows_translation, load_rows_udhr)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUT = REPO_ROOT / "probe" / "caches"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TEMPLATES = REPO_ROOT / "data" / "qa_context_templates.json"

# qa-context adds four segments of instruction the bare style never paid for. Measured against the
# real tokenizer over all 24 templates: +54 tokens for eng, +158 worst case (ben). Extending the
# budget by that worst case rather than sharing the old 1024 keeps the *passage* allowance -- and
# so the truncation rate -- comparable between the two styles, which is the whole point of running
# them against each other.
MAX_LEN_BY_STYLE = {"bare": 1024, "qa-context": 1200}


def encode_row(tokenizer, row, max_len, style, templates, sep):
    """Render one row to token ids via the shared renderer -> (ids, truncated). Uses `context`, not
    `input`, so the passage can be truncated with the question intact; under qa-context the template
    keys on the question language.
    """
    return build_prompt(
        tokenizer, row["context"], row["question"], max_len,
        style=style, templates=templates, sep=sep,
        question_lang=row["_meta"]["question_lang"].split("_")[0],
    )[1:]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--source", default="tydi",
                   choices=["tydi", "translation", "belebele", "mist-test", "udhr"],
                   help="tydi = monolingual; translation = cross-lingual (TyDi validation passage "
                        "+ machine-translated question); belebele = constructed negatives with "
                        "both passage and question language varying; mist-test = the "
                        "organizers' 8,640 qa-context test rows, split back into passage + "
                        "question and labelled from data/silver/ (SILVER, not gold); "
                        "udhr = udhr_qa's 17,280 held-out rows, all 24x25 language pairs at the "
                        "test set's own 20.3% unanswerable rate -- the calibration set")
    p.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    p.add_argument("--udhr", type=Path, default=None,
                   help="--source udhr; default data/udhr/data/udhr_qa_train.jsonl")
    p.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    p.add_argument("--min-confidence", default=None, choices=[None, "medium", "high"],
                   help="--source mist-test: drop silver labels below this confidence")
    p.add_argument("--belebele", type=Path, default=DEFAULT_BELEBELE)
    p.add_argument("--belebele-languages", default=None,
                   help="comma-separated languages to keep as BOTH passage and question language. "
                        "Rows are then restricted to the item_ids common to all of them, which is "
                        "what makes a cross-lingual comparison item-comparable. On a subset this "
                        "intersection is much larger than the all-language one in _strict.")
    p.add_argument("--belebele-format", default="tydi", choices=["tydi", "native"],
                   help="tydi = passage\\n\\nquestion, options dropped, matching the probe's "
                        "training format; native = Belebele's own 4-option rendering")
    p.add_argument("--keep-option-dependent", action="store_true",
                   help="Keep Belebele items whose question refers to the four answer options. "
                        "The `tydi` rendering hides those options, so such an item is unanswerable "
                        "whatever the passage says -- a mislabelled row for the probe, not merely "
                        "an unscorable one for the generator. Dropped by default (6% of "
                        "delete_strict rows, 20% of belebele8 items). Set this to reproduce "
                        "results computed before the filter existed, e.g. RESULTS.md's Belebele "
                        "tables, which is the only way to tell the filter's effect apart from a "
                        "model change.")
    p.add_argument("--split", default="train", choices=["train", "validation"],
                   help="--source tydi only; the translation set is validation-derived by construction")
    p.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    p.add_argument("--spec", type=Path, default=DEFAULT_SPEC,
                   help="the export spec; the only thing carrying qid -> source language")
    p.add_argument("--out", type=Path, default=None, help="default: probe/caches/<split-or-source>")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--languages", default=None,
                   help="comma-separated lang_codes (default: all). For --source translation this "
                        "filters the *question* language.")
    p.add_argument("--save-layers", default=None,
                   help="comma-separated layer indices to store (default: all). Cuts cache size "
                        "proportionally; the indices are recorded in config.json.")
    p.add_argument("--prompt-style", default="bare", choices=list(PROMPT_STYLES),
                   help="bare = passage\\n\\nquestion, no instruction (what the probe was always "
                        "fit on). qa-context = the organizers' four-segment template, rendered "
                        "identically to generate_answers.py. Use qa-context when probe and "
                        "generation must come from the same prompt: with bare, 'the probe knew but "
                        "the model answered anyway' is an upper bound, because part of the gap is "
                        "the probe reading a prompt the generator never got.")
    p.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES,
                   help="--prompt-style qa-context only")
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--batch-size", type=int, default=64, help="upper bound on rows per batch")
    p.add_argument("--token-budget", type=int, default=16384,
                   help="rows x padded length per batch; governs peak memory (16x1024 by default)")
    p.add_argument("--max-len", type=int, default=None,
                   help="prompt token budget; passage is truncated to fit. Default depends on "
                        f"--prompt-style ({MAX_LEN_BY_STYLE}), so the passage allowance stays "
                        "comparable once the instruction overhead is paid.")
    p.add_argument("--limit-per-lang", type=int, default=0, help="0 = no limit")
    p.add_argument("--pilot", action="store_true", help="shorthand for --limit-per-lang 200")
    p.add_argument("--device", default=None, help="e.g. cuda:0 | cpu (default: pick the largest "
                                                  "CUDA device this torch build has kernels for)")
    p.add_argument("--require-gpu", action="store_true",
                   help="refuse to start on CPU. Use for unattended runs: the CPU fallback is ~100x "
                        "slower, so silently taking it turns a 30-minute job into a multi-day one.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    args.device = pick_device(args.device, require_gpu=args.require_gpu)
    if args.max_len is None:
        args.max_len = MAX_LEN_BY_STYLE[args.prompt_style]
    if args.require_gpu and not args.device.startswith("cuda"):
        sys.exit("--require-gpu was set but no usable GPU is available (see the warning above). "
                 "On this machine the 3090 is gated by the `gpu0` group and membership is fixed at "
                 "process start, so a shell that predates `usermod` will not see it: "
                 "re-enter with `sg gpu0 -c <cmd>` or start a fresh login shell.")
    if args.device == "cpu" and args.dtype == "bfloat16":
        args.dtype = "float32"          # bf16 matmul on CPU is slow and patchily supported
    if args.pilot:
        args.limit_per_lang = args.limit_per_lang or 200
    out = args.out or (DEFAULT_OUT / (args.split if args.source == "tydi" else args.source))
    out.mkdir(parents=True, exist_ok=True)
    languages = set(args.languages.split(",")) if args.languages else None

    if args.source == "tydi":
        rows = load_rows(args.data_dir, args.split, languages, args.limit_per_lang)
        what = args.split
    elif args.source == "translation":
        rows = load_rows_translation(args.data_dir, args.translations, args.spec,
                                     languages, args.limit_per_lang)
        what = f"{args.translations.stem} x TyDi validation"
    elif args.source == "mist-test":
        rows = load_rows_mist_test(args.tests, args.templates, args.silver,
                                   languages, args.limit_per_lang,
                                   min_confidence=args.min_confidence)
        what = f"{args.tests.name} qa-context x silver labels"
    elif args.source == "udhr":
        rows = load_rows_udhr(args.udhr, args.templates, languages, args.limit_per_lang)
        what = f"{Path(args.udhr).name if args.udhr else 'udhr_qa_train.jsonl'} (held-out calibration)"
    else:
        subset = set(args.belebele_languages.split(",")) if args.belebele_languages else None
        rows = load_rows_belebele(args.belebele, args.belebele_format,
                                  languages, args.limit_per_lang, subset=subset,
                                  drop_option_dep=not args.keep_option_dependent)
        what = f"{args.belebele.stem} ({args.belebele_format} rendering)"
    print(f"{len(rows)} rows from {what}"
          + (f" (<= {args.limit_per_lang}/group)" if args.limit_per_lang else ""))

    templates = sep = None
    if args.prompt_style == "qa-context":
        templates, sep = load_templates(args.templates)
        # Drop rows whose *question* language has no organizer template, and say which. Silently
        # keeping them would mean rendering some rows bare and some instructed inside one cache --
        # a prompt confound perfectly correlated with language, which is the variable under study.
        before, dropped = len(rows), collections.Counter()
        keep = []
        for r in rows:
            if r["_meta"]["question_lang"].split("_")[0] in templates:
                keep.append(r)
            else:
                dropped[r["_meta"]["question_lang"]] += 1
        rows = keep
        if dropped:
            print(f"no qa-context template for {len(dropped)} question language(s), dropped "
                  f"{before - len(rows)} of {before} rows: "
                  + ", ".join(f"{k} ({v:,})" for k, v in sorted(dropped.items())))
    if not rows:
        sys.exit("no rows selected -- check --languages / --translations / --prompt-style")

    meta_path, acts_path, cfg_path = out / "meta.jsonl", out / "acts.npy", out / "config.json"
    done = set()
    if meta_path.exists():
        with open(meta_path) as f:
            done = {json.loads(l)["id"] for l in f}
        print(f"resuming: {len(done)} rows already cached")
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print("nothing to do")
        return

    tokenizer = load_tokenizer(args.model, "right")   # we index the true last token ourselves

    print(f"loading {args.model} on {args.device} ({args.dtype}) ...", flush=True)
    model, n_layers, hidden = load_backbone(args.model, args.device, args.dtype)
    # Qwen3.5 is a multimodal checkpoint, so `hidden_states` returning the text stack is an
    # assumption rather than a given. Check it now: the failure mode is a cache of exactly the
    # right shape built from the wrong tower, which surfaces hours later as probes at chance.
    verify_hidden_states(model, tokenizer, args.device, n_layers, hidden)
    print(f"hidden states verified: {n_layers} layers (incl. embedding) x {hidden}", flush=True)

    saved_layers = list(range(n_layers))
    if args.save_layers:
        saved_layers = sorted({int(x) for x in args.save_layers.split(",")})
        bad = [l for l in saved_layers if not 0 <= l < n_layers]
        if bad:
            sys.exit(f"--save-layers out of range for this model (0..{n_layers - 1}): {bad}")
        print(f"storing {len(saved_layers)} of {n_layers} layers: {saved_layers}")
    layer_sel = torch.tensor(saved_layers, device=args.device)

    if cfg_path.exists():
        # A resumed run must not change the shape of rows already written -- nor what those rows
        # mean. Model, prompt style and max_len all silently alter the *content* of an activation
        # while leaving its shape intact, so a cache half-built under one and half under another
        # would be undetectable downstream.
        old = json.loads(cfg_path.read_text())
        if old.get("saved_layers", list(range(old["n_layers"]))) != saved_layers:
            sys.exit(f"{cfg_path} was written with saved_layers="
                     f"{old.get('saved_layers', 'all')}, which does not match this run. Use a "
                     f"different --out, or delete the cache to rebuild it.")
        for key, now in (("model", args.model), ("prompt_style", args.prompt_style),
                         ("max_len", args.max_len), ("pooling", args.pooling)):
            was = old.get(key)
            if was is not None and was != now:
                sys.exit(f"{cfg_path} was written with {key}={was!r}, this run has {now!r}. "
                         f"Mixing them in one cache is silent corruption -- use a different "
                         f"--out, or delete the cache to rebuild it.")
    else:
        cfg_path.write_text(json.dumps({
            "model": args.model, "pooling": args.pooling,
            # `udhr` is not TyDi-derived at all, so calling it "validation" (which the other
            # non-tydi sources are) would misreport where its rows came from.
            "split": (args.split if args.source == "tydi"
                      else "udhr" if args.source == "udhr" else "validation"),
            "source": args.source,
            "n_layers": len(saved_layers), "saved_layers": saved_layers,
            "model_n_layers": n_layers,
            "hidden_size": hidden, "max_len": args.max_len,
            "dtype": "float16",
            "prompt_style": args.prompt_style,
            "enable_thinking": False,
            **({"templates": str(args.templates)} if args.prompt_style == "qa-context" else {}),
            **({"translations": str(args.translations)} if args.source == "translation" else {}),
            **({"belebele": str(args.belebele), "belebele_format": args.belebele_format}
               if args.source == "belebele" else {}),
        }, indent=2))

    # Append-only: activations are flushed per batch alongside their meta rows, so the two files
    # stay in step even if the run dies mid-way.
    acts_f = open(acts_path, "ab")
    meta_f = open(meta_path, "a")
    n_trunc = 0

    # Tokenise everything up front so batches can be grouped by length. Random batches pad to the
    # longest member: mean prompt is 257 tokens but p99 is the full 1024, so a random batch of 16
    # costs about as much as 16 maximal sequences. Sorting by length cuts the padded token count
    # 3.2x on this data (measured), which is the difference between a 4-hour and a 1.3-hour run.
    print(f"tokenising {len(todo)} prompts ({args.prompt_style}, max_len {args.max_len}) ...",
          flush=True)
    encoded = []
    for i, r in enumerate(todo):
        ids, trunc = encode_row(tokenizer, r, args.max_len, args.prompt_style, templates, sep)
        encoded.append((ids, r))
        n_trunc += trunc
        if i and i % 20000 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    encoded.sort(key=lambda e: len(e[0]))

    # Group by a fixed token budget rather than a fixed row count, so peak memory is constant:
    # short rows batch wide, long rows batch narrow. --batch-size is only an upper bound.
    batches, cur = [], []
    for ids, r in encoded:
        width = max([len(i) for i, _ in cur] + [len(ids)])
        if cur and (len(cur) + 1) * width > args.token_budget or len(cur) >= args.batch_size:
            batches.append(cur); cur = []
        cur.append((ids, r))
    if cur:
        batches.append(cur)
    print(f"{len(batches)} batches (token budget {args.token_budget}, max {args.batch_size} rows)")

    done_count = 0
    with torch.no_grad():
        for bi, pairs in enumerate(batches):
            id_lists = [ids for ids, _ in pairs]
            batch = [r for _, r in pairs]
            maxlen = max(len(i) for i in id_lists)
            input_ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for i, ids in enumerate(id_lists):
                input_ids[i, :len(ids)] = torch.tensor(ids)
                attn[i, :len(ids)] = 1
            input_ids, attn = input_ids.to(args.device), attn.to(args.device)

            hs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True).hidden_states

            # Pool each layer separately and stack the pooled vectors. Stacking the raw states
            # first would materialise (B, L, T, H) -- at B=16, T=1024, 33 layers that is ~2.8 GB of
            # bf16 we immediately throw away.
            rows_ix = torch.arange(len(batch), device=input_ids.device)
            if args.pooling == "last":
                idx = attn.sum(dim=1) - 1                         # true last token per row
                pooled = torch.stack([h[rows_ix, idx, :] for h in hs], dim=1)      # (B, L, H)
            else:
                mask = attn[:, :, None].to(hs[0].dtype)
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = torch.stack([(h * mask).sum(dim=1) / denom for h in hs], dim=1)

            if len(saved_layers) != len(hs):
                pooled = pooled.index_select(1, layer_sel)
            acts_f.write(pooled.to(torch.float16).cpu().numpy().tobytes())
            for r in batch:
                meta_f.write(json.dumps(r["_meta"], ensure_ascii=False) + "\n")
            acts_f.flush(); meta_f.flush()

            done_count += len(batch)
            if bi % 50 == 0 or done_count == len(todo):
                print(f"  {done_count}/{len(todo)}", flush=True)

    acts_f.close(); meta_f.close()
    total = len(done) + len(todo)
    print(f"\nwrote {out}/acts.npy  shape ({total}, {len(saved_layers)}, {hidden}) float16 "
          f"= {total * len(saved_layers) * hidden * 2 / 1e9:.1f} GB")
    if n_trunc:
        print(f"NOTE: {n_trunc}/{len(todo)} passages truncated to fit --max-len {args.max_len}")


if __name__ == "__main__":
    main()
