#!/usr/bin/env python3
"""Generate qa-context answers for the test set with a trained adapter (loaded with Unsloth
FastModel). Filters data/test/tests.jsonl to task == "qa-context", generates for every row
(abstention is applied afterwards by the probe) and writes {"id", "output"} lines. Batched
generation with left padding, falling back to one prompt at a time if batching fails. Resumes from
an existing --out-path.

    python3 adapter/infer/generate_test_qa_context.py --checkpoint <dir> [--lang deu] [--limit 16] [--out-path <file>]
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

WORK_DIR = REPO_ROOT / "adapter" / "runs"          # adapter/train writes runs/<variant>/{outputs,adapter}
DEFAULT_ADAPTER_DIR = WORK_DIR / "v8" / "adapter"

TEST_PATH = REPO_ROOT / "data" / "test" / "tests.jsonl"
DEFAULT_OUT_PATH = WORK_DIR / "qa_context_test_out.jsonl"

# Matches training (qwen3.5_9b_unsloth_qa_context_finetune.py): MAX_SEQ_LENGTH,
# SPECIAL_TOKEN_MARGIN, HEAD_FRACTION are the same constants, so truncated test prompts
# stay in the same distribution the adapter was fine-tuned on. RESPONSE_BUDGET/
# MAX_NEW_TOKENS sized from the training completions' length distribution (96,385 rows:
# p50 117 chars, p90 285, p99 652) with headroom for tokenization variance across scripts.
MAX_SEQ_LENGTH = 4096
RESPONSE_BUDGET = 320
SPECIAL_TOKEN_MARGIN = 32
HEAD_FRACTION = 0.7
MAX_NEW_TOKENS = 320

BATCH_SIZE = 8   # matches the finetune script's PER_DEVICE_TRAIN_BATCH_SIZE - known to fit.


def _truncate_head_tail(token_ids, budget):
    if budget <= 0:
        return token_ids[:0]
    if len(token_ids) <= budget:
        return token_ids
    head_n = int(budget * HEAD_FRACTION)
    tail_n = budget - head_n
    return token_ids[:head_n] + token_ids[-tail_n:] if tail_n > 0 else token_ids[:head_n]


def _text_tokenizer(tokenizer):
    """The plain text tokenizer, not the full multimodal processor FastModel returns as
    "tokenizer" for this model - see qwen3.5_9b_unsloth_qa_context_finetune.py's own
    _text_tokenizer for why this indirection exists."""
    return getattr(tokenizer, "tokenizer", tokenizer)


def load_examples(lang, limit):
    examples = []
    with open(TEST_PATH) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("task") == "qa-context" and (lang is None or obj.get("question_lang") == lang):
                examples.append(obj)
    print(f"Loaded {len(examples)} qa-context test examples out of the full test set", flush=True)
    if limit is not None:
        examples = examples[:limit]
        print(f"--limit applied: using the first {len(examples)}", flush=True)
    return examples


def load_done_ids(out_path):
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["id"])
    return done


def load_model(source, max_seq_length):
    from unsloth import FastModel

    print(f"Loading model + adapter from {source}", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(source),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    FastModel.for_inference(model)

    text_tok = _text_tokenizer(tokenizer)
    if text_tok.pad_token_id is None:
        text_tok.pad_token = text_tok.eos_token
    # Left-padding is required for batched causal-LM generation: right-padding would put
    # pad tokens between the prompt and the generated continuation for every row shorter
    # than the batch max, corrupting position ids for those rows.
    text_tok.padding_side = "left"
    return model, tokenizer


def truncate_prompt(text_tok, raw_prompt):
    budget = MAX_SEQ_LENGTH - RESPONSE_BUDGET - SPECIAL_TOKEN_MARGIN
    ids = text_tok(text=raw_prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return raw_prompt
    ids = _truncate_head_tail(ids, budget)
    return text_tok.decode(ids)


def _to_device(inputs, device):
    if hasattr(inputs, "keys"):
        return {k: v.to(device) for k, v in inputs.items()}
    return inputs.to(device)


def generate_one(model, tokenizer, prompt):
    import torch

    text_tok = _text_tokenizer(tokenizer)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt",
    )
    input_len = inputs["input_ids"].shape[1] if hasattr(inputs, "keys") else inputs.shape[1]
    inputs = _to_device(inputs, model.device)
    with torch.inference_mode():
        if hasattr(inputs, "keys"):
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                                     pad_token_id=text_tok.pad_token_id)
        else:
            output = model.generate(input_ids=inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                                     pad_token_id=text_tok.pad_token_id)
    return text_tok.decode(output[0][input_len:], skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, prompts):
    """Batched generation: render each prompt with apply_chat_template(tokenize=False), then
    tokenize and left-pad the strings with the plain text tokenizer, since the multimodal
    processor does not pad batches.
    """
    import torch

    text_tok = _text_tokenizer(tokenizer)
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]
    # add_special_tokens=False: the chat template text already contains
    # <|im_start|>/<|im_end|> etc. - adding them again would duplicate BOS/EOS.
    inputs = text_tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                                 pad_token_id=text_tok.pad_token_id)
    completions = text_tok.batch_decode(output[:, input_len:], skip_special_tokens=True)
    return [c.strip() for c in completions]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="Load this specific checkpoint/adapter dir instead of the "
                              "promoted adapter at --adapter-dir.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--lang", default=None,
                         help="Restrict to one question_lang code, e.g. 'deu' (default: all qa-context examples)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N matching examples - for a smoke test.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing --out-path content and overwrite from scratch "
                              "(default: skip ids already present in --out-path and append).")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    args = parser.parse_args()

    source = args.checkpoint if args.checkpoint is not None else args.adapter_dir
    if not (source / "adapter_config.json").exists():
        raise SystemExit(f"error: {source} has no adapter_config.json - not a valid adapter/checkpoint dir.")
    if not TEST_PATH.exists():
        raise SystemExit(f"error: {TEST_PATH} not found")

    examples = load_examples(args.lang, args.limit)

    done_ids = set() if args.no_resume else load_done_ids(args.out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} ids already in {args.out_path}, skipping those.", flush=True)
    examples = [ex for ex in examples if ex["id"] not in done_ids]
    print(f"{len(examples)} examples left to generate.", flush=True)
    if not examples:
        return

    model, tokenizer = load_model(source, args.max_seq_length)
    text_tok = _text_tokenizer(tokenizer)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (done_ids and not args.no_resume) else "w"

    batch_ok = True
    done = 0
    with open(args.out_path, mode) as fout:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            prompts = [truncate_prompt(text_tok, ex["prompt"]) for ex in batch]

            outputs = None
            if batch_ok and len(prompts) > 1:
                try:
                    outputs = generate_batch(model, tokenizer, prompts)
                except Exception as e:
                    print(f"WARNING: batched generation failed ({e!r}); falling back to "
                          f"one-at-a-time generation for the rest of the run.", flush=True)
                    batch_ok = False

            if outputs is None:
                outputs = []
                for p, ex in zip(prompts, batch):
                    try:
                        outputs.append(generate_one(model, tokenizer, p))
                    except Exception as e:
                        print(f"FAILED id={ex.get('id')}: {e!r}", flush=True)
                        outputs.append("")

            for ex, output in zip(batch, outputs):
                fout.write(json.dumps({"id": ex["id"], "output": output}, ensure_ascii=False) + "\n")
            fout.flush()

            done += len(batch)
            if done % (args.batch_size * 5) == 0 or done == len(examples):
                print(f"{done}/{len(examples)} done", flush=True)

    print("Saved to", args.out_path, flush=True)


if __name__ == "__main__":
    main()
