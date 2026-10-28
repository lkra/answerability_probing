#!/usr/bin/env python3
"""Qwen3.5-9B LoRA fine-tuning (Unsloth) for the qa-context adapters.

Recipe (the defaults): unsloth/Qwen3.5-9B, LoRA r=16 alpha=16 dropout=0 target_modules="all-linear",
vision tower frozen, train_on_responses_only, max_seq_length 4096, batch 8 x grad-accum 1, lr 1e-4,
1 epoch, warmup 3%, linear schedule, adamw_8bit, seed 3407. Loaded with FastModel (multimodal
checkpoint, hybrid attention; "all-linear" covers the linear-attention layers). --variant v4|v8
selects data/built/qa_context_<variant>_train.jsonl and adapter/runs/<variant>/. Phases: train,
eval-history (no GPU), test, save, gguf (untested on this architecture).

    python3 adapter/train/qwen35_9b_qa_context_finetune.py --variant v8 --phase train
    python3 adapter/train/qwen35_9b_qa_context_finetune.py --variant v8 --phase eval-history
    python3 adapter/train/qwen35_9b_qa_context_finetune.py --variant v8 --phase save --checkpoint adapter/runs/v8/outputs/checkpoint-12400
"""

from __future__ import annotations  # `Path | None` annotations on Python < 3.10

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent  # adapter/

# ---- Config ------------------------------------------------------------------

MODEL_NAME = "unsloth/Qwen3.5-9B"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = False  # see qwen3_8b_unsloth_qa_context_finetune.py's same setting - bf16
                       # fits an HPC A100/H100 easily, no reason to pay 4-bit overhead.

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
# "all-linear" instead of a target_modules list: covers both attention mechanisms' Linear
# layers without hand-naming them. The finetune_* flags scope which components it applies to.
TARGET_MODULES = "all-linear"
FINETUNE_VISION_LAYERS = False      # text-only task, so the vision tower stays frozen
FINETUNE_LANGUAGE_LAYERS = True
FINETUNE_ATTENTION_MODULES = True
FINETUNE_MLP_MODULES = True
GRAD_CHECKPOINT = "unsloth"  # hardware-specific - re-tune for actual available VRAM

NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8   # effective batch = this * GRADIENT_ACCUMULATION_STEPS -
GRADIENT_ACCUMULATION_STEPS = 1  # copied from the 8B script's starting point, not
                                  # re-tuned for this model - smoke-test first (see Usage).
LEARNING_RATE = 1e-4        # Run 1's recipe, held fixed across the ablation (the main-repo
                            # script defaults to Run 2's 2e-4 -- easy to confound a variant with)
WARMUP_RATIO = 0.03         # a ratio, not a step count, so it scales with actual run length
MAX_GRAD_NORM = 1.0         # SFTTrainer's default

VALID_FRACTION = 0.01       # held-out eval catches a divergence within a few hundred
VALID_MAX_EXAMPLES = 200    # steps instead of only at the end of a run
EVAL_STEPS = 200
SAVE_STEPS = 200
# Keep every checkpoint: cheap for LoRA, and selection needs earlier ones to compare.
SAVE_TOTAL_LIMIT = None
SEED = 3407

SPECIAL_TOKEN_MARGIN = 32
HEAD_FRACTION = 0.7

DATASET_NUM_PROC = 2  # re-tune per this server's actual core count

GGUF_QUANTIZATIONS = ["q4_k_m", "q8_0"]

VARIANTS = ("v4", "v8")
DATA_DIR = ROOT.parent / "data" / "built"   # built by data/build_all.sh
RUNS_DIR = ROOT / "runs"   # runs/<variant>/{outputs,adapter,gguf}; outputs/ holds the HF Trainer
                           # checkpoints and is resumable across HPC job resubmits


def variant_paths(variant: str) -> dict:
    return {
        "data_path": [DATA_DIR / f"qa_context_{variant}_train.jsonl"],
        "output_dir": RUNS_DIR / variant / "outputs",
        "adapter_dir": RUNS_DIR / variant / "adapter",
        "gguf_dir": RUNS_DIR / variant / "gguf",
    }


def check_deps():
    missing = []
    for mod in ("unsloth", "torch", "trl", "datasets", "transformers", "bitsandbytes"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("This script needs an Unsloth install on a CUDA Linux box:")
        print("  pip install unsloth  # pulls in a matching trl/peft/accelerate/bitsandbytes")
        print("Unsloth does not run on Apple Silicon or CPU-only machines.")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("No CUDA GPU detected (torch.cuda.is_available() is False) - this script needs an NVIDIA GPU.")
        sys.exit(1)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        pass


# ---- 1. Load the qa-context training data -------------------------------------

def load_data(data_paths: list[Path]):
    """Loads one or more qa_context_train.jsonl-shaped files and concatenates them into
    one dataset."""
    from datasets import concatenate_datasets, load_dataset

    for path in data_paths:
        if not path.exists():
            raise SystemExit(f"error: {path} not found - run data/build_all.sh first")

    parts = []
    for path in data_paths:
        part = load_dataset("json", data_files=str(path), split="train")
        print(f"  {path.name}: {len(part)} examples")
        parts.append(part)

    raw_dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    print(f"Combined dataset: {len(raw_dataset)} examples")
    if "question_lang" in raw_dataset.column_names:
        from collections import Counter
        print("Language counts:", dict(Counter(raw_dataset["question_lang"])))
    return raw_dataset


# ---- 2. Format examples, with truncation that protects the label ------------

def _truncate_head_tail(token_ids, budget):
    if budget <= 0:
        return token_ids[:0]
    if len(token_ids) <= budget:
        return token_ids
    head_n = int(budget * HEAD_FRACTION)
    tail_n = budget - head_n
    if tail_n > 0:
        return token_ids[:head_n] + token_ids[-tail_n:]
    return token_ids[:head_n]


def prepare_dataset(raw_dataset, tokenizer, max_seq_length, dataset_num_proc):
    # Use the inner plain tokenizer for length measurements: the multimodal processor adds a
    # batch dimension, so len() returns 1 and silently disables the truncation guard below.
    text_tok = _text_tokenizer(tokenizer)

    def build_text(example):
        user_content = example["prompt"]
        assistant_content = example["completion"]

        response_ids = text_tok(assistant_content, add_special_tokens=False)["input_ids"]
        budget = max_seq_length - len(response_ids) - SPECIAL_TOKEN_MARGIN

        user_ids = text_tok(user_content, add_special_tokens=False)["input_ids"]
        if len(user_ids) > budget:
            user_ids = _truncate_head_tail(user_ids, budget)
            user_content = text_tok.decode(user_ids)

        convo = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        return {"text": text}

    dataset = raw_dataset.map(build_text, remove_columns=raw_dataset.column_names, num_proc=dataset_num_proc)
    print(dataset)

    sample = dataset.select(range(min(200, len(dataset))))
    lengths = [len(text_tok(t, add_special_tokens=False)["input_ids"]) for t in sample["text"]]
    print("max tokenized length in a 200-example sample:", max(lengths))
    if max(lengths) > max_seq_length + SPECIAL_TOKEN_MARGIN:
        print("WARNING: sample exceeds MAX_SEQ_LENGTH + margin - raise SPECIAL_TOKEN_MARGIN.")

    valid_size = min(VALID_MAX_EXAMPLES, max(1, int(len(dataset) * VALID_FRACTION)))
    split = dataset.train_test_split(test_size=valid_size, seed=SEED)
    print(f"train: {len(split['train'])}  valid: {len(split['test'])}")
    return split["train"], split["test"]


# ---- 3. Load Qwen3.5-9B and attach LoRA adapters -----------------------------

def load_model(max_seq_length, grad_checkpoint):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect (bf16 on Ampere+, fp16 otherwise)
        load_in_4bit=LOAD_IN_4BIT,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=FINETUNE_VISION_LAYERS,
        finetune_language_layers=FINETUNE_LANGUAGE_LAYERS,
        finetune_attention_modules=FINETUNE_ATTENTION_MODULES,
        finetune_mlp_modules=FINETUNE_MLP_MODULES,
        target_modules=TARGET_MODULES,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=grad_checkpoint,
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = sorted(
        (d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def print_eval_history(output_dir: Path) -> None:
    """Prints eval_loss per step from the latest checkpoint's trainer_state.json - Trainer
    accumulates the full log_history there, not just that checkpoint's own eval, so this is
    the whole run's loss curve without loading the model. The quantitative starting point
    for picking which checkpoint(s) to actually compare with --phase test --checkpoint."""
    source = find_latest_checkpoint(output_dir)
    if source is None:
        raise SystemExit(f"No checkpoint-* dir under {output_dir}. Run --phase train first.")
    state = json.loads((source / "trainer_state.json").read_text())
    evals = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]
    if not evals:
        print(f"No eval_loss entries found in {source}/trainer_state.json.")
        return

    best_step, best_loss = min(evals, key=lambda x: x[1])
    print(f"{'step':>8}  {'eval_loss':>10}")
    for step, loss in evals:
        print(f"{step:>8}  {loss:>10.4f}" + ("  <- lowest" if step == best_step else ""))
    print(f"\nLowest eval_loss: step {best_step} ({best_loss:.4f}).")
    print(f"Compare it against later checkpoints before trusting it - a lower loss doesn't "
          f"rule out memorization by itself:")
    print(f"  python {Path(__file__).name} --phase test --checkpoint {output_dir}/checkpoint-{best_step}")


def load_trained_model(max_seq_length, adapter_dir: Path, output_dir: Path, checkpoint: Path | None = None):
    """Load a trained model for save/gguf/test: `checkpoint` if given, else the saved adapter, else the
    latest Trainer checkpoint.
    """
    from unsloth import FastModel

    if checkpoint is not None:
        if not (checkpoint / "adapter_config.json").exists():
            raise SystemExit(f"--checkpoint {checkpoint} has no adapter_config.json - not a valid checkpoint/adapter dir.")
        source = checkpoint
    elif (adapter_dir / "adapter_config.json").exists():
        source = adapter_dir
    else:
        source = find_latest_checkpoint(output_dir)
        if source is None:
            raise SystemExit(
                f"No trained model found: neither {adapter_dir} nor a checkpoint-* dir under "
                f"{output_dir} exists. Run --phase train (or --phase all) first."
            )
    print(f"Loading previously trained model from {source}")
    return FastModel.from_pretrained(
        model_name=str(source),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )


def _text_tokenizer(tokenizer):
    """The plain text tokenizer inside FastModel's multimodal processor (or the tokenizer itself if it is
    already plain), for call sites that cannot pass text= explicitly, such as SFTTrainer.
    """
    return getattr(tokenizer, "tokenizer", tokenizer)


# ---- 4. Train with LoRA -------------------------------------------------------

def run_training(model, tokenizer, train_dataset, eval_dataset, max_seq_length,
                  batch_size, grad_accum, learning_rate, epochs, output_dir, dataset_num_proc):
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only
    import torch

    trainer = SFTTrainer(
        model=model,
        tokenizer=_text_tokenizer(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=dataset_num_proc,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=WARMUP_RATIO,
            max_grad_norm=MAX_GRAD_NORM,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            # No group_by_length: this trl's SFTConfig has no such field (TypeError).
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir=str(output_dir),
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # Resumable across HPC job resubmissions - picks up the latest checkpoint automatically.
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoints = sorted(
        (d for d in os.listdir(output_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    )
    if existing_checkpoints:
        print(f"Found {len(existing_checkpoints)} checkpoint(s) in {output_dir} - resuming from {existing_checkpoints[-1]}.")
    else:
        print(f"No checkpoints found in {output_dir} - starting fresh.")

    trainer_stats = trainer.train(resume_from_checkpoint=bool(existing_checkpoints))
    print(trainer_stats)
    return model, tokenizer


# ---- 5. Save the LoRA adapter -------------------------------------------------

def save_adapter(model, tokenizer, adapter_dir):
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("Saved LoRA adapter to", adapter_dir)


# ---- 6. Merge + quantize to GGUF ---------------------------------------------

def run_gguf(model, tokenizer, gguf_dir):
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=GGUF_QUANTIZATIONS,
    )
    print("Saved quantized GGUF binaries to", gguf_dir)


# ---- 7. Quick generation sanity check ----------------------------------------

def run_test(model, tokenizer, raw_dataset, prompt_index=0):
    from unsloth import FastModel

    FastModel.for_inference(model)
    sample_prompt = raw_dataset[prompt_index]["prompt"]
    # content must be typed parts: apply_chat_template(tokenize=True) scans it for
    # image/video parts and chokes on a bare string (training uses tokenize=False).
    messages = [{"role": "user", "content": [{"type": "text", "text": sample_prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt",
    ).to(model.device)
    # A multimodal processor may return a dict/BatchFeature rather than a bare tensor.
    if hasattr(inputs, "keys"):
        input_len = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, max_new_tokens=300, use_cache=True)
    else:
        input_len = inputs.shape[1]
        output = model.generate(input_ids=inputs, max_new_tokens=300, use_cache=True)
    print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))


# ---- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B LoRA fine-tuning (Unsloth) on the qa-context task. "
                    "See the module docstring in this file for why it differs from the 8B script."
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True,
                         help="which adapter to train: sets --data-path/--output-dir/--adapter-dir/"
                              "--gguf-dir defaults to data/qa_context_<variant>_train.jsonl and "
                              "runs/<variant>/ (any of them can still be overridden).")
    parser.add_argument("--phase", choices=["train", "eval-history", "save", "gguf", "test", "all"],
                         default="all",
                         help="\"all\" is train+test only, not save/gguf too - picking which "
                              "checkpoint to ship is a separate step (see eval-history / "
                              "--checkpoint).")
    parser.add_argument("--data-path", type=Path, nargs="+", default=None,
                         help="One or more qa_context_train.jsonl-shaped files (concatenated).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="For save/gguf/test run standalone: load this specific "
                              "checkpoint-N dir instead of the saved adapter/latest checkpoint "
                              "(e.g. output-dir/checkpoint-7000) - for comparing checkpoints "
                              "before deciding, or promoting a chosen one via --phase save.")
    parser.add_argument("--gguf-dir", type=Path, default=None)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--dataset-num-proc", type=int, default=DATASET_NUM_PROC,
                         help="Worker processes for dataset mapping/tokenization - re-tune "
                              "per server, don't assume the default core count.")
    parser.add_argument("--no-grad-checkpoint", action="store_true",
                         help="Disable Unsloth gradient checkpointing - VRAM-dependent, test before relying on it.")
    args = parser.parse_args()
    for name, value in variant_paths(args.variant).items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    print(f"variant={args.variant}  data={[str(p) for p in args.data_path]}  output_dir={args.output_dir}")
    print(f"recipe: max_seq={args.max_seq_length} batch={args.batch_size} grad_accum={args.grad_accum} "
          f"lr={args.learning_rate} epochs={args.epochs}")

    if args.phase == "eval-history":
        print_eval_history(args.output_dir)
        return

    check_deps()
    # "all" deliberately excludes save/gguf - promoting a checkpoint is a separate step.
    phases = ["train", "test"] if args.phase == "all" else [args.phase]
    grad_checkpoint = False if args.no_grad_checkpoint else GRAD_CHECKPOINT

    raw_dataset = None
    model = tokenizer = None

    if "train" in phases:
        raw_dataset = load_data(args.data_path)
        model, tokenizer = load_model(args.max_seq_length, grad_checkpoint)
        train_dataset, eval_dataset = prepare_dataset(
            raw_dataset, tokenizer, args.max_seq_length, args.dataset_num_proc
        )
        model, tokenizer = run_training(
            model, tokenizer, train_dataset, eval_dataset, args.max_seq_length,
            args.batch_size, args.grad_accum, args.learning_rate, args.epochs, args.output_dir,
            args.dataset_num_proc,
        )

    if "save" in phases:
        if model is None:
            model, tokenizer = load_trained_model(args.max_seq_length, args.adapter_dir, args.output_dir, args.checkpoint)
        save_adapter(model, tokenizer, args.adapter_dir)

    if "gguf" in phases:
        if model is None:
            model, tokenizer = load_trained_model(args.max_seq_length, args.adapter_dir, args.output_dir, args.checkpoint)
        run_gguf(model, tokenizer, args.gguf_dir)

    if "test" in phases:
        if model is None:
            model, tokenizer = load_trained_model(args.max_seq_length, args.adapter_dir, args.output_dir, args.checkpoint)
        if raw_dataset is None:
            raw_dataset = load_data(args.data_path)
        run_test(model, tokenizer, raw_dataset)


if __name__ == "__main__":
    main()
