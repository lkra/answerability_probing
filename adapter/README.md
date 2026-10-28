# adapter

The adapters are LoRA fine-tunes of `unsloth/Qwen3.5-9B` that answer a question about a passage.
They are trained on answerable questions only; the probe in `../probe/` decides when to abstain.

| | training data | rows | submitted as |
|---|---|---|---|
| Adapter 1 (`--variant v8`) | TyDiQA/cross23; target = answer sentence if ≤120 chars, else answer span; capitalised and terminated | 100,466 | `TeamSpinoza-primary` |
| Adapter 2 (`--variant v4`) | 40/50/10 TyDiQA/cross · BELEBELE · UDHR, ≤4 rows per distinct completion per language | 60,000 | `TeamSpinoza-Variant_V4` |

Training files: `data/built/qa_context_v8_train.jsonl`, `data/built/qa_context_v4_train.jsonl`
(`../data/README.md`).

## Recipe

Defaults of `train/qwen35_9b_qa_context_finetune.py`, identical for both:

| | |
|---|---|
| base model | `unsloth/Qwen3.5-9B`, bf16 |
| LoRA | r 16, alpha 16, dropout 0, all linear layers, vision tower frozen |
| loss | completions only |
| sequence length | 4096 |
| batch | 8 × grad-accum 1 |
| learning rate | 1e-4, warmup 3%, linear decay, adamw_8bit, weight decay 0.01 |
| epochs | 1 |
| checkpoints | every 200 steps, 200 held-out rows; lowest held-out loss promoted |
| seed | 3407 |

For Adapter 1 we kept checkpoint 12,400 of 12,534 (eval loss 0.3905).

## Commands

```bash
mkdir -p logs                                                    # SLURM logs land here
sbatch adapter/train/run_finetune.sh --variant v8 --phase train
sbatch adapter/train/run_finetune.sh --variant v4 --phase train
python3 adapter/train/qwen35_9b_qa_context_finetune.py --variant v8 --phase eval-history   # eval-loss curve
sbatch adapter/train/run_finetune.sh --variant v8 --phase save --checkpoint adapter/runs/v8/outputs/checkpoint-12400
python3 adapter/infer/generate_test_qa_context.py --checkpoint adapter/runs/v8/outputs/checkpoint-12400 \
    --out-path adapter/runs/v8/qa_context_test_out.jsonl
python3 adapter/infer/score_qa_context.py --predictions adapter/runs/v8/qa_context_test_out.jsonl \
    --exclude-lang ckb --by question_lang
```

`--variant` sets data file and run directory (`adapter/runs/<variant>/`). Training resumes from the
latest checkpoint. The launchers' `#SBATCH` partition, conda path and scratch directory are
placeholders in angle brackets; fill them in or set `CONDA_SH`, `CONDA_ENV`, `HF_HOME`, `TMPDIR`.

The off-target column of `score_qa_context.py` needs fastText's
[`lid.176.ftz`](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz) at
`data/upstream/lid.176.ftz` (or `LID_MODEL`).
