# probe

The probe is an L2-regularised logistic regression (LBFGS) on the hidden state of the frozen
`Qwen3.5-9B` after layer 19, taken at the final token of the chat-template prompt (passage and
question, no instruction). The submitted probe is trained on TyDiQA and UDHR (46,987 rows) and uses
one threshold, fitted on held-out UDHR items (`submission/README.md`). We chose it over the
TyDi-only probe with a paired bootstrap over held-out UDHR items (`arms.json`).

```
probing/      extraction, probes, evaluation, controls
runs/         drivers and the committed result records
submission/   abstention mask and submitted files
caches/       activation caches (gitignored)
generation/   base-model answers (gitignored)
```

## Data

| set | file | rows |
|---|---|---|
| TyDiQA train / validation | `data/built/tydiqa_answerability/{train,validation}.jsonl` | 3,000 per language / 13,325 |
| TyDiQA/cross18 | `data/translated/Hy-MT2-7B-FP8.jsonl` on TyDi validation | 232,230 |
| UDHR | `data/udhr/data/udhr_qa_train.jsonl` | 17,280, 164 items |
| BELEBELE/unanswerable | `data/built/belebele_answerability_delete_strict.jsonl` | 22,464 |
| BELEBELE-8 | `data/built/belebele_answerability_delete_perlang.jsonl`, 8 languages | 2,048 monolingual |
| BELEBELE-8 swap | `data/built/belebele_answerability_hard_b8.jsonl` | 2,048 |
| test set | `data/test/tests.jsonl`, `data/silver/silver_test_labels_final.jsonl` | 8,640 |

Loaders: `probing/_sources.py`. Prompt: `probing/_prompts.py`. All splits are item-disjoint.

## Commands

```bash
bash probe/runs/reproduce.sh             
bash probe/runs/reproduce.sh --cpu-only  # reuse the activation caches
```

Result records land in `runs/qwen35-9b/bare/`; the 4B pilot sweep is `runs/qwen35-4b/bare/lolo_train.json`.
