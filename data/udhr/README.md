# udhr

This folder builds training and evaluation data from the 24 UDHR articles the test set does not
use. Model-pass
outputs (question authoring, review, transplant judgement, localisation) are committed under
`data/`; `run_all.sh` re-renders the rows from them.

```bash
data/udhr/run_all.sh --check
```

| step | script | result |
|---|---|---|
| 1 | `fetch_udhr.py` | "UDHR in Unicode" corpus, sha256 in `data/udhr_source.json` |
| 2 | `parse_udhr.py` | 24 language editions parsed, language mapping checked against the test prompts |
| 3 | `build_passages.py` | 18 passages of 224–714 characters |
| 4–5 | `export_question_jobs.py`, `merge_questions.py` | 5 questions per passage, authored by Claude Opus 5 and reviewed: 90 items, 18 unanswerable |
| 6–7 | `build_transplants.py`, `merge_transplants.py` | 74 unanswerables by moving a question to a passage that does not answer it |
| 8–9 | `build_translation_inputs.py`, `merge_translations.py` | questions and answers localised into the 24 question languages |
| 10 | `build_rows.py` | qa-context rows |
| 11 | `verify.py` | checks against `data/test/tests.jsonl` |

Rows are not committed; `data/build_all.sh` renders all three from the committed inputs.

| output | rows | use |
|---|---|---|
| `data/udhr_qa_train.jsonl` | 17,280 | probe set: 164 items, 24 question × 25 passage languages, 20.3% unanswerable |
| `data/udhr_qa_train_answerable_ctx25.jsonl` | 43,008 | pool for Adapter 2 (`--answerable-only --ctx-answerable 25`) |
| `data/udhr_qa_train_answerable.jsonl` | 13,776 | same at 8 passage languages per item |

Ids start at 500001.
