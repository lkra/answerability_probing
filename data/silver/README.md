# silver

Silver answerability labels and reference answers for the 8,640 `qa-context` test rows. They are
not gold labels and are used for evaluation only.

## Structure of the test set

The 8,640 rows come from 100 items (a source text and a question), each rendered in many
languages: 20 source texts (UDHR, UNESCO 2003 Convention, UNSC Resolution 2347, EU AI Act Art. 5,
EU Charter Art. 21, TFEU Art. 167, five Wikipedia articles) in 25 context languages, with 5
questions each, asked in 24 question languages.
Each item has exactly one English rendering; labels were assigned on that rendering with Claude Opus
5 and propagated to all renderings, so the label is invariant across languages by construction.

20 of the 100 items are unanswerable (20.3% of rows), one per source text.

## References

Reference answers are single extractive sentences. `target_source` per row:

| class | rows | reference |
|---|---:|---|
| `exact` | 1,752 | the refusal string quoted in the row's own prompt |
| `silver_english` | 287 | the English reference (English-question rows) |
| `silver_grounded` | 155 | wording taken from the official passage in the answer language (≥0.8 character-4-gram coverage; same-language rows) |
| `silver_translated` | 6,446 | translated from the English reference |

## Flags

- `confidence` (`high` / `medium` / `low`) on the answerability label; 12 items are medium or low, all 20 unanswerable items are high.
- `translation_uncertain` (576 rows): the translation pass flagged the unit.
- `reference_unreliable` (288 rows, items 033, 056, 073, 082): the reference does not survive translation. Exclude from per-language content scores.
- Central Kurdish references are Kurmanji, not Sorani; `ckb` is excluded from every reported score.

## Build

```bash
python3 data/silver/build_silver_items.py
python3 data/silver/propagate_silver_labels.py
python3 data/silver/build_translation_inputs.py
#   translate_workflow.js fills translation_outputs/<lang>.json (model pass; outputs committed)
python3 data/silver/merge_translations.py         # -> silver_test_labels_final.jsonl
```

## Files

| file | contents |
|---|---|
| `silver_items.json` | 100 items: English passage, question, row ids |
| `silver_labels.json` | 100 labels: answerable, English answer, confidence |
| `silver_test_labels_final.jsonl` | 8,640 rows: `id, item_id, question_lang, context_lang, answerable, confidence, target, target_source, grounding_coverage, translation_uncertain, reference_unreliable` |
| `translation_inputs/`, `translation_outputs/`, `translation_uncertainty.json` | the translation pass |
