# submission

The system abstains when the TyDi+UDHR probe's `p(answerable)`
(`../runs/qwen35-9b/bare/mist_test_predictions_tydi_udhr.jsonl`) is below 0.4223, which
corresponds to a global bias of +0.3134 fitted on held-out UDHR items. This applies to 1,768 of the
8,640 rows. The output on an
abstained row is the refusal string quoted in that row's prompt (`data/qa_context_templates.json`).

```bash
R=probe/runs/qwen35-9b/bare
python3 probe/submission/build_submission.py --predictions $R/mist_test_predictions_tydi_udhr.jsonl \
    --offsets $R/mist_test_predictions_tydi_udhr_offsets_global.json --offsets-mode centred \
    --out probe/submission/qa-context-abstentions.jsonl
python3 probe/submission/build_full_submission.py \
    --qa-context <adapter answers> --sum-sum <file> --qa-oeg <file> --out <submission.jsonl>
```

`build_full_submission.py` writes the mask over the adapter's answers, adds the other two
sub-tasks, and refuses to write unless all 12,775 ids are covered with non-empty output.

`submitted/`: `TeamSpinoza-primary.jsonl` (Adapter 1), `TeamSpinoza-Variant_V4.jsonl` (Adapter 2),
`TeamSpinoza-Variant_sum-verifier3.jsonl` (Sub-task 2 system, no abstentions). 12,775 rows each.
