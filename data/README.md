# data

```text
sources.json               pinned upstream revisions
qa_context_templates.json  the organisers' 24 prompt templates and refusal strings, from test/tests.jsonl
test/                      the test release (gitignored; fetched by fetch_sources.py)
upstream/                  downloaded corpora (gitignored)
translated/                the translation bundle (gitignored)
translation/               how translated/ was produced
tydi/  belebele/  mix/     builders
udhr/                      UDHR construction (udhr/README.md)
silver/                    silver labels for the test set (silver/README.md)
built/                     outputs (jsonl gitignored; *_stats.json committed)
build_all.sh               all builders in order, with row-count checks
```

Setup and build commands are in the [main README](../README.md). `build_all.sh` writes:

| set | builder | output (`built/`) | rows |
| --- | --- | --- | --- |
| templates | `extract_qa_templates.py` | `../qa_context_templates.json` | 24 |
| UDHR | `udhr/build_rows.py` | `../udhr/data/udhr_qa_train{,_answerable,_answerable_ctx25}.jsonl` | 17,280 / 13,776 / 43,008 |
| TyDiQA answerability | `tydi/build_tydiqa.py` | `tydiqa_answerability/{train,validation}.jsonl` | 116,067 / 13,325 |
| TyDiQA/cross23, sentence targets | `tydi/build_qa_context_train.py` | `qa_context_tydi_train.jsonl` | 100,466 |
| TyDiQA/cross23, adaptive targets | same, `--answer-form adaptive` | `qa_context_tydi_adaptive_train.jsonl` | 100,466 |
| Adapter 1 training file | `tydi/build_v8_wellformed.py` | `qa_context_v8_train.jsonl` | 100,466 |
| BELEBELE generation set | `belebele/build_belebele_qa_context.py` | `belebele_qa_context_train.jsonl` | 54,924 |
| BELEBELE/unanswerable | `belebele/build_belebele.py --negatives delete --delete-align strict --pairs all` | `belebele_answerability_delete_strict.jsonl` | 22,464 |
| BELEBELE-8, per-language alignment | `belebele/build_belebele.py --negatives delete --delete-align per-language --pairs all` | `belebele_answerability_delete_perlang.jsonl` | 430,110 |
| BELEBELE-8, swap negatives | `belebele/build_belebele.py --negatives hard --languages <8> --pairs same`; `belebele/restrict_hard_to_belebele8.py` | `belebele_answerability_hard_b8.jsonl` | 2,560 |
| Adapter 2 training file | `mix/build_qa_context_mix.py` | `qa_context_v4_train.jsonl` | 60,000 |

## Row format

`{id, prompt, completion, task, question_lang}`; `id = qa-context_<n>_<question_lang>_<passage_lang>`;
`prompt = intro \n\n passage \n\n prefix+question \n\n instruction` in the question language, with
`\n` the literal two characters the organisers' prompts contain. Ids: TyDi from 1, UDHR from
500001, BELEBELE from 700001. BELEBELE and UDHR rows carry `source`, `context_lang`, `item_id`.

## TyDiQA/cross

Each of the 64,257 TyDiQA items is assigned two of the 24 question languages; its question, gold
span and answer sentence are translated (`translation/`), the passage is not. Sentence targets use
the translated answer sentence, else the span; adaptive targets use the sentence if ≤120 characters,
else the span (median 40 characters). `tydi/build_v8_wellformed.py` capitalises each target and
ends it with the language's majority terminator (`।` Bengali, Hindi; `。` Japanese, Chinese; `.`
otherwise). The result covers 23 question and 11 passage languages.

## BELEBELE

Generation set: the 597 items whose question does not depend on the answer options, in every
question language, with the passage in English and three other languages (no Bhojpuri). Probe sets:
the unanswerable twin is the true passage with the answer-bearing sentence deleted. *Strict* keeps
the 16 items whose passages split identically in every language; *per-language* keeps each item in
the languages where its passage splits like English. The swap set substitutes another passage
instead, on the eight-language subset.

## Adapter 2 mix

The mix takes 24,000 TyDiQA/cross rows (sentence targets), 30,000 BELEBELE rows and 6,000 UDHR
rows, with at most 4 rows per question language and completion. The builder stops if any language
has fewer than 35% distinct completions. The seed is 3407.
