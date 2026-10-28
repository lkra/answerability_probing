# test

`tests.jsonl`: the WMT26 MIST test release (Hugging Face `pinzhenchen/wmt26-mist-test`, 20 July 2026 version, CC BY 4.0),
downloaded by `data/fetch_sources.py` at the revision pinned in `data/sources.json` (gitignored). 12,775 rows: 8,640 `qa-context`, 2,359 `qa-oeg`, 1,776 `sum-sum`.

Fields: `id`, `prompt`, `task`, `question_lang`. `id` = `<task>_<n>_<question_lang>_<context_lang>`.
