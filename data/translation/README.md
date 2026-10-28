# translation

These scripts produced `data/translated/`.

| step | script | output |
|---|---|---|
| export TyDi validation questions | `export_questions_for_translation.py` | `translated/questions_to_translate.tsv`, 6,666 questions |
| translate into the 17 other Hy-MT2 languages; compile | `translate_questions.py --model hymt2`, `compile_question_translations.py` | `translated/Hy-MT2-7B-FP8.jsonl` |
| add English under the same decoding settings | `run_english_translation.sh`, `append_language_to_compiled.py` | same file, 116,176 rows, 18 languages |
| fine-tuning job spec: question, span, answer sentence; two target languages per item | `build_translation_jobs.py` | `translated/qa_translation_items.jsonl`, `translated/to-translate/<lang>.tsv` |
| translate the jobs | `translate_questions.py --model hymt2 --input-dir data/translated/to-translate --output-dir data/translated/finetune/hymt2` | `translated/finetune/hymt2/`, 18 languages |
| re-translate Persian | same, `--output-dir .../hymt2_pes_fix` | `translated/finetune/hymt2_pes_fix/`, overrides `hymt2` |
| translate the MADLAD-only languages, 1,500 items each; filter | `translate_questions.py --model madlad400`, `filter_madlad_translations.py` | `translated/madlad400/{kept,excluded}/` |

Models: `tencent/Hy-MT2-7B-FP8` (temperature 0.7, top-p 0.6, top-k 20, repetition penalty 1.05),
`google/madlad400-10b-mt` (4 beams). `run_translate_questions.sh` and `run_check_backtranslation.sh`
are the Snellius launchers.

Persian: Hy-MT2 renders Arabic *من* ("who") as Persian *من* ("I"); affected pairs were re-translated.

MADLAD filter (`filter_madlad_translations.py`): drops empty output, repetition loops, a known fixed
hallucination, copy-through, length-ratio outliers, over-frequent answers, and answers not
recoverable from the translated answer sentence (character 4-gram containment). Retention over
1,500 items: Bhojpuri 33.6% (742 items returned no answer translation), Central Kurdish 55.9%,
Finnish 57.2%, Haitian Creole 62.7%, Slovak 62.7%. Yoruba was dropped after the 100-question
round-trip check (`check_backtranslation.py`). `kept/` is what the builders read.
