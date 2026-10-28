r"""Round-trip check for the MADLAD-only languages: translate the output back to English with MADLAD
and score it with chrF against an English reference of the source question. Low agreement flags
rows to read by hand. --min-chrf also splits rows into kept/ and excluded/.

    python3 data/translation/check_backtranslation.py --translations-dir <dir> \
        --languages "Bhojpuri" "Central Kurdish" "Finnish" "Haitian Creole" "Slovak" "Yoruba"
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "google/madlad400-10b-mt"
ENGLISH_CODE = "en"


def load_questions(input_path: Path) -> dict[str, dict]:
    questions = {}
    with input_path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            questions[row["qid"]] = {"question": row["question"], "src_language": row["src_language"]}
    return questions


def load_translations(path: Path) -> dict[str, str]:
    translations = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            translations[obj["id"]] = obj["question"]
    return translations


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def translate_to_english(model, tokenizer, texts: list[str], batch_size: int, gen_kwargs: dict) -> list[str]:
    results = []
    for batch in batched(texts, batch_size):
        prefixed = [f"<2{ENGLISH_CODE}> {t}" for t in batch]
        enc = tokenizer(prefixed, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**enc, **gen_kwargs)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return results


def load_english_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            cache[obj["id"]] = obj["english"]
    return cache


def get_source_english(qids: list[str], questions: dict[str, dict], cache_path: Path,
                        model, tokenizer, batch_size: int, gen_kwargs: dict) -> dict[str, str]:
    """Translate original source questions to English, reusing/extending a shared cache."""
    cache = load_english_cache(cache_path)
    missing = [qid for qid in qids if qid not in cache]
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        texts = [questions[qid]["question"] for qid in missing]
        translations = translate_to_english(model, tokenizer, texts, batch_size, gen_kwargs)
        with cache_path.open("a") as f:
            for qid, english in zip(missing, translations):
                f.write(json.dumps({"id": qid, "english": english}, ensure_ascii=False) + "\n")
                cache[qid] = english
    return cache


def write_filtered(target_lang: str, translated: dict[str, str], qids: list[str], scores: list[float],
                    min_chrf: float, output_dir: Path) -> None:
    """Splits the checked rows into kept/excluded jsonl files (same {id, lang, question} shape
    as translations-dir), by whether their round-trip chrF meets min_chrf."""
    kept_dir = output_dir / "kept"
    excluded_dir = output_dir / "excluded"
    kept_dir.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)
    kept_path = kept_dir / f"{target_lang.lower()}.jsonl"
    excluded_path = excluded_dir / f"{target_lang.lower()}.jsonl"

    n_kept = 0
    with kept_path.open("w") as kf, excluded_path.open("w") as ef:
        for qid, score in zip(qids, scores):
            record = json.dumps({"id": qid, "lang": target_lang, "question": translated[qid]}, ensure_ascii=False)
            if score >= min_chrf:
                kf.write(record + "\n")
                n_kept += 1
            else:
                ef.write(record + "\n")
    print(f"[{target_lang}] filtered at chrF>={min_chrf}: {n_kept}/{len(qids)} kept "
          f"-> {kept_path} (excluded -> {excluded_path})")


def check_language(target_lang: str, questions: dict[str, dict], translations_dir: Path, output_dir: Path,
                    cache_path: Path, model, tokenizer, batch_size: int, gen_kwargs: dict, chrf: CHRF,
                    min_chrf: float | None = None) -> None:
    translated = load_translations(translations_dir / f"{target_lang.lower()}.jsonl")
    qids = [qid for qid in translated if qid in questions]
    print(f"[{target_lang}] {len(qids)} rows to check")
    if not qids:
        return

    source_english = get_source_english(qids, questions, cache_path, model, tokenizer, batch_size, gen_kwargs)
    roundtrip_english = translate_to_english(
        model, tokenizer, [translated[qid] for qid in qids], batch_size, gen_kwargs
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_lang.lower()}.tsv"
    scores = []
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["qid", "src_language", "source_question", "target_translation",
                          "roundtrip_english", "source_english_ref", "chrf"])
        for qid, roundtrip in zip(qids, roundtrip_english):
            score = chrf.sentence_score(roundtrip, [source_english[qid]]).score
            scores.append(score)
            question = questions[qid]
            writer.writerow([
                qid, question["src_language"], question["question"], translated[qid],
                roundtrip, source_english[qid], f"{score:.2f}",
            ])

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    print(f"[{target_lang}] chrF mean={sum(scores)/n:.2f} min={sorted_scores[0]:.2f} "
          f"p50={sorted_scores[n // 2]:.2f} max={sorted_scores[-1]:.2f} -> {output_path}")

    if min_chrf is not None:
        write_filtered(target_lang, translated, qids, scores, min_chrf, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Back-translate target-language output to English and compare against "
                     "an English reference of the source question, for languages with no "
                     "second model's output to diff against."
    )
    parser.add_argument("--questions", type=Path,
                         default=Path("data/translated/questions_to_translate.tsv"))
    parser.add_argument("--translations-dir", type=Path, required=True,
                         help="Dir of per-language jsonl files (id/lang/question) to check.")
    parser.add_argument("--languages", nargs="+", required=True,
                         help="Target languages to check, e.g. \"Central Kurdish\" Finnish Yoruba.")
    parser.add_argument("--output-dir", type=Path,
                         default=Path("data/translated/backtranslation_check"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-chrf", type=float, default=None,
                         help="If set, also split each language's rows into output-dir/kept/ "
                              "and output-dir/excluded/ jsonl files (same shape as the input "
                              "translations) by this round-trip chrF threshold.")
    args = parser.parse_args()

    gen_kwargs = {"max_new_tokens": args.max_tokens, "do_sample": False, "num_beams": 4}

    questions = load_questions(args.questions)
    cache_path = args.output_dir / "_source_english_cache.jsonl"
    chrf = CHRF()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    for target_lang in args.languages:
        check_language(target_lang, questions, args.translations_dir, args.output_dir,
                        cache_path, model, tokenizer, args.batch_size, gen_kwargs, chrf, args.min_chrf)


if __name__ == "__main__":
    main()
