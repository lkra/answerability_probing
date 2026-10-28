import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

# Decoding defaults are per-model, because the two models come from opposite
# conventions and a single global default cannot serve both. Any flag passed on
# the command line overrides the model's value.
MODELS = {
    "hymt2": {
        "path": "tencent/Hy-MT2-7B-FP8",
        "type": "causal",
        # Tencent's recommended recipe for the 7B, from the model card. These are
        # a bundle, not independent knobs: temperature 0.7 is only safe because
        # top_k 20 / top_p 0.6 cut off the tail first. What was here before
        # (top_p 1.0, top_k -1, repetition_penalty 1.0) is the card's *30B-A3B*
        # row, left over from when that checkpoint was being tried -- so the 7B
        # was being sampled at 0.7 over the full, untruncated distribution.
        "decoding": {
            "temperature": 0.7, "top_p": 0.6, "top_k": 20,
            "repetition_penalty": 1.05, "num_beams": 1, "max_tokens": 2048,
        },
    },
    "madlad400": {
        "path": "google/madlad400-10b-mt",
        "type": "seq2seq",
        # No vendor recipe to follow: an encoder-decoder MT model, where beam
        # search is the convention and sampling has no backing.
        #
        # max_tokens is 2048, not the old 256. Questions top out at 220 chars and
        # gold spans run ~12, but `answer_sentence` reaches 3,502 chars (128 over
        # 1,024) and the Belebele Bhojpuri fill carries passages to 1,352 -- and
        # those go into Devanagari, where a Latin-centric sentencepiece needs
        # roughly one token per two characters. A truncated sentence is a
        # half-sentence *training target*, so the cap is set where it does not
        # bind rather than where it usually doesn't. Since batches are
        # length-sorted below, short batches still stop at EOS and pay nothing
        # for the headroom.
        "decoding": {
            "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "repetition_penalty": 1.0, "num_beams": 4, "max_tokens": 2048,
        },
    },
}

TARGET_LANGUAGES = [
    "Arabic", "Bengali", "Czech", "English", "German", "Hindi", "Indonesian",
    "Italian", "Japanese", "Korean", "Marathi", "Persian", "Portuguese",
    "Russian", "Spanish", "Turkish", "Vietnamese", "Chinese",
]

# Languages only MADLAD-400 covers (not offered as a default for other models).
MADLAD_EXTRA_LANGUAGES = [
    "Bhojpuri", "Central Kurdish", "Finnish", "Haitian Creole", "Slovak", "Yoruba",
]

# MADLAD-400 expects the target language as a "<2xx>" prefix token rather
# than a natural-language instruction, so it needs its own code per language.
MADLAD_LANG_CODES = {
    "Arabic": "ar", "Bengali": "bn", "Czech": "cs", "English": "en",
    "German": "de", "Hindi": "hi",
    "Indonesian": "id", "Italian": "it", "Japanese": "ja", "Korean": "ko",
    "Marathi": "mr", "Persian": "fa", "Portuguese": "pt", "Russian": "ru",
    "Spanish": "es", "Turkish": "tr", "Vietnamese": "vi", "Chinese": "zh",
    "Bhojpuri": "bho", "Central Kurdish": "ckb", "Finnish": "fi",
    "Haitian Creole": "ht", "Slovak": "sk", "Yoruba": "yo",
}

MODEL_LANGUAGES = {
    "hymt2": TARGET_LANGUAGES,
    "madlad400": TARGET_LANGUAGES + MADLAD_EXTRA_LANGUAGES,
}

# Maps a question's source-language iso639_1 code to the matching entry in
# TARGET_LANGUAGES, so we can skip translating a question into the language
# it's already written in.
ISO_TO_TARGET_LANG = {
    "ar": "Arabic",
    "bn": "Bengali",
    "en": "English",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
}

PROMPT_TEMPLATE = (
    "Translate the following text into {target_lang}. Note that you should "
    "only output the translated result without any additional explanation: {question}"
)
# Names the source language too - naming only the target leaves room for the model to
# misread source-script text through the target language's orthography. Documented for
# hymt2 specifically on Arabic -> Persian: Arabic "من" (who) read as Persian "من" (I),
# producing first-person statements instead of questions (data/translation/README.md).
# Scoped to just that one pair, not applied universally, so every other (source, target)
# pair keeps the exact prompt already used for the rest of the dataset - untested elsewhere,
# and no reason to risk a working pair to fix one that isn't.
SOURCE_NAMED_PAIRS = {("Arabic", "Persian")}
PROMPT_TEMPLATE_WITH_SOURCE = (
    "Translate the following {src_lang} text into {target_lang}. Note that you should "
    "only output the translated result without any additional explanation: {question}"
)


def build_prompt(record: dict, target_lang: str) -> str:
    if (record.get("src_language"), target_lang) in SOURCE_NAMED_PAIRS:
        return PROMPT_TEMPLATE_WITH_SOURCE.format(
            src_lang=record["src_language"], target_lang=target_lang, question=record["question"]
        )
    return PROMPT_TEMPLATE.format(target_lang=target_lang, question=record["question"])


def load_questions(input_path: Path) -> list[dict]:
    records = []
    with input_path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            records.append({
                "id": row["qid"],
                "question": row["question"],
                "src_iso639_1": row["iso639_1"],
            })
    return records


def load_questions_dir(
    input_dir: Path, wanted_languages: set[str] | None = None
) -> tuple[dict[str, list[dict]], list[str]]:
    """Read a directory of per-target-language job files (to-translate/<flores_code>.tsv: qid, item_id,
    field, src_lang, tgt_lang, tgt_language, question), keyed by tgt_language. For files whose
    language is not in wanted_languages only the header and first row are read. Returns (by_lang,
    skipped_languages).
    """
    by_lang: dict[str, list[dict]] = {}
    skipped: list[str] = []
    for path in sorted(input_dir.glob("*.tsv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            try:
                first_row = next(reader)
            except StopIteration:
                continue
            target_lang = first_row["tgt_language"]
            if wanted_languages is not None and target_lang not in wanted_languages:
                skipped.append(target_lang)
                continue
            rows = [first_row, *reader]
        by_lang[target_lang] = [
            {"id": row["qid"], "question": row["question"], "item_id": row["item_id"],
             "field": row["field"], "src_language": row["src_language"]}
            for row in rows
        ]
    return by_lang, skipped


def already_translated_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open() as f:
        for line in f:
            ids.add(json.loads(line)["id"])
    return ids


def batched(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def translate_batch(model, tokenizer, records: list[dict], target_lang: str, gen_kwargs: dict,
                     model_type: str) -> list[str]:
    if model_type == "seq2seq":
        lang_code = MADLAD_LANG_CODES[target_lang]
        texts = [f"<2{lang_code}> {r['question']}" for r in records]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**enc, **gen_kwargs)

        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(r, target_lang)}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for r in records
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**enc, **gen_kwargs)

    generated = outputs[:, enc["input_ids"].shape[-1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def translate_language(model, tokenizer, records: list[dict], target_lang: str, output_path: Path,
                        batch_size: int, gen_kwargs: dict, model_type: str) -> None:
    applicable = [r for r in records if ISO_TO_TARGET_LANG.get(r.get("src_iso639_1")) != target_lang]
    skipped = len(records) - len(applicable)

    done_ids = already_translated_ids(output_path)
    todo = [r for r in applicable if r["id"] not in done_ids]

    print(f"[{target_lang}] {skipped} skipped (already in this language), "
          f"{len(done_ids)} already done, {len(todo)} remaining")
    if not todo:
        return

    # Group similarly-sized texts together, as translate.py does for abstracts.
    # Two wins, and the second is the larger one: a batch is not padded up to a
    # long outlier sharing it, and `generate` steps the whole batch until its
    # *last* sequence hits EOS -- so a batch of 12-char gold spans is otherwise
    # kept running while one 3,500-char sentence generates. This is what makes a
    # generous --max-tokens close to free. The constant per-model wrapper (the
    # "<2xx> " prefix, or the chat template) does not affect the ordering, so the
    # raw text is the right sort key.
    #
    # Longest first, so the batch with the highest peak memory is the *first*
    # one: an OOM then costs seconds on a queued GPU job rather than surfacing
    # hours in, at the tail of a language. The cost is a pessimistic ETA early on
    # (the slowest batches set the initial rate), which self-corrects.
    todo.sort(key=lambda r: len(tokenizer(r["question"])["input_ids"]), reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(applicable)
    num_done = len(done_ids)
    start = time.monotonic()
    processed = 0
    with output_path.open("a") as f:
        for batch in batched(todo, batch_size):
            translations = translate_batch(model, tokenizer, batch, target_lang, gen_kwargs, model_type)
            for record, translation in zip(batch, translations):
                out = {"id": record["id"], "lang": target_lang, "question": translation}
                if "item_id" in record:
                    out["item_id"] = record["item_id"]
                if "field" in record:
                    out["field"] = record["field"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            num_done += len(batch)
            processed += len(batch)

            elapsed = time.monotonic() - start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = format_duration((len(todo) - processed) / rate) if rate > 0 else "n/a"
            print(f"[{target_lang}] translated {len(batch)} (total done: {num_done}/{total}, ETA: {eta})")


def main():
    parser = argparse.ArgumentParser(description="Translate questions into target languages, one language at a time.")
    parser.add_argument("--input", type=Path,
                         default=Path("data/translated/questions_to_translate.tsv"),
                         help="Single TSV expanded to every target language by this script "
                              "(ignored if --input-dir is given).")
    parser.add_argument("--input-dir", type=Path, default=None,
                         help="Directory of per-target-language job files (e.g. answerability's "
                              "to-translate/<flores_code>.tsv), already split/self-language-"
                              "filtered upstream - overrides --input.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/translated/translated"))
    parser.add_argument("--model", choices=sorted(MODELS), default="hymt2",
                         help="Which translation model to use. Also sets the decoding "
                              "defaults: "
                              + "; ".join(
                                  f"{m}: " + ", ".join(f"{k}={v}" for k, v in s["decoding"].items())
                                  for m, s in sorted(MODELS.items())))
    parser.add_argument("--languages", nargs="+", default=None,
                         help="Subset of target languages to translate into "
                              "(default: all languages supported by --model).")
    parser.add_argument("--batch-size", type=int, default=8)
    # Decoding flags all default to None so that "not passed" is distinguishable
    # from "passed the same value the model happens to default to"; the model's
    # own value fills in below.
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=None, help="0 disables sampling.")
    parser.add_argument("--num-beams", type=int, default=None,
                         help="Beam width when not sampling (--temperature 0). 1 = greedy.")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None, help="-1 disables top-k filtering.")
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only process a random N questions (for testing).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for --limit's random sample. Keep fixed across runs "
                              "you want to compare on the same subset.")
    args = parser.parse_args()

    if args.input_dir:
        # Always bounded by what --model supports, whether or not --languages was
        # passed explicitly - otherwise requesting a model-unsupported language
        # would load (and silently attempt) it instead of erroring below.
        wanted = set(MODEL_LANGUAGES[args.model])
        if args.languages is not None:
            wanted &= set(args.languages)
        records_by_lang, skipped = load_questions_dir(args.input_dir, wanted_languages=wanted)
        if skipped:
            print(f"Note: skipped {len(skipped)} language file(s) in {args.input_dir} not in the "
                  f"requested/supported set: {sorted(skipped)}")
        available_languages = sorted(records_by_lang)
    else:
        records_by_lang = None
        available_languages = MODEL_LANGUAGES[args.model]

    if args.languages is None:
        args.languages = available_languages
    unknown = set(args.languages) - set(available_languages)
    if unknown:
        raise ValueError(f"Unknown target language(s) for model {args.model!r}: {sorted(unknown)}")

    # Fill unset decoding flags from the model's own defaults.
    overridden = []
    for name, value in MODELS[args.model]["decoding"].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
        elif getattr(args, name) != value:
            overridden.append(name)

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.temperature > 0:
        gen_kwargs.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k if args.top_k > 0 else 0,  # HF disables top-k with 0, not -1
        })
    else:
        gen_kwargs["num_beams"] = args.num_beams

    if records_by_lang is not None:
        if args.limit:
            rng = random.Random(args.seed)
            records_by_lang = {
                lang: rng.sample(recs, args.limit) if len(recs) > args.limit else recs
                for lang, recs in records_by_lang.items()
            }
        total_loaded = sum(len(records_by_lang[lang]) for lang in args.languages)
        print(f"Loaded {total_loaded} jobs across {len(args.languages)} target languages from {args.input_dir}")
    else:
        records = load_questions(args.input)
        if args.limit:
            records = random.Random(args.seed).sample(records, args.limit)
        print(f"Loaded {len(records)} questions from {args.input}")

    model_path = MODELS[args.model]["path"]
    model_type = MODELS[args.model]["type"]

    # Printed rather than inferred later: the output rows carry only the
    # translation, so the log is the only record of how it was decoded.
    print(f"Model {args.model} ({model_path}), decoding: "
          + ", ".join(f"{k}={v}" for k, v in sorted(gen_kwargs.items()))
          + (f"  [overridden on the command line: {', '.join(overridden)}]"
             if overridden else ""))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if model_type == "causal":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    for target_lang in args.languages:
        output_path = args.output_dir / f"{target_lang.lower()}.jsonl"
        lang_records = records_by_lang[target_lang] if records_by_lang is not None else records
        translate_language(model, tokenizer, lang_records, target_lang, output_path, args.batch_size, gen_kwargs, model_type)


if __name__ == "__main__":
    main()
