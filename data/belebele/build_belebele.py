#!/usr/bin/env python3
r"""Build BELEBELE answerability sets: every item twice, with its true passage (answerable=1) and
with a constructed unanswerable twin (answerable=0).

Items are aligned across the 27 languages on (link, question_number); the negative for an item is
decided once, on English, and applied in every language. --negatives random|hard substitute
another passage (hard: the most lexically similar); delete keeps the true passage and removes the
sentence carrying the answer, kept only where the answer sits in one sentence and does not survive
elsewhere, and only in languages whose passage splits into as many sentences as English
(--delete-align strict drops the item everywhere otherwise, per-language only there). Thai yields
no delete negatives. Negated questions whose distractor loses support are dropped. --pairs
same|english-source|all selects the (passage, question) language pairs; --languages restricts.

Output rows: id, item_id, passage_lang, question_lang, answerable, input, output, passage_link,
true_link, negatives, deleted_index, deleted_en, options_affected.

    python3 data/belebele/build_belebele.py --negatives delete --delete-align strict --pairs all \
        --output data/built/belebele_answerability_delete_strict.jsonl
"""

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

# The 27 WMT26 MIST languages, all present in Belebele.
MIST_LANGUAGES = [
    "arb_Arab", "ben_Beng", "ces_Latn", "ckb_Arab", "deu_Latn", "eng_Latn",
    "fin_Latn", "fra_Latn", "hat_Latn", "hin_Deva", "ind_Latn", "ita_Latn",
    "jpn_Jpan", "kor_Hang", "mar_Deva", "pes_Arab", "por_Latn", "rus_Cyrl",
    "slk_Latn", "spa_Latn", "swh_Latn", "tel_Telu", "tha_Thai", "tur_Latn",
    "vie_Latn", "yor_Latn", "zho_Hans",
]

# Abstention strings verified in the organizer sample (MCIF). Deliberately not
# extended with guessed translations -- see --unanswerable-target.
VERIFIED_ABSTENTION = {
    "eng_Latn": "Not answerable.",
    "deu_Latn": "Nicht zu beantworten.",
    "ita_Latn": "Non è possibile rispondere.",
    "zho_Hans": "无法回答。",
}


def load_languages(belebele_dir: Path, languages: list[str]) -> dict[str, dict]:
    """Return {lang: {item_key: row}}. item_key is (link, question_number)."""
    data = {}
    for lang in languages:
        path = belebele_dir / f"{lang}.jsonl"
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            sys.exit(1)
        rows = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[(r["link"], r["question_number"])] = r
        data[lang] = rows
    return data


def check_parallel(data: dict[str, dict]) -> list:
    """Verify every language carries the same item set, and that the correct
    answer index agrees across languages. Returns the sorted item keys."""
    keysets = {lang: frozenset(rows) for lang, rows in data.items()}
    reference = next(iter(keysets.values()))
    mismatched = [lang for lang, ks in keysets.items() if ks != reference]
    if mismatched:
        raise RuntimeError(f"Item sets differ from reference in: {mismatched}")

    keys = sorted(reference)
    ref_lang = next(iter(data))
    disagree = 0
    for key in keys:
        nums = {data[lang][key]["correct_answer_num"] for lang in data}
        if len(nums) != 1:
            disagree += 1
    if disagree:
        raise RuntimeError(
            f"correct_answer_num disagrees across languages on {disagree} items; "
            "options are not index-aligned and this script's assumption is wrong"
        )
    print(
        f"Parallelism verified: {len(keys)} items identical across "
        f"{len(data)} languages, answer index agrees on all of them "
        f"(reference {ref_lang})."
    )
    return keys


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 2]


def tfidf_vectors(passages: dict[str, str]) -> dict[str, dict[str, float]]:
    """L2-normalized tf-idf vectors over word unigrams.

    Deliberately not character n-grams: those are dominated by shared function
    words and produce distractors that are lexically similar but topically
    unrelated, which is not what "hard" should mean here. idf weighting pushes
    the similarity onto content words instead.
    """
    docs = {k: collections.Counter(tokenize(v)) for k, v in passages.items()}
    df: collections.Counter = collections.Counter()
    for counts in docs.values():
        df.update(counts.keys())
    n_docs = len(docs)
    vectors = {}
    for key, counts in docs.items():
        vec = {}
        for term, tf in counts.items():
            idf = math.log(n_docs / (1 + df[term])) + 1.0
            vec[term] = (1.0 + math.log(tf)) * idf
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        vectors[key] = {t: w / norm for t, w in vec.items()}
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


# Questions that quote the multiple-choice list. In --format open the options
# are not shown, so these cannot be answered from any passage and a model's
# abstention on them says nothing about answerability detection.
OPTION_DEPENDENT = [
    r"which of the following", r"which of these", r"all of the following",
    r"\bof the following\b", r"none of the (above|following)",
    r"which statement", r"which one",
]


def is_option_dependent(question: str) -> bool:
    return any(re.search(p, question, re.I) for p in OPTION_DEPENDENT)


def report_negative_strength(sims: list[float], weak_below: float = 0.05) -> None:
    """How topically close the chosen distractors actually are.

    'hard' only means 'hardest available in a pool of 488 passages', which for
    an isolated topic can still be very far away. Quote this distribution rather
    than the label 'hard' when reporting: a negative at cosine ~0.01 is
    detectable from the topic alone, and inflates measured answerability skill.
    """
    ordered = sorted(sims)
    n = len(ordered)

    def pct(p: float) -> float:
        return ordered[min(n - 1, int(p * n))]

    weak = sum(1 for s in sims if s < weak_below)
    print(
        f"  distractor tf-idf cosine: min {ordered[0]:.3f} | p25 {pct(.25):.3f} | "
        f"median {pct(.5):.3f} | p75 {pct(.75):.3f} | max {ordered[-1]:.3f}"
    )
    print(
        f"  {weak}/{n} passages ({100 * weak / n:.0f}%) have no distractor above "
        f"cosine {weak_below} -- effectively topic-mismatch negatives"
    )


# ---------------------------------------------------------------------------
# Deletion negatives
# ---------------------------------------------------------------------------

# FLORES Thai separates sentences with a plain space, which also occurs inside
# them. There is no reliable boundary, so no delete negatives are offered.
UNSPLITTABLE = {"tha_Thai"}
CJK_LANGS = {"zho_Hans", "jpn_Jpan"}
CJK_TERMINATORS = "。！？"
# Latin/Cyrillic full stop, Arabic full stop and question mark, Devanagari danda
# (Telugu and Marathi in this corpus use the Latin full stop, but the danda is
# cheap to accept and harmless where it never occurs).
TERMINATORS = ".!?۔؟।"
CLOSERS = "\"'”’)]}」』»"

# "Dr. Damadian" is not two sentences. Spurious boundaries here are the main
# reason a naive splitter disagrees with itself across a parallel corpus, and
# every disagreement costs an item to the alignment gate.
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "sr", "jr", "vs", "etc", "approx",
    "no", "fig", "dept", "gen", "col", "lt", "capt", "rev", "hon", "inc",
    "ltd", "co", "univ", "al", "u.s", "u.k", "e.g", "i.e", "ca", "cf", "vol",
    "pp", "ed", "est", "min", "max", "mt", "ft", "hrs",
}

# A sentence does not begin with a lowercase letter, so a full stop followed by
# one is an abbreviation we did not list. Worth 11 points of sentence-count
# agreement on Czech, 12 on Slovak, 7 on Finnish; a no-op on the caseless
# scripts, where it costs nothing.
LOWERCASE_START = re.compile(r"[a-zà-öø-ÿа-яα-ω]")

# German "am 15. August", Czech/Slovak/Finnish/Turkish dates: an ordinal marker,
# not a sentence end. Restricted to 1-2 digit numbers so that a sentence ending
# in a year ("...in 1977.") still splits, and to the languages that write dates
# this way so English sentences ending in a small number are unaffected.
ORDINAL_PERIOD_LANGS = {"deu_Latn", "ces_Latn", "slk_Latn", "fin_Latn", "tur_Latn"}

NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")
# English only -- the deletion target is chosen on English, so no per-language
# numeral vocabulary is ever needed. Belebele golds are frequently a bare
# numeral ("4") whose supporting sentence spells it out ("Four skiers...").
NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
    "thousand": "1000", "million": "1000000",
}

# "what did X not do", "which is not true", "all except". For these the gold is
# already the option the passage does not support; see the module docstring.
NEGATED_QUESTION = re.compile(r"\b(not|never|except|least|neither|nor)\b", re.I)


def _preceding_word(text: str, dot: int) -> str:
    j = dot
    while j > 0 and (text[j - 1].isalpha() or text[j - 1] == "."):
        j -= 1
    return text[j:dot]


def _is_abbreviation(text: str, dot: int) -> bool:
    word = _preceding_word(text, dot)
    if not word:
        return False
    if len(word) == 1 and word.isalpha():  # an initial, e.g. "D. K. Arya"
        return True
    return word.lower().strip(".") in ABBREVIATIONS


def sentence_spans(text: str, lang: str) -> list[tuple[int, int]] | None:
    """Character spans of each sentence, or None if the language is unsplittable.

    Spans rather than strings so a deletion can be a byte-exact excision from
    the original passage: everything the model sees that was not deleted is
    identical to the answerable row, with no re-joining artefact that could by
    itself signal "this passage was tampered with".
    """
    if lang in UNSPLITTABLE:
        return None
    terminators = CJK_TERMINATORS if lang in CJK_LANGS else TERMINATORS
    spans: list[tuple[int, int]] = []
    start, i, n = 0, 0, len(text)
    while i < n:
        if text[i] not in terminators:
            i += 1
            continue
        if lang not in CJK_LANGS and text[i] == "." and _is_abbreviation(text, i):
            i += 1
            continue
        if lang in ORDINAL_PERIOD_LANGS and text[i] == ".":
            j = i
            while j > 0 and text[j - 1].isdigit():
                j -= 1
            if 0 < i - j <= 2:
                i += 1
                continue
        end = i + 1
        while end < n and text[end] in CLOSERS:
            end += 1
        # In CJK a terminator ends the sentence outright; elsewhere it only does
        # so when whitespace (or the end of the passage) follows, which keeps
        # decimals like "4:41.30" and urls intact.
        if lang not in CJK_LANGS and end < n and not text[end].isspace():
            i += 1
            continue
        nxt = end
        while nxt < n and text[nxt].isspace():
            nxt += 1
        if lang not in CJK_LANGS and nxt < n and LOWERCASE_START.match(text[nxt]):
            i += 1
            continue
        if text[start:end].strip():
            spans.append((start, end))
        start = i = nxt
    if text[start:].strip():
        spans.append((start, n))
    return spans


def word_tokens(text: str) -> set[str]:
    """Content words. Bare numerals are excluded here and handled by NUM_RE, so
    that "4:41.30" does not silently contribute a token "41"."""
    return {t for t in re.split(r"\W+", text.lower()) if len(t) > 2 and not t.isdigit()}


def key_tokens(text: str) -> set[str]:
    """Content words plus numerals, with English number words folded onto their
    digits so that a gold of "4" matches a sentence saying "Four skiers"."""
    words = word_tokens(text)
    nums = set(NUM_RE.findall(text))
    for w in list(words):
        if w in NUM_WORDS:
            words.discard(w)
            nums.add(NUM_WORDS[w])
    return words | nums


def coverage(needle: set[str], haystack: set[str]) -> float:
    return len(needle & haystack) / len(needle) if needle else 0.0


def select_deletion(en_row: dict, spans: list, min_cov: float, min_margin: float):
    """Choose the sentence to delete, on English. Returns (pick, None) or
    (None, reason). Conservative by design: anything ambiguous is dropped, since
    a wrong choice produces a row labelled unanswerable that is still
    answerable, which is worse than a smaller dataset."""
    text = en_row["flores_passage"]
    n = int(en_row["correct_answer_num"])
    gold_key = key_tokens(en_row[f"mc_answer{n}"])
    if not gold_key:
        return None, "gold-has-no-content-tokens"
    if len(spans) < 2:
        return None, "passage-is-a-single-sentence"

    sentences = [text[a:b] for a, b in spans]
    sent_keys = [key_tokens(s) for s in sentences]
    q_key = key_tokens(en_row["question"])

    covs = [coverage(gold_key, sk) for sk in sent_keys]
    order = sorted(
        range(len(sentences)),
        key=lambda i: (covs[i], coverage(q_key, sent_keys[i])),
        reverse=True,
    )
    best = order[0]
    if covs[best] < min_cov:
        return None, "answer-not-localised-in-any-sentence"
    if covs[best] - covs[order[1]] < min_margin:
        return None, "answer-spread-over-several-sentences"

    remainder: set[str] = set()
    for i, sk in enumerate(sent_keys):
        if i != best:
            remainder |= sk
    if coverage(gold_key, remainder) >= min_cov:
        return None, "answer-survives-the-deletion"

    # Which distractors were only supported by the sentence we are removing.
    full: set[str] = set()
    for sk in sent_keys:
        full |= sk
    affected = []
    for k in range(1, 5):
        if k == n:
            continue
        opt_key = key_tokens(en_row[f"mc_answer{k}"])
        if not opt_key:
            continue
        if coverage(opt_key, full) >= min_cov and coverage(opt_key, remainder) < min_cov:
            affected.append(k)

    return {
        "index": best,
        "span": spans[best],
        "sentence": sentences[best],
        "affected": affected,
    }, None


def build_deletion_map(data: dict, keys: list, passage_langs: list, args):
    """Return ({item_key: {lang: reduced_passage}}, {item_key: pick}). The sentence to delete is chosen
    on English and applied by index in every language; under --delete-align strict an item is kept
    only if all of `passage_langs` (the languages actually rendered) split into the same sentence
    count.
    """
    english = data["eng_Latn"]
    splittable = [l for l in passage_langs if l not in UNSPLITTABLE]
    skipped = [l for l in passage_langs if l in UNSPLITTABLE]
    if "eng_Latn" not in splittable:
        splittable = ["eng_Latn"] + splittable
    if skipped:
        print(f"  no delete negatives for {', '.join(skipped)} "
              "(no reliable sentence boundary); those passage languages are "
              "excluded from the output")

    spans = {
        lang: {k: sentence_spans(data[lang][k]["flores_passage"], lang) for k in keys}
        for lang in splittable
    }

    aligned = collections.Counter()
    for k in keys:
        n_en = len(spans["eng_Latn"][k])
        for lang in splittable:
            if len(spans[lang][k]) == n_en:
                aligned[lang] += 1
    print(f"  sentence-count agreement with English, over {len(keys)} items:")
    for lang in sorted(splittable, key=lambda l: -aligned[l]):
        if lang == "eng_Latn":
            continue
        print(f"    {lang}: {aligned[lang]}/{len(keys)} "
              f"({100 * aligned[lang] / len(keys):.0f}%)")

    reasons: collections.Counter = collections.Counter()
    reduced: dict = {}
    audit: dict = {}
    gate_passed: list = []  # items the English-side gates accepted, pre-alignment
    for k in keys:
        pick, reason = select_deletion(
            english[k], spans["eng_Latn"][k],
            args.delete_min_coverage, args.delete_min_margin,
        )
        if pick is None:
            reasons[reason] += 1
            continue
        if (pick["affected"] and NEGATED_QUESTION.search(english[k]["question"])
                and not args.keep_affected_negated):
            reasons["negated-question-loses-a-distractor"] += 1
            continue

        per_lang, misaligned = {}, []
        n_en = len(spans["eng_Latn"][k])
        for lang in splittable:
            if len(spans[lang][k]) != n_en:
                misaligned.append(lang)
                continue
            a, b = spans[lang][k][pick["index"]]
            passage = data[lang][k]["flores_passage"]
            per_lang[lang] = re.sub(r"\s{2,}", " ", passage[:a] + passage[b:]).strip()
        gate_passed.append((k, set(per_lang)))
        if misaligned and args.delete_align == "strict":
            reasons["sentence-count-misaligned-in-some-language"] += 1
            continue
        if not per_lang:
            reasons["sentence-count-misaligned-in-some-language"] += 1
            continue
        reduced[k] = per_lang
        audit[k] = pick

    print(f"  kept {len(reduced)}/{len(keys)} items "
          f"({100 * len(reduced) / len(keys):.0f}%). Dropped:")
    for reason, count in reasons.most_common():
        print(f"    {count:4d}  {reason}")
    n_affected = sum(1 for k in reduced if audit[k]["affected"])
    if n_affected:
        print(f"  {n_affected} kept items still strip support from a distractor "
              "option (non-negated questions) -> options_affected is set on them")

    # What the alignment policy actually costs, in items per language. Under
    # strict every language gets the same item set by definition; the numbers
    # that matter then are how much bigger each language could have been.
    per_lang_counts = {
        lang: sum(1 for k in reduced if lang in reduced[k]) for lang in splittable
    }
    per_lang_possible = {
        lang: sum(1 for _, langs in gate_passed if lang in langs)
        for lang in splittable
    }
    if args.delete_align == "strict":
        print(f"  strict alignment: all {len(splittable)} passage language(s) "
              f"share the same {len(reduced)} items. {len(gate_passed)} items "
              "passed the English gates; --delete-align per-language would "
              "have kept, per language:")
        for lang in sorted(splittable, key=lambda l: -per_lang_possible[l]):
            print(f"    {lang}: {per_lang_possible[lang]} items")
    else:
        print("  per-language alignment: item counts differ by language --")
        for lang in sorted(per_lang_counts, key=lambda l: -per_lang_counts[l]):
            print(f"    {lang}: {per_lang_counts[lang]} items")
        print("  NOTE: the item sets are NOT identical across languages, so a "
              "cross-lingual difference in results is partly a difference in "
              "which items each language was scored on. Join on item_id and "
              "restrict to the intersection before comparing languages.")
    return reduced, audit


def build_distractor_map(
    english: dict, keys: list, strategy: str, seed: int
) -> dict[str, str]:
    """Map each passage link -> the link of the passage used as its distractor.

    Computed on English only, then applied by item id in every language, so the
    construction is identical everywhere.
    """
    links = sorted({link for link, _ in keys})
    passage_of = {}
    for (link, qnum) in keys:
        passage_of.setdefault(link, english[(link, qnum)]["flores_passage"])

    if strategy == "random":
        rng = random.Random(seed)
        shuffled = links[:]
        # Derangement: no link may be its own distractor.
        for _ in range(1000):
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(links, shuffled)):
                break
        else:
            raise RuntimeError("could not build a derangement")
        return dict(zip(links, shuffled))

    if strategy == "hard":
        vectors = tfidf_vectors(passage_of)
        mapping = {}
        sims = []
        for a in links:
            va = vectors[a]
            best, best_sim = None, -1.0
            for b in links:
                if b == a:
                    continue
                sim = cosine(va, vectors[b])
                if sim > best_sim:
                    best, best_sim = b, sim
            if best is None:  # degenerate: fall back to any other passage
                best = links[0] if links[0] != a else links[1]
            mapping[a] = best
            sims.append(best_sim)
        report_negative_strength(sims)
        return mapping

    raise ValueError(f"unknown negative strategy: {strategy}")


def resolve_pairs(mode: str, languages: list[str], explicit: str | None) -> list:
    if explicit:
        pairs = []
        for tok in explicit.split(","):
            p, q = tok.strip().split(":")
            pairs.append((p, q))
        return pairs
    if mode == "same":
        return [(l, l) for l in languages]
    if mode == "english-source":
        return [("eng_Latn", q) for q in languages]
    if mode == "all":
        return [(p, q) for p in languages for q in languages]
    raise ValueError(f"unknown pairs mode: {mode}")


def render(row_passage: dict, row_question: dict, fmt: str,
           passage_override: str | None = None) -> tuple[str, str]:
    """Return (input, gold_output). The passage comes from one language, the
    question/options/answer from another. passage_override substitutes the
    passage text (used by --negatives delete, where the shown passage is the
    true one minus a sentence)."""
    passage = row_passage["flores_passage"] if passage_override is None else passage_override
    q = row_question
    if fmt == "mcq":
        # Byte-identical to the organizer's Belebele rendering (verified 300/300
        # against their zho_Hans rows).
        inp = (
            f"{passage}\n{q['question']}\n"
            f"1. {q['mc_answer1']}\n2. {q['mc_answer2']}\n"
            f"3. {q['mc_answer3']}\n4. {q['mc_answer4']}"
        )
        n = int(q["correct_answer_num"])
        return inp, f"{n}: {q[f'mc_answer{n}']}"
    if fmt == "open":
        inp = f"{passage}\n{q['question']}"
        n = int(q["correct_answer_num"])
        return inp, q[f"mc_answer{n}"]
    raise ValueError(f"unknown format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--belebele-dir", type=Path, default=Path("data/upstream/belebele"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/built/belebele_answerability.jsonl")
    )
    parser.add_argument(
        "--languages",
        default=",".join(MIST_LANGUAGES),
        help="Comma-separated subset of languages to load.",
    )
    parser.add_argument(
        "--pairs",
        choices=["same", "english-source", "all"],
        default="same",
        help="Which (passage_lang, question_lang) pairs to materialise. "
        "'same' (default) = 27 monolingual pairs. 'english-source' = English "
        "passage, question in each language. 'all' = 729 pairs, ~1.3M rows.",
    )
    parser.add_argument(
        "--explicit-pairs",
        default=None,
        help="Override --pairs with e.g. 'eng_Latn:swh_Latn,swh_Latn:eng_Latn'.",
    )
    parser.add_argument("--format", choices=["mcq", "open"], default="mcq",
                        help="'mcq' (default): Belebele's native 4-option form, "
                        "byte-identical to the organizer rendering; abstention "
                        "is a discrete choice, which sidesteps per-language "
                        "answer matching. 'open': passage+question only, gold is "
                        "the answer text -- more comparable with the TyDi and "
                        "MCIF sets, but 34%% of Belebele questions reference the "
                        "option list ('which of the following...') and are "
                        "incoherent without it, so pair it with "
                        "--drop-option-dependent.")
    parser.add_argument(
        "--drop-option-dependent", action="store_true",
        help="Drop items whose question references the option list. Only "
        "meaningful with --format open, where such questions cannot be answered "
        "at all and would be scored as abstentions for the wrong reason.",
    )
    parser.add_argument(
        "--negatives", choices=["random", "hard", "delete"], default="random",
        help="How the unanswerable row is built. 'random'/'hard' swap in "
        "another passage; 'delete' keeps the true passage and removes the "
        "sentence carrying the answer (see module docstring -- this filters "
        "the item set, so it is not the same population as the other two).",
    )
    parser.add_argument(
        "--delete-min-coverage", type=float, default=0.6,
        help="--negatives delete: fraction of the gold answer's content tokens "
        "that a sentence must contain to count as carrying the answer.",
    )
    parser.add_argument(
        "--delete-min-margin", type=float, default=0.3,
        help="--negatives delete: how far the best sentence must beat the "
        "runner-up. Below this the answer is not localised in one sentence and "
        "the item is dropped.",
    )
    parser.add_argument(
        "--delete-align", choices=["strict", "per-language"], default="strict",
        help="--negatives delete: 'strict' (default) keeps an item only if "
        "every language splits its passage into the same number of sentences "
        "as English, so the deleted sentence is the same everywhere and cross-"
        "lingual differences cannot be a construction artefact. 'per-language' "
        "keeps it in the languages that do align and drops it in the others, "
        "which yields more rows at the cost of that guarantee.",
    )
    parser.add_argument(
        "--keep-affected-negated", action="store_true",
        help="--negatives delete: keep negated questions ('what did X not "
        "do...') even when the deleted sentence was the only support for a "
        "distractor option. Those items then have two defensible answers; off "
        "by default.",
    )
    parser.add_argument(
        "--unanswerable-target", choices=["empty", "english"], default="empty",
        help="Target output for constructed unanswerable rows. Same policy as "
        "build_tydiqa.py: no guessed per-language translations.",
    )
    parser.add_argument("--max-items", type=int, default=0,
                        help="Use only the first N items (0 = all 900).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    print(f"Loading {len(languages)} languages from {args.belebele_dir} ...")
    data = load_languages(args.belebele_dir, languages)
    keys = check_parallel(data)

    if args.drop_option_dependent:
        before = len(keys)
        keys = [k for k in keys if not is_option_dependent(data["eng_Latn"][k]["question"])]
        print(f"Dropped {before - len(keys)} option-dependent items "
              f"({100 * (before - len(keys)) / before:.0f}%), {len(keys)} remain.")
    elif args.format == "open":
        n_dep = sum(1 for k in keys if is_option_dependent(data["eng_Latn"][k]["question"]))
        print(f"WARNING: --format open with {n_dep}/{len(keys)} "
              f"({100 * n_dep / len(keys):.0f}%) option-dependent questions kept. "
              "These cannot be answered without the option list; consider "
              "--drop-option-dependent.")

    if args.max_items:
        keys = keys[: args.max_items]
        print(f"Restricted to the first {len(keys)} items.")

    if "eng_Latn" not in data:
        print("error: eng_Latn must be loaded (distractors are computed on it)",
              file=sys.stderr)
        sys.exit(1)
    # item_index is assigned before any strategy-specific filtering, so
    # belebele_0123 denotes the same underlying item whichever --negatives was
    # used. A delete run is a subset of those ids, never a renumbering of them.
    item_index = {key: i for i, key in enumerate(keys)}

    pairs = resolve_pairs(args.pairs, languages, args.explicit_pairs)

    distractor: dict = {}
    reduced: dict = {}
    audit: dict = {}
    if args.negatives == "delete":
        blocked = sorted({p for p, _ in pairs if p in UNSPLITTABLE})
        if blocked:
            pairs = [(p, q) for p, q in pairs if p not in UNSPLITTABLE]
            print(f"Dropped every pair whose passage language is in "
                  f"{', '.join(blocked)}: no sentence boundaries, no deletion.")
        if not pairs:
            print("error: no usable language pairs left for --negatives delete",
                  file=sys.stderr)
            sys.exit(1)
        # Only the passage languages of the pairs being materialised constrain
        # the alignment gate; a language used solely for questions never has its
        # passage split, so it cannot veto an item.
        passage_langs = sorted({p for p, _ in pairs})
        print("Building delete negatives (choice of sentence made on English) ...")
        reduced, audit = build_deletion_map(data, keys, passage_langs, args)
        keys = [k for k in keys if k in reduced]
        if not keys:
            print("error: no items survived the delete gates", file=sys.stderr)
            sys.exit(1)
    else:
        distractor = build_distractor_map(
            data["eng_Latn"], keys, args.negatives, args.seed
        )
        print(f"Built {args.negatives} distractor map over "
              f"{len(set(distractor))} passages.")

    print(f"Materialising {len(pairs)} language pair(s) x {len(keys)} items x 2 "
          f"= {len(pairs) * len(keys) * 2} rows (fewer if --delete-align "
          f"per-language drops some).")

    # A passage link -> one representative item key, for fetching its passage.
    rep_of_link = {}
    for key in keys:
        rep_of_link.setdefault(key[0], key)

    abstention = (
        "" if args.unanswerable_target == "empty" else VERIFIED_ABSTENTION["eng_Latn"]
    )

    # A distractor can accidentally contain the gold answer, which makes the
    # "unanswerable" label wrong. Detected once on English (the construction is
    # parallel, so a leak in English is a leak everywhere) and flagged per row
    # rather than silently dropped -- with short generic answers like "Class"
    # the match is often coincidental, so this needs eyeballing, not a filter.
    # (--negatives delete has no distractor passage to leak into; its equivalent
    # check is the "answer-survives-the-deletion" gate in select_deletion.)
    leaky = set()
    for key in keys if args.negatives != "delete" else []:
        en_row = data["eng_Latn"][key]
        n = int(en_row["correct_answer_num"])
        gold = str(en_row[f"mc_answer{n}"]).strip()
        d_passage = data["eng_Latn"][rep_of_link[distractor[key[0]]]]["flores_passage"]
        if len(gold) > 2 and gold.lower() in d_passage.lower():
            leaky.add(key)
    if leaky:
        print(f"  {len(leaky)}/{len(keys)} items ({100 * len(leaky) / len(keys):.1f}%) "
              "have the gold answer present in their distractor passage "
              "-> flagged answer_in_distractor=true")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped_unaligned = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for p_lang, q_lang in pairs:
            for key in keys:
                idx = item_index[key]
                item_id = f"belebele_{idx:04d}"
                q_row = data[q_lang][key]
                true_link = key[0]

                if args.negatives == "delete":
                    # --delete-align per-language: this item may be usable in
                    # some passage languages and not others. Skip the pair
                    # entirely rather than emit an unpaired answerable row,
                    # which would quietly break the 50/50 balance.
                    reduced_passage = reduced[key].get(p_lang)
                    if reduced_passage is None:
                        skipped_unaligned += 1
                        continue
                    pick = audit[key]
                    d_link = true_link
                    inp_neg, _ = render(data[p_lang][key], q_row, args.format,
                                        passage_override=reduced_passage)
                    neg_extra = {
                        "deleted_index": pick["index"],
                        "deleted_en": pick["sentence"],
                        "options_affected": pick["affected"],
                    }
                else:
                    d_link = distractor[true_link]
                    d_key = rep_of_link[d_link]
                    inp_neg, _ = render(data[p_lang][d_key], q_row, args.format)
                    neg_extra = {
                        "deleted_index": None,
                        "deleted_en": None,
                        "options_affected": None,
                    }

                # answerable: the passage that actually supports the question
                inp, gold = render(data[p_lang][key], q_row, args.format)
                out.write(json.dumps({
                    "id": f"{item_id}_{p_lang}_{q_lang}_pos",
                    "item_id": item_id,
                    "passage_lang": p_lang,
                    "question_lang": q_lang,
                    "answerable": 1,
                    "input": inp,
                    "output": gold,
                    "passage_link": true_link,
                    "true_link": true_link,
                    "negatives": None,
                    "answer_in_distractor": False,
                    "deleted_index": None,
                    "deleted_en": None,
                    "options_affected": None,
                }, ensure_ascii=False) + "\n")

                # unanswerable: same question, a passage that does not support
                # it -- another passage entirely for random/hard, the true
                # passage minus its answer sentence for delete.
                out.write(json.dumps({
                    "id": f"{item_id}_{p_lang}_{q_lang}_neg",
                    "item_id": item_id,
                    "passage_lang": p_lang,
                    "question_lang": q_lang,
                    "answerable": 0,
                    "input": inp_neg,
                    "output": abstention,
                    "passage_link": d_link,
                    "true_link": true_link,
                    "negatives": args.negatives,
                    "answer_in_distractor": key in leaky,
                    **neg_extra,
                }, ensure_ascii=False) + "\n")
                n += 2

    if skipped_unaligned:
        print(f"  skipped {skipped_unaligned} (item, passage language) "
              "combinations whose sentence split disagreed with English "
              "(--delete-align per-language)")
    print(f"Wrote {n} rows to {args.output} (50/50 answerable by construction).")


if __name__ == "__main__":
    main()
