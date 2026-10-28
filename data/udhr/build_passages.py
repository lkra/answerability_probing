#!/usr/bin/env python3
"""Cut the UDHR articles the test set does not use (Arts. 3-17, 20-25, 28-30; every article the test
touches is excluded whole) into passages, parallel across the 25 context languages. Articles are
packed into runs twice, left-to-right and right-to-left, so passages overlap and a question can be
answerable from one passage and not from another.

Output: data/passages.json
"""

import argparse
import collections
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Target passage size, in characters of English. The qa-context test passages
# run 304-1693 with a median near 450; the UDHR ones specifically are 304-689.
# Staying inside that band keeps prompt length in distribution.
TARGET_CHARS = 430
MAX_CHARS = 720
MIN_CHARS = 200


def article_of(unit_id: str) -> str:
    return unit_id.split(".")[0]


def para_index(unit_id: str) -> int:
    return int(unit_id.split(".")[1])


def greedy_pack(articles: list[str], lengths: dict[str, int],
                reverse: bool = False) -> list[list[str]]:
    """Pack an ordered article list into contiguous runs near TARGET_CHARS.

    `reverse` scans from the end, which moves every boundary without ever
    producing a run that is out of document order -- each run is reversed back
    before it is returned.
    """
    seq = list(reversed(articles)) if reverse else list(articles)
    runs, cur, cur_len = [], [], 0
    for a in seq:
        la = lengths[a]
        # Close the current run if adding this article overshoots and what we
        # already have is big enough to stand alone.
        if cur and (cur_len + la > MAX_CHARS or
                    (cur_len >= TARGET_CHARS and cur_len + la > TARGET_CHARS + 120)):
            runs.append(cur)
            cur, cur_len = [], 0
        cur.append(a)
        cur_len += la
    if cur:
        # A trailing run below MIN_CHARS is merged back into the previous one
        # rather than shipped short -- unless that would push the merged run
        # over MAX_CHARS, in which case a slightly short passage is the lesser
        # evil (the test set's shortest is 304 chars, its longest 1693, so
        # overshooting the cap is the more visible distribution error).
        prev_len = sum(lengths[a] for a in runs[-1]) if runs else 0
        if runs and cur_len < MIN_CHARS and prev_len + cur_len <= MAX_CHARS:
            runs[-1].extend(cur)
        else:
            runs.append(cur)
    if reverse:
        runs = [list(reversed(r)) for r in reversed(runs)]
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", type=Path, default=SCRIPT_DIR / "data/udhr_units.json")
    ap.add_argument("--alignment", type=Path, default=SCRIPT_DIR / "data/udhr_alignment.json")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "data/passages.json")
    args = ap.parse_args()

    units = json.loads(args.units.read_text(encoding="utf-8"))
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    langs = sorted(units)

    excluded = set(alignment["used_articles"])
    eng = units["eng"]

    by_article: dict[str, list[str]] = collections.defaultdict(list)
    for uid in eng:
        by_article[article_of(uid)].append(uid)
    for a in by_article:
        by_article[a].sort(key=para_index)

    free = sorted((a for a in by_article if a not in excluded), key=int)
    lengths = {a: sum(len(eng[u]) for u in by_article[a]) for a in free}
    print(f"excluded articles (test set touches them): "
          f"{', '.join(sorted(excluded, key=lambda x: (0 if x == 'P' else int(x))))}")
    print(f"free articles: {len(free)}  ({sum(lengths.values()):,} chars of English)")

    # Consistency: every language must have exactly the same paragraph ids for
    # every free article, or a passage is not the same passage across languages.
    for lang in langs:
        for a in free:
            want = by_article[a]
            got = sorted((u for u in units[lang] if article_of(u) == a), key=para_index)
            if got != want:
                raise SystemExit(
                    f"error: {lang} article {a} has units {got}, English has {want}. "
                    f"Passages would not be parallel; refusing to build.")
    print(f"parallelism check: all {len(langs)} languages have identical unit ids "
          f"for all {len(free)} free articles")

    packings = {
        "fwd": greedy_pack(free, lengths, reverse=False),
        "rev": greedy_pack(free, lengths, reverse=True),
    }

    passages, seen = [], {}
    for pack_name, runs in packings.items():
        for run in runs:
            key = tuple(run)
            if key in seen:
                seen[key]["packings"].append(pack_name)
                continue
            unit_ids = [u for a in run for u in by_article[a]]
            rendering = {
                lang: " ".join(units[lang][u] for u in unit_ids) for lang in langs
            }
            p = {
                "passage_id": f"udhr_p{len(passages) + 1:03d}",
                "articles": run,
                "unit_ids": unit_ids,
                "packings": [pack_name],
                "n_chars_eng": len(rendering["eng"]),
                "text": rendering,
            }
            passages.append(p)
            seen[key] = p

    print(f"\npackings: fwd={len(packings['fwd'])} runs, rev={len(packings['rev'])} runs "
          f"-> {len(passages)} distinct passages")
    print(f"\n{'id':14s} {'articles':22s} {'chars':>5s} {'packing':10s} first words")
    print("-" * 100)
    for p in passages:
        arts = "+".join(p["articles"])
        print(f"{p['passage_id']:14s} {arts:22s} {p['n_chars_eng']:5d} "
              f"{','.join(p['packings']):10s} {p['text']['eng'][:44]}")

    # Article coverage: how many passages each article lands in.
    cover = collections.Counter(a for p in passages for a in p["articles"])
    print(f"\narticle coverage: min {min(cover.values())}, max {max(cover.values())}, "
          f"{sum(1 for a in free if cover[a] == 0)} uncovered")

    lens = [p["n_chars_eng"] for p in passages]
    print(f"passage length (eng chars): min {min(lens)}, median "
          f"{sorted(lens)[len(lens) // 2]}, max {max(lens)}")

    # Cross-language length ratios, a cheap smell test for a truncated or
    # padded translation in the corpus.
    print(f"\nrendering length relative to English (mean over passages):")
    ratios = {
        lang: sum(len(p["text"][lang]) / p["n_chars_eng"] for p in passages) / len(passages)
        for lang in langs
    }
    for lang in sorted(ratios, key=lambda l: -ratios[l]):
        bar = "#" * round(ratios[lang] * 20)
        print(f"  {lang:5s} {ratios[lang]:5.2f} {bar}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "n_passages": len(passages),
            "languages": langs,
            "excluded_articles": sorted(excluded, key=lambda x: (0 if x == "P" else int(x))),
            "free_articles": free,
            "passages": passages,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\nNext: python3 data/udhr/export_question_jobs.py")


if __name__ == "__main__":
    main()
