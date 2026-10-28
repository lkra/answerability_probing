#!/usr/bin/env python3
"""Sample the TyDi / BELEBELE / UDHR mix that trains Adapter 2 (the defaults are that recipe).

Each pool is first capped at --max-per-completion rows per (question_lang, completion), in file
order (UDHR: 43,008 -> 6,068 rows). TyDi and BELEBELE are sampled uniformly, stratified by question
language; UDHR is sampled per rendered passage (item_id, context_lang), keeping a random
--udhr-min-langs..--udhr-max-langs subset of its question languages. Targets 24,000 / 30,000 /
6,000, seed 3407. Fails if any question language has fewer than --min-distinct-ratio distinct
completions per row.

Inputs: data/built/qa_context_tydi_train.jsonl, data/built/belebele_qa_context_train.jsonl,
data/udhr/data/udhr_qa_train_answerable_ctx25.jsonl.
Output: data/built/qa_context_v4_train.jsonl, qa_context_v4_stats.json

    python3 data/mix/build_qa_context_mix.py
"""

import argparse
import collections
import json
import os
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent  # data/


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict], key_fn, target: int, rng: random.Random) -> list[dict]:
    """Uniform random sample of `target` rows, allocated proportionally across
    key_fn(row) groups (largest-remainder method) so each group's share of the
    source is preserved rather than left to chance."""
    if target >= len(rows):
        return list(rows)

    groups = collections.defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    quotas = {}
    remainders = []
    for key, group in groups.items():
        exact = target * len(group) / len(rows)
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))
    shortfall = target - sum(quotas.values())
    remainders.sort(reverse=True)
    for _, key in remainders[:shortfall]:
        quotas[key] += 1

    out = []
    for key, group in groups.items():
        out.extend(rng.sample(group, min(quotas[key], len(group))))
    return out


def udhr_passage_subset_sample(rows: list[dict], target: int, rng: random.Random,
                                min_langs: int = 6, max_langs: int = 20) -> list[dict]:
    """Group by (item_id, context_lang) -- one group per rendered passage --
    and keep a random-sized, random subset of each group's question-language
    rows. Then trim or top up uniformly at random so the total lands exactly
    on `target`, since the per-group random sizes won't sum to it exactly."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["item_id"], r["context_lang"])].append(r)

    kept, leftover = [], []
    for group in groups.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        k = rng.randint(min(min_langs, len(shuffled)), min(max_langs, len(shuffled)))
        kept.extend(shuffled[:k])
        leftover.extend(shuffled[k:])

    if len(kept) > target:
        rng.shuffle(kept)
        kept = kept[:target]
    elif len(kept) < target:
        rng.shuffle(leftover)
        kept.extend(leftover[: target - len(kept)])
    return kept


def lang_counts(rows: list[dict]) -> dict:
    return dict(sorted(collections.Counter(r["question_lang"] for r in rows).items()))


def cap_by_completion(rows: list[dict], max_per: int) -> list[dict]:
    """Keep at most `max_per` rows per (question_lang, completion), first come
    first kept in file order. Deterministic, so a rebuild from the same pool
    file removes the same rows."""
    seen = collections.Counter()
    kept = []
    for r in rows:
        key = (r["question_lang"], r["completion"])
        if seen[key] < max_per:
            seen[key] += 1
            kept.append(r)
    return kept


def distinct_ratio_by_lang(rows: list[dict]) -> dict[str, float]:
    """Distinct completions / rows, per question language -- the diversity
    measure used to check the mix (e.g. bho 6% in an uncapped mix, 25% after the cap)."""
    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r["question_lang"]].append(r["completion"])
    return {lang: round(len(set(c)) / len(c), 4) for lang, c in sorted(by_lang.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tydi", type=Path, default=REPO_ROOT / "built/qa_context_tydi_train.jsonl")
    ap.add_argument("--belebele", type=Path, default=REPO_ROOT / "built/belebele_qa_context_train.jsonl")
    ap.add_argument("--udhr-pool", type=Path,
                     default=REPO_ROOT / "udhr/data/udhr_qa_train_answerable_ctx25.jsonl")
    ap.add_argument("--tydi-target", type=int, default=24000)
    ap.add_argument("--belebele-target", type=int, default=30000)
    ap.add_argument("--udhr-target", type=int, default=6000)
    ap.add_argument("--udhr-min-langs", type=int, default=6)
    ap.add_argument("--udhr-max-langs", type=int, default=20)
    ap.add_argument("--max-per-completion", type=int, default=4,
                     help="cap each source pool at this many rows per distinct "
                          "(question_lang, completion) before sampling (V4: 4). "
                          "Pass 0 to disable.")
    ap.add_argument("--min-distinct-ratio", type=float, default=0.35,
                     help="fail the build if any question language in the finished mix "
                          "has fewer distinct completions per row than this (V4: 0.35).")
    ap.add_argument("--exclude-lang", nargs="*", default=[],
                     help="question languages to drop from every pool before sampling "
                          "(ISO 639-3, e.g. yor). V4 excluded none; note MADLAD Yoruba is "
                          "already absent from the TyDi file, this is for the other sources.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "built/qa_context_v4_train.jsonl")
    ap.add_argument("--stats-out", type=Path, default=REPO_ROOT / "built/qa_context_v4_stats.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    tydi = load_jsonl(args.tydi)
    belebele = load_jsonl(args.belebele)
    udhr_pool = load_jsonl(args.udhr_pool)
    print(f"Loaded  tydi={len(tydi):,}  belebele={len(belebele):,}  udhr_pool={len(udhr_pool):,}")

    for r in tydi:
        r.setdefault("source", "tydi")

    pool_sizes_raw = {"tydi": len(tydi), "belebele": len(belebele), "udhr": len(udhr_pool)}
    if args.exclude_lang:
        drop = set(args.exclude_lang)
        tydi = [r for r in tydi if r["question_lang"] not in drop]
        belebele = [r for r in belebele if r["question_lang"] not in drop]
        udhr_pool = [r for r in udhr_pool if r["question_lang"] not in drop]
        print(f"Excluded {sorted(drop)}: tydi={len(tydi):,}  belebele={len(belebele):,}  udhr_pool={len(udhr_pool):,}")

    if args.max_per_completion:
        tydi = cap_by_completion(tydi, args.max_per_completion)
        belebele = cap_by_completion(belebele, args.max_per_completion)
        udhr_pool = cap_by_completion(udhr_pool, args.max_per_completion)
        print(f"Capped at {args.max_per_completion}/completion: "
              f"tydi={len(tydi):,}  belebele={len(belebele):,}  udhr_pool={len(udhr_pool):,}")
    pool_sizes_sampled_from = {"tydi": len(tydi), "belebele": len(belebele), "udhr": len(udhr_pool)}

    tydi_sampled = stratified_sample(tydi, lambda r: r["question_lang"], args.tydi_target, rng)
    belebele_sampled = stratified_sample(belebele, lambda r: r["question_lang"], args.belebele_target, rng)
    udhr_sampled = udhr_passage_subset_sample(
        udhr_pool, args.udhr_target, rng, args.udhr_min_langs, args.udhr_max_langs)

    print(f"Sampled tydi={len(tydi_sampled):,}  belebele={len(belebele_sampled):,}  udhr={len(udhr_sampled):,}")

    combined = tydi_sampled + belebele_sampled + udhr_sampled
    rng.shuffle(combined)

    ids = [r["id"] for r in combined]
    assert len(ids) == len(set(ids)), "duplicate row id across sources"

    distinct = distinct_ratio_by_lang(combined)
    lo, hi = min(distinct, key=distinct.get), max(distinct, key=distinct.get)
    print(f"Distinct completions per row: min {lo} {distinct[lo]:.1%}, max {hi} {distinct[hi]:.1%}")
    if args.min_distinct_ratio is not None:
        below = {l: v for l, v in distinct.items() if v < args.min_distinct_ratio}
        if below:
            raise SystemExit(f"error: --min-distinct-ratio {args.min_distinct_ratio} tripped for "
                             + ", ".join(f"{l} ({v:.1%})" for l, v in below.items())
                             + " -- raise --max-per-completion's bite or rebalance the sources")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_rows": len(combined),
        "by_source": {
            "tydi": len(tydi_sampled),
            "belebele": len(belebele_sampled),
            "udhr": len(udhr_sampled),
        },
        "share_by_source": {
            "tydi": round(len(tydi_sampled) / len(combined), 4),
            "belebele": round(len(belebele_sampled) / len(combined), 4),
            "udhr": round(len(udhr_sampled) / len(combined), 4),
        },
        "question_lang_counts": {
            "tydi": lang_counts(tydi_sampled),
            "belebele": lang_counts(belebele_sampled),
            "udhr": lang_counts(udhr_sampled),
            "combined": lang_counts(combined),
        },
        "udhr_unique_passage_renderings_used": len({(r["item_id"], r["context_lang"]) for r in udhr_sampled}),
        "distinct_completion_ratio_by_lang": distinct,
        "pool_sizes_raw": pool_sizes_raw,
        "pool_sizes_sampled_from": pool_sizes_sampled_from,
        "max_per_completion": args.max_per_completion,
        "min_distinct_ratio": args.min_distinct_ratio,
        "exclude_lang": sorted(args.exclude_lang),
        "inputs": {k: os.path.relpath(v, REPO_ROOT.parent) for k, v in
                   {"tydi": args.tydi, "belebele": args.belebele, "udhr_pool": args.udhr_pool}.items()},
        "seed": args.seed,
    }
    args.stats_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nWrote {len(combined):,} rows to {args.out}")
    print(f"Shares: tydi={stats['share_by_source']['tydi']:.1%}  "
          f"belebele={stats['share_by_source']['belebele']:.1%}  "
          f"udhr={stats['share_by_source']['udhr']:.1%}")
    print(f"Wrote stats to {args.stats_out}")


if __name__ == "__main__":
    main()
