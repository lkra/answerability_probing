#!/usr/bin/env python3
r"""Order a generation row set round-robin over (label x probe-confidence bin x question language),
using both probes' scores, and write ids_<set>_phase1.jsonl (the first --phase1 rows) and
ids_<set>_all.jsonl (all rows, same order). A prefix is cell-balanced: per-bin rates from a partial
run are usable, aggregates need the full run.

    python3 probe/probing/order_generation_rows.py --scores probe/generation/qwen35-9b/probe_{probe}_validation.jsonl \
        --probes tydi,tydi_udhr --set validation --phase1 2000 --out-dir probe/generation/qwen35-9b
"""

import argparse
import collections
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Unanswerable rows carry the result, so they are binned finely and exactly as
# `analyze_abstention.py` reports them -- an ordering built on different edges than the
# analysis would balance cells nobody reads. Answerable rows only feed the mirror-error
# rate, which is one number, so three bins are enough.
UNANS_BINS = [(0, .01), (.01, .05), (.05, .1), (.1, .25), (.25, .5), (.5, 1.01)]
ANS_BINS = [(0, .5), (.5, .9), (.9, 1.01)]


def bin_of(p, answerable):
    for lo, hi in (ANS_BINS if answerable else UNANS_BINS):
        if lo <= p < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "[1.00,1.00]"          # p == 1.0 exactly; folded into the top bin by the caller


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True,
                    help="path to the per-probe score files, with '{probe}' where the probe "
                         "name goes")
    ap.add_argument("--probes", default="tydi,tydi_udhr",
                    help="comma-separated, as they appear in the score filenames")
    ap.add_argument("--set", required=True, help="validation | translation | belebele")
    ap.add_argument("--phase1", type=int, default=2000,
                    help="rows in the phase-1 file. 2,000 lands ~200-250 rows in every "
                         "confidence bin, which is +-6-7pp on a per-bin rate -- ample for a "
                         "41-point trend, and the full run supersedes it anyway.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows, seen = {}, {}
    for probe in args.probes.split(","):
        path = Path(args.scores.replace("{probe}", probe))
        if not path.exists():
            raise SystemExit(f"{path} not found -- run score_bare_cache.py first")
        n = 0
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rows.setdefault(r["id"], r)
            seen.setdefault(r["id"], {})[probe] = r["prob_answerable"]
            n += 1
        print(f"{probe:10s} {n:7,d} scored rows  <- {path.name}")

    ragged = [i for i, s in seen.items() if len(s) != len(args.probes.split(","))]
    if ragged:
        raise SystemExit(f"{len(ragged):,} ids are scored by only some probes (e.g. "
                         f"{ragged[0]}) -- the score files cover different row sets, and an "
                         f"ordering built from them would not be the same experiment per arm")

    # Position under one probe = index within that probe's (label, bin, question language)
    # cell, rows ordered by id so the whole thing is deterministic and re-derivable from the
    # committed score files alone. A row's final key is the *best* position any probe gives
    # it, so a row that is rare for either arm is scheduled early.
    best = {i: None for i in rows}
    for probe in args.probes.split(","):
        cells = collections.defaultdict(list)
        for i, r in rows.items():
            cells[(r["answerable"], bin_of(seen[i][probe], r["answerable"]),
                   r["question_lang"])].append(i)
        for key in cells:
            for pos, i in enumerate(sorted(cells[key])):
                best[i] = pos if best[i] is None else min(best[i], pos)

    order = sorted(rows, key=lambda i: (best[i], i))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n1 = min(args.phase1, len(order))

    for name, ids in (("phase1", order[:n1]), ("all", order)):
        out = args.out_dir / f"ids_{args.set}_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rank, i in enumerate(ids):
                r = rows[i]
                f.write(json.dumps({
                    "id": i, "rank": rank, "answerable": r["answerable"],
                    "question_lang": r["question_lang"], "passage_lang": r["passage_lang"],
                    "p": {pr: round(seen[i][pr], 6) for pr in seen[i]},
                }, ensure_ascii=False) + "\n")
        print(f"wrote {out}  ({len(ids):,} rows)")

    print(f"\nphase-1 coverage ({n1:,} of {len(order):,} rows):")
    head = set(order[:n1])
    for probe in args.probes.split(","):
        counts = collections.Counter(
            bin_of(seen[i][probe], rows[i]["answerable"])
            for i in head if rows[i]["answerable"] == 0)
        total = collections.Counter(
            bin_of(seen[i][probe], rows[i]["answerable"])
            for i in rows if rows[i]["answerable"] == 0)
        print(f"  {probe} (unanswerable rows, phase1 / all):")
        for lo, hi in UNANS_BINS:
            k = f"[{lo:.2f},{hi:.2f})"
            print(f"    {k:16s} {counts.get(k, 0):5,d} / {total.get(k, 0):6,d}")
        nlang = len({rows[i]["question_lang"] for i in head})
        print(f"    {nlang} question languages covered")


if __name__ == "__main__":
    main()
