#!/usr/bin/env python3
"""Derive belebele_answerability_hard_b8.jsonl: swap ("hard") negatives on exactly the items and
languages of the item-matched BELEBELE-8 delete set, so that only the negative construction differs.

Inputs: --hard (build_belebele.py --negatives hard --languages <the 8> --pairs same) and --perlang
(build_belebele.py --negatives delete --delete-align per-language --pairs all). An item is kept when
the per-language set has it as a monolingual pair in all eight languages. The option-dependent
filter is applied at load time by probe/probing/_sources.py, as for the delete set.

    python3 data/belebele/restrict_hard_to_belebele8.py
"""
import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
B8 = ["yor_Latn", "swh_Latn", "ckb_Arab", "hat_Latn", "por_Latn", "eng_Latn", "zho_Hans", "pes_Arab"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hard", type=Path, default=DATA / "built/belebele_answerability_hard8.jsonl")
    ap.add_argument("--perlang", type=Path, default=DATA / "built/belebele_answerability_delete_perlang.jsonl")
    ap.add_argument("--languages", default=",".join(B8))
    ap.add_argument("--output", type=Path, default=DATA / "built/belebele_answerability_hard_b8.jsonl")
    args = ap.parse_args()
    langs = set(args.languages.split(","))

    covered = {}  # item_id -> set of passage languages with a monolingual delete pair
    with open(args.perlang, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["passage_lang"] == r["question_lang"] and r["passage_lang"] in langs:
                covered.setdefault(r["item_id"], set()).add(r["passage_lang"])
    items = {i for i, s in covered.items() if s == langs}

    n = 0
    with open(args.hard, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            if r["item_id"] in items and r["passage_lang"] in langs and r["question_lang"] == r["passage_lang"]:
                out.write(line); n += 1
    print(f"{len(items)} items present in all {len(langs)} languages of the per-language delete set; "
          f"wrote {n} hard rows to {args.output}")


if __name__ == "__main__":
    main()
