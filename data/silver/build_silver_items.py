"""Collapse the 8,640 qa-context test rows into their 100 (source text, question) items: rows are
clustered by English question string and split by English-context passage (one question is asked
against two passages). Each item records its canonical English passage and question and every test
row id it covers.

Output: silver_items.json
"""

import argparse
import collections
import json
import pathlib

QUESTION_MARKER = "The question is:"


def parse_prompt(prompt):
    """Split a qa-context prompt into (passage, question).

    Prompts are '<localised intro>:\\n\\n<passage>\\n\\n<marker> <question>\\n\\n<instruction>',
    where the newlines are literal backslash-n sequences rather than real newlines.
    """
    blocks = prompt.split("\\n\\n")
    passage = blocks[1]
    question = next(b for b in blocks if b.startswith(QUESTION_MARKER))
    return passage, question[len(QUESTION_MARKER):].strip()


def build_items(rows):
    qa = [r for r in rows if r["task"] == "qa-context"]

    # Every row for a given (integer id, context language) shares one passage and one
    # question -- only the instruction wrapper and question language are localised.
    renderings = collections.defaultdict(dict)
    for r in qa:
        _, n, qlang, clang = r["id"].split("_")
        renderings[(int(n), clang)][qlang] = r["id"]

    # The English-question row of each rendering is the one we read the English
    # question off; its passage is in the context language.
    eng_rows = {}
    for r in qa:
        _, n, qlang, clang = r["id"].split("_")
        if qlang == "eng":
            eng_rows[(int(n), clang)] = r

    # Cluster renderings by English question text, then split by English-context passage.
    clusters = collections.defaultdict(list)
    for key, r in eng_rows.items():
        passage, question = parse_prompt(r["prompt"])
        clusters[question].append((key, passage))

    items = []
    for question, members in sorted(clusters.items()):
        eng_members = [(k, p) for k, p in members if k[1] == "eng"]
        if len(eng_members) <= 1:
            groups = [(eng_members, members)]
        else:
            # Same question against two different source texts: one item per English
            # passage, with the non-English renderings assigned by passage similarity.
            groups = [([em], [em]) for em in eng_members]
            for k, p in members:
                if k[1] == "eng":
                    continue
                best = max(eng_members, key=lambda em: _overlap(p, em[1]))
                next(g for g in groups if g[0][0] is best)[1].append((k, p))

        for eng_members_g, all_members in groups:
            (eng_key, eng_passage), = eng_members_g
            row_ids = sorted(
                rid
                for key, _ in all_members
                for rid in renderings[key].values()
            )
            items.append({
                "item_id": f"item_{len(items) + 1:03d}",
                "english_passage": eng_passage,
                "english_question": question,
                "english_row_id": renderings[eng_key]["eng"],
                "context_langs": sorted({k[1] for k, _ in all_members}),
                "n_renderings": len(all_members),
                "row_ids": row_ids,
            })
    return items


def _overlap(a, b):
    """Crude character-trigram overlap, used only to split the single ambiguous cluster."""
    ta = {a[i:i + 3] for i in range(len(a) - 2)}
    tb = {b[i:i + 3] for i in range(len(b) - 2)}
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests-path", default="data/test/tests.jsonl")
    ap.add_argument("--output", default="data/silver/silver_items.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.tests_path, encoding="utf-8")]
    items = build_items(rows)

    n_qa = sum(1 for r in rows if r["task"] == "qa-context")
    covered = sum(len(it["row_ids"]) for it in items)
    assert len(items) == 100, f"expected 100 items, got {len(items)}"
    assert covered == n_qa, f"items cover {covered} rows, expected {n_qa}"
    assert len({rid for it in items for rid in it["row_ids"]}) == n_qa, "duplicate row ids"

    pathlib.Path(args.output).write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(items)} items covering {covered}/{n_qa} qa-context rows -> {args.output}")
    print("renderings per item:", sorted(collections.Counter(
        it["n_renderings"] for it in items).items()))


if __name__ == "__main__":
    main()
