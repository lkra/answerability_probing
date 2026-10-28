#!/usr/bin/env python3
"""Render the UDHR qa-context rows in the organisers' format ({id, prompt, completion, task,
question_lang} plus provenance; ids from 500001). Every item is rendered with English and
--ctx-answerable - 1 other context languages, rotating so all 25 appear about equally, and asked in
all 24 question languages. Transplanted unanswerables are admitted up to the row-level share
--unanswerable-target (0.203). --answerable-only emits answerable rows only; --dev-articles holds
articles out as a dev split.

Output: data/udhr_qa_train.jsonl, data/udhr_qa_stats.json (udhr_qa_dev.jsonl with --dev-articles)
"""

import argparse
import collections
import contextlib
import json
import re
import unicodedata
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

_WS_RE = re.compile(r"\s+")


def clean(s: str) -> str:
    """Same rule as data/tydi/build_qa_context_train.py's clean_text().

    Real newlines are the thing being guarded against: the organizers' prompts
    have none, so one carried through from a passage or a localised answer
    would be a train/test format mismatch.
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def rotate(seq: list, start: int, k: int) -> list:
    """k items from seq starting at `start`, wrapping."""
    return [seq[(start + i) % len(seq)] for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=SCRIPT_DIR / "data/items.json")
    ap.add_argument("--transplants", type=Path, default=SCRIPT_DIR / "data/transplants.json")
    ap.add_argument("--passages", type=Path, default=SCRIPT_DIR / "data/passages.json")
    ap.add_argument("--localised", type=Path, default=SCRIPT_DIR / "data/localised.json")
    ap.add_argument("--templates", type=Path,
                    default=REPO_ROOT / "qa_context_templates.json")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "data/udhr_qa_train.jsonl")
    ap.add_argument("--stats-out", type=Path, default=SCRIPT_DIR / "data/udhr_qa_stats.json")
    ap.add_argument("--ctx-answerable", type=int, default=8,
                    help="Context languages per answerable authored item (English + N-1 others).")
    ap.add_argument("--ctx-unanswerable", type=int, default=4,
                    help="Context languages per authored unanswerable item. Lower than "
                         "--ctx-answerable on purpose: the unanswerable row budget is "
                         "fixed by --unanswerable-target, so every row spent on an "
                         "authored unanswerable is a row not spent on a transplant, and "
                         "transplants are the ones whose label is mechanical. 4 splits "
                         "that budget roughly evenly between the two kinds.")
    ap.add_argument("--ctx-transplant", type=int, default=1,
                    help="Context languages per transplanted unanswerable item.")
    ap.add_argument("--unanswerable-target", type=float, default=0.203,
                    help="Row-level unanswerable share to hit (test set: 0.203).")
    ap.add_argument("--start-index", type=int, default=500001,
                    help="Row-id counter start. Default is far above "
                         "qa_context_train.jsonl's range so the two files can be "
                         "concatenated without id collisions.")
    ap.add_argument("--dev-articles", nargs="*", default=[],
                    help="Hold these UDHR articles out as a dev split, e.g. "
                         "--dev-articles 28 29 30. The split is by ARTICLE, not by "
                         "passage: the two packings overlap, so splitting on passages "
                         "would put the same article on both sides. Passages that "
                         "straddle the boundary are dropped from both splits rather "
                         "than assigned to one.")
    ap.add_argument("--dev-out", type=Path, default=SCRIPT_DIR / "data/udhr_qa_dev.jsonl")
    ap.add_argument("--exclusions", type=Path,
                    default=SCRIPT_DIR / "data/passage_exclusions.json",
                    help="(passage, context language) renderings to withhold because the "
                         "official UN text is defective there. See that file for the "
                         "reason attached to each one.")
    ap.add_argument("--drop-near-duplicates", action="store_true",
                    help="Skip items merge_questions.py flagged as near-duplicates of an "
                         "earlier item (>0.70 English question similarity). They arise "
                         "from the two overlapping passage packings; the passages differ "
                         "so they are not strictly redundant, but they reduce effective "
                         "question diversity.")
    ap.add_argument("--answerable-only", action="store_true",
                    help="Emit only answerable rows: no authored unanswerable items, no "
                         "transplants, no abstention literals anywhere in the file. For "
                         "the split setup where the generator is trained to answer every "
                         "question and a separate probe decides abstention. "
                         "--unanswerable-target and --ctx-unanswerable/--ctx-transplant "
                         "are ignored.")
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding="utf-8"))["items"]
    passages = {p["passage_id"]: p
                for p in json.loads(args.passages.read_text(encoding="utf-8"))["passages"]}
    localised = json.loads(args.localised.read_text(encoding="utf-8"))
    templates = json.loads(args.templates.read_text(encoding="utf-8"))["templates"]
    transplants = []
    if args.transplants.exists():
        transplants = json.loads(args.transplants.read_text(encoding="utf-8"))["transplants"]

    exclusions = set()
    if args.exclusions.exists():
        for e in json.loads(args.exclusions.read_text(encoding="utf-8"))["exclusions"]:
            exclusions.add((e["passage_id"], e["lang"]))

    question_langs = sorted(templates)                      # 24
    context_langs = sorted(passages[next(iter(passages))]["text"])  # 25
    print(f"{len(question_langs)} question languages, {len(context_langs)} context languages")

    # A context-language count above the number of languages that exist makes the
    # rotation wrap and hand the same language to one item twice, which produces
    # rows that are byte-identical in prompt and completion and differ only in id.
    # Silent duplicate training data is the worst outcome here, so this is an
    # error rather than a clamp: the caller asked for something impossible and
    # should be told the ceiling rather than quietly given something else.
    for flag, value in (("--ctx-answerable", args.ctx_answerable),
                        ("--ctx-unanswerable", args.ctx_unanswerable),
                        ("--ctx-transplant", args.ctx_transplant)):
        if value > len(context_langs):
            raise SystemExit(
                f"error: {flag}={value} exceeds the {len(context_langs)} context "
                f"languages that exist. Above that the rotation repeats a language "
                f"within one item and emits duplicate rows. Maximum is "
                f"{len(context_langs)}.")
        if value < 1:
            raise SystemExit(f"error: {flag}={value} must be at least 1.")

    by_item = {i["item_id"]: i for i in items}

    def localised_pair(item_id: str, qlang: str) -> tuple[str | None, str | None, dict]:
        """(question, answer, grounding info) in `qlang` for an authored item."""
        it = by_item[item_id]
        if qlang == "eng":
            return it["question"], it["answer"], {
                "answer_exact_span": it["answer_is_span"], "answer_coverage": None}
        rec = localised.get(qlang, {}).get(item_id)
        if rec is None:
            return None, None, {}
        return rec["question"], rec["answer"], {
            "answer_exact_span": rec["exact_span"],
            "answer_coverage": rec["coverage"],
            "localisation_uncertain": rec["uncertain"] or None,
        }

    # ---- how many transplant rows fit under the target share -----------------
    n_ans = sum(1 for i in items if i["answerable"])
    n_unans = len(items) - n_ans
    ans_rows = n_ans * args.ctx_answerable * len(question_langs)

    if args.answerable_only:
        # Generator-only mode: the model is trained to answer every question and
        # a separate probe decides abstention, so no unanswerable row belongs in
        # this file at all. The whole target-share mechanism below is moot --
        # nothing is competing for a fixed unanswerable budget.
        n_unans_used, n_transplants = 0, 0
        unans_rows = 0
        print(f"\nANSWERABLE-ONLY: {n_ans} answerable items -> {ans_rows:,} rows")
        print(f"  excluded: {n_unans} authored unanswerable items, "
              f"{len(transplants)} verified transplants")
        print(f"  the unanswerable-share target is not applied in this mode")
        print(f"  NOTE: --ctx-answerable is unconstrained in this mode (currently "
              f"{args.ctx_answerable}; it defaults to 8 for comparability with the full "
              f"build, where it is held down to leave room for transplants under the "
              f"20.3% target). Raising it needs no new localisation -- questions and "
              f"answers are localised per item, not per row. "
              f"12 -> {n_ans * 12 * len(question_langs):,} rows, "
              f"{len(context_langs)} (every context language) -> "
              f"{n_ans * len(context_langs) * len(question_langs):,}.")
    else:
        n_unans_used = n_unans
        unans_rows = n_unans * args.ctx_unanswerable * len(question_langs)
        t = args.unanswerable_target
        budget = (t * ans_rows - (1 - t) * unans_rows) / (1 - t)
        rows_per_transplant = args.ctx_transplant * len(question_langs)
        n_transplants_wanted = max(int(round(budget / rows_per_transplant)), 0)
        n_transplants = min(n_transplants_wanted, len(transplants))

        print(f"\nauthored items: {len(items)} ({n_ans} answerable, {n_unans} unanswerable)")
        print(f"  -> {ans_rows:,} answerable rows + {unans_rows:,} unanswerable rows")
        print(f"transplants available: {len(transplants)}; "
              f"target share {t:.1%} leaves room for {n_transplants_wanted}; "
              f"using {n_transplants}")
        if n_transplants_wanted > len(transplants):
            print(f"  NOTE: {n_transplants_wanted - len(transplants)} short -- the final "
                  f"unanswerable share will land below target")
        if len(transplants) > n_transplants:
            print(f"  NOTE: {len(transplants) - n_transplants} verified transplants held "
                  f"back to keep the share on target (raise --unanswerable-target to use them)")

    # ---- assemble the work list ---------------------------------------------
    # (kind, item_id, question_item_id, passage_id, answerable, n_ctx, stream_i)
    #
    # stream_i counts within a *stream* (answerable / authored-unanswerable /
    # transplant), not globally, and it is what drives context-language rotation.
    # A single global counter interleaves the three streams and leaves each
    # language with a different answerable:unanswerable ratio -- 12% to 26% in
    # the first build. That is a spurious correlation between context language
    # and abstention rate, which is precisely the thing this dataset exists to
    # let us measure, so it must not be an artefact of the row builder. Rotating
    # each stream independently keeps every language's share within a couple of
    # points of the global one.
    work = []
    stream_counts: collections.Counter = collections.Counter()
    n_dup_skipped = 0
    for it in items:
        if args.answerable_only and not it["answerable"]:
            continue
        if args.drop_near_duplicates and "near_duplicate_of" in it:
            n_dup_skipped += 1
            continue
        stream = "ans" if it["answerable"] else "unans"
        n_ctx = args.ctx_answerable if it["answerable"] else args.ctx_unanswerable
        work.append(("authored", it["item_id"], it["item_id"], it["passage_id"],
                     it["answerable"], n_ctx, stream_counts[stream]))
        stream_counts[stream] += 1
    for tr in transplants[:n_transplants]:
        work.append(("transplant", tr["transplant_id"], tr["item_id"],
                     tr["target_passage_id"], False, args.ctx_transplant,
                     stream_counts["transplant"]))
        stream_counts["transplant"] += 1

    # ---- optional article-level train/dev split ------------------------------
    dev_articles = set(args.dev_articles)
    if dev_articles:
        unknown = dev_articles - {a for p in passages.values() for a in p["articles"]}
        if unknown:
            raise SystemExit(f"error: --dev-articles names articles not in any passage: "
                             f"{sorted(unknown)}")

    def split_of(passage_id: str, q_item_id: str) -> str | None:
        """'train', 'dev', or None to drop.

        A transplant touches two passages (the question's home and the target),
        so both have to fall on the same side or it is dropped -- otherwise a dev
        question would appear in a training row.
        """
        if not dev_articles:
            return "train"
        sides = set()
        for pid in {passage_id, by_item[q_item_id]["passage_id"]}:
            arts = set(passages[pid]["articles"])
            if arts <= dev_articles:
                sides.add("dev")
            elif not (arts & dev_articles):
                sides.add("train")
            else:
                return None  # passage straddles the boundary
        return sides.pop() if len(sides) == 1 else None

    # ---- render --------------------------------------------------------------
    n = args.start_index
    written = 0
    per_split = collections.Counter()
    split_unans = collections.Counter()
    skipped = collections.Counter()
    stats = {
        "by_question_lang": collections.Counter(),
        "by_context_lang": collections.Counter(),
        "by_pair": collections.Counter(),
        "unanswerable_by_context_lang": collections.Counter(),
        "unanswerable_by_question_lang": collections.Counter(),
        "by_origin": collections.Counter(),
        "by_question_type": collections.Counter(),
        "grounded_answers": 0,
        "answer_rows": 0,
    }

    # Rotation offsets are derived from position in the work list, so the
    # assignment is deterministic and every context language gets its turn.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    non_eng_ctx = [c for c in context_langs if c != "eng"]

    with contextlib.ExitStack() as stack:
        handles = {"train": stack.enter_context(args.out.open("w", encoding="utf-8"))}
        if dev_articles:
            handles["dev"] = stack.enter_context(args.dev_out.open("w", encoding="utf-8"))

        for (kind, row_item_id, q_item_id, passage_id, answerable, n_ctx, s_i) in work:
            split = split_of(passage_id, q_item_id)
            if split is None:
                skipped["straddles_dev_boundary"] += 1
                continue
            f = handles[split]
            src = by_item[q_item_id]
            passage_texts = passages[passage_id]["text"]
            if n_ctx > 1:
                # English always present, mirroring the test set (every one of its
                # 100 items has exactly one English-context rendering), plus a
                # rotating slice of the rest. Stepping by (n_ctx - 1) makes the
                # stream sweep the language list without gaps or repeats.
                ctxs = ["eng"] + rotate(non_eng_ctx, s_i * (n_ctx - 1), n_ctx - 1)
            else:
                # Single-context items (transplants) rotate over all 25 including
                # English, so they are not silently absent from English-context rows.
                ctxs = [context_langs[s_i % len(context_langs)]]

            for qlang in question_langs:
                template = templates[qlang]
                question, answer, ground = localised_pair(q_item_id, qlang)
                if not question:
                    skipped["missing_localised_question"] += 1
                    continue
                if answerable:
                    if not answer:
                        skipped["missing_localised_answer"] += 1
                        continue
                    completion = clean(answer)
                else:
                    # Exact, not silver: the organizers' own abstention literal.
                    completion = template["not_answerable"]

                for clang in ctxs:
                    if (passage_id, clang) in exclusions:
                        skipped["excluded_defective_passage"] += 1
                        continue
                    passage = clean(passage_texts[clang])
                    prompt = template["sep"].join([
                        template["intro"],
                        passage,
                        template["q_prefix"] + clean(question),
                        template["instruction"],
                    ])
                    row = {
                        "id": f"qa-context_{n}_{qlang}_{clang}",
                        "prompt": prompt,
                        "completion": completion,
                        "task": "qa-context",
                        "question_lang": qlang,
                        # provenance
                        "source": "udhr",
                        "item_id": row_item_id,
                        "question_item_id": q_item_id,
                        "passage_id": passage_id,
                        "articles": passages[passage_id]["articles"],
                        "context_lang": clang,
                        "answerable": answerable,
                        "origin": kind,
                        "question_type": src["question_type"] if kind == "authored"
                                         else "transplanted_unanswerable",
                        "split": split,
                    }
                    if answerable:
                        row.update({k: v for k, v in ground.items() if v is not None})
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                    written += 1
                    per_split[split] += 1
                    if not answerable:
                        split_unans[split] += 1

                    stats["by_question_lang"][qlang] += 1
                    stats["by_context_lang"][clang] += 1
                    stats["by_pair"][f"{clang}->{qlang}"] += 1
                    stats["by_origin"][kind] += 1
                    stats["by_question_type"][row["question_type"]] += 1
                    if not answerable:
                        stats["unanswerable_by_context_lang"][clang] += 1
                        stats["unanswerable_by_question_lang"][qlang] += 1
                    else:
                        stats["answer_rows"] += 1
                        if ground.get("answer_exact_span"):
                            stats["grounded_answers"] += 1

    # ---- report --------------------------------------------------------------
    n_un = sum(stats["unanswerable_by_context_lang"].values())
    print(f"\nwrote {written:,} rows to {args.out}")
    if args.answerable_only:
        print(f"  unanswerable: {n_un:,} (answerable-only mode; must be 0)")
    else:
        print(f"  unanswerable: {n_un:,} ({n_un / max(written, 1):.1%})   [test set: 20.3%]")
    print(f"  answers reusing official UDHR wording verbatim: "
          f"{stats['grounded_answers']:,}/{stats['answer_rows']:,} "
          f"({stats['grounded_answers'] / max(stats['answer_rows'], 1):.0%})")
    print(f"  origin: {dict(stats['by_origin'])}")
    if args.drop_near_duplicates:
        print(f"  near-duplicate items skipped: {n_dup_skipped}")
    if dev_articles:
        print(f"\ntrain/dev split, held out by article ({', '.join(sorted(dev_articles, key=int))}):")
        for split in ("train", "dev"):
            tot = per_split[split]
            print(f"  {split:5s} {tot:8,d} rows  "
                  f"{split_unans[split]:6,d} unanswerable ({split_unans[split] / max(tot, 1):.1%})")
        print(f"  dropped for straddling the boundary: "
              f"{skipped.get('straddles_dev_boundary', 0)} items")
        # The transplant budget is computed globally and then filtered by the
        # same-side rule, so a small dev side keeps far fewer transplants than
        # its share of the rows -- which drags its unanswerable rate down. Say so
        # rather than letting an 11% dev abstention rate be read as comparable to
        # the test set's 20.3%.
        for split in ("train", "dev"):
            tot = per_split[split]
            if tot and abs(split_unans[split] / tot - args.unanswerable_target) > 0.05:
                print(f"  WARNING: the {split} split's unanswerable share "
                      f"({split_unans[split] / tot:.1%}) is far from the "
                      f"{args.unanswerable_target:.1%} target. Abstention numbers "
                      f"measured on it are not comparable to the test set's without "
                      f"reweighting.")
        print(f"  train file: {args.out}")
        print(f"  dev file:   {args.dev_out}")

    print(f"\nlanguage-pair grid: {len(stats['by_pair'])} of "
          f"{len(context_langs) * len(question_langs)} (context -> question) cells populated")
    cell = sorted(stats["by_pair"].values())
    print(f"  rows per populated cell: min {cell[0]}, median {cell[len(cell) // 2]}, "
          f"max {cell[-1]}")

    print(f"\n{'lang':6s} {'as question':>12s} {'as context':>11s} "
          f"{'unans (ctx)':>12s}")
    print("-" * 46)
    for lang in sorted(set(question_langs) | set(context_langs)):
        q = stats["by_question_lang"].get(lang, 0)
        c = stats["by_context_lang"].get(lang, 0)
        u = stats["unanswerable_by_context_lang"].get(lang, 0)
        print(f"{lang:6s} {q:12,d} {c:11,d} {u:6,d} {u / max(c, 1):5.0%}")

    if skipped:
        print("\nskipped:")
        for k, v in skipped.items():
            print(f"  {k:32s} {v}")

    args.stats_out.write_text(json.dumps({
        "n_rows": written,
        "n_unanswerable": n_un,
        "unanswerable_share": round(n_un / max(written, 1), 4),
        "answerable_only": args.answerable_only,
        "n_authored_items": len(items),
        "n_authored_items_used": len(items) - (0 if not args.answerable_only else n_unans),
        "n_transplant_items": n_transplants,
        "n_transplants_available": len(transplants),
        "dev_articles": sorted(dev_articles, key=int) if dev_articles else [],
        "rows_by_split": dict(per_split),
        "unanswerable_by_split": dict(split_unans),
        "grounded_answer_rows": stats["grounded_answers"],
        "answer_rows": stats["answer_rows"],
        "design": {
            "ctx_answerable": args.ctx_answerable,
            "ctx_unanswerable": args.ctx_unanswerable,
            "ctx_transplant": args.ctx_transplant,
            "unanswerable_target": args.unanswerable_target,
        },
        **{k: dict(v) for k, v in stats.items() if isinstance(v, collections.Counter)},
        "skipped": dict(skipped),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.stats_out}")
    print("\nNext: python3 data/udhr/verify.py")


if __name__ == "__main__":
    main()
