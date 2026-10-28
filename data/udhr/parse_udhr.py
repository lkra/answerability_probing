#!/usr/bin/env python3
"""Parse the UDHR XML into paragraph units addressed as <article>.<paragraph> (P.n for the preamble),
identically across languages, and decide which file is each MIST language: where the test set
contains a UDHR passage in that language, the file that reproduces it exactly; otherwise the file
whose script matches the test passages'.

Output: data/udhr_units.json, data/udhr_alignment.json (test items that are UDHR, and their units),
data/udhr_mapping.json
"""

import argparse
import collections
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

NS = "{http://www.unhchr.ch/udhr}"
SEP = "\\n\\n"  # the organizers' literal backslash-n-backslash-n, not a newline

# The 25 context languages the qa-context test set uses (24 question languages
# + fra, which appears only as a passage language).
MIST_CONTEXT_LANGS = [
    "arb", "ben", "bho", "ces", "ckb", "deu", "eng", "fin", "fra", "hat", "hin",
    "ind", "ita", "jpn", "kor", "mar", "pes", "por", "rus", "slk", "spa", "tur",
    "vie", "yor", "zho",
]

# Candidate UDHR file keys per MIST language, best guess first. Where the test
# set can decide it, it does (and overrides this order); where it cannot, the
# first candidate that exists is used and the choice is recorded as unproven.
# The rationale for each multi-candidate entry:
#   deu  1996 is the current orthography (1901 is pre-reform).
#   hat  "kreyol" is the standard register; "popular" is a simplified rewrite.
#   pes  pes_1 and pes_2 are two independent Persian translations.
#   por  European vs Brazilian Portuguese.
#   vie  Latin (quoc ngu) vs the Han-script rendering.
#   zho  Simplified vs Traditional.
#   ckb  see the module docstring: the file labelled ckb is Kurmanji-Latin.
#        kmr holds the same text. There is no Sorani UDHR in this corpus, so
#        whichever is chosen, it is Kurmanji -- which is what the test uses too.
CANDIDATES = {
    "deu": ["deu_1996", "deu_1901"],
    "hat": ["hat_kreyol", "hat_popular"],
    "pes": ["pes_1", "pes_2"],
    "por": ["por_PT", "por_BR"],
    "vie": ["vie", "vie_han"],
    "zho": ["cmn_hans", "cmn_hant"],
    "ckb": ["ckb", "kmr"],
}

_WS_RE = re.compile(r"\s+")
# Soft hyphen, zero-width space, zero-width non-joiner: invisible, and they
# differ between the organizers' rendering and this corpus for no meaningful
# reason. Stripped only for *comparison*, never from the text that ships.
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍﻿"), None)


def norm(s: str) -> str:
    """Comparison form: NFC, invisibles dropped, whitespace collapsed."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s).translate(_INVISIBLE)).strip()


def clean(s: str) -> str:
    """Shipping form: NFC + whitespace collapsed. Keeps every visible character.

    Whitespace collapse is not cosmetic here -- the organizers' prompts contain
    zero real newline characters, so a paragraph carrying one through would be a
    train/test format mismatch. Same rule as
    data/tydi/build_qa_context_train.py's clean_text().
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", s)).strip()


def parse_units(path: Path) -> tuple[dict, dict[str, str]]:
    """-> (file attributes, {unit_id: text}) in document order."""
    root = ET.parse(path).getroot()
    attrs = {
        "key": root.get("key"),
        "iso639_3": root.get("iso639-3"),
        "iso15924": root.get("iso15924"),
        "name": root.get("n"),
        "dir": root.get("dir"),
    }
    units: dict[str, str] = {}

    preamble = root.find(NS + "preamble")
    if preamble is not None:
        for i, p in enumerate(preamble.iter(NS + "para"), start=1):
            units[f"P.{i}"] = clean("".join(p.itertext()))

    for art in root.findall(NS + "article"):
        number = art.get("number")
        if number is None:
            continue
        # Descendants, not children: Arts. 26/27 wrap paragraphs in
        # <orderedlist><listitem>. See the module docstring.
        for i, p in enumerate(art.iter(NS + "para"), start=1):
            units[f"{number}.{i}"] = clean("".join(p.itertext()))

    return attrs, {k: v for k, v in units.items() if v}


def unit_sort_key(unit_id: str) -> tuple[int, int]:
    art, _, para = unit_id.partition(".")
    return (0 if art == "P" else int(art), int(para))


def script_histogram(text: str) -> collections.Counter:
    """Which Unicode scripts the text is written in, by character count."""
    c: collections.Counter = collections.Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        c[name.split()[0]] += 1
    return c


def dominant_script(text: str) -> str | None:
    h = script_histogram(text)
    return h.most_common(1)[0][0] if h else None


def load_test_passages(tests_path: Path, items_path: Path):
    """-> ({(item_id, context_lang): passage}, {item_id: english_passage})"""
    rows = {}
    with tests_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    items = json.loads(items_path.read_text(encoding="utf-8"))
    passages, english = {}, {}
    for it in items:
        english[it["item_id"]] = it["english_passage"]
        for rid in it["row_ids"]:
            lang = rid.split("_")[-1]
            key = (it["item_id"], lang)
            if key not in passages:
                passages[key] = rows[rid]["prompt"].split(SEP)[1]
    return passages, english


def best_span(target: str, units: dict[str, str], max_span: int = 8):
    """Best contiguous run of units matching `target`. -> (ratio, [unit_ids])."""
    ids = sorted(units, key=unit_sort_key)
    t = norm(target)
    best = (0.0, [])
    for i in range(len(ids)):
        acc = []
        for j in range(i, min(i + max_span, len(ids))):
            acc.append(ids[j])
            text = norm(" ".join(units[k] for k in acc))
            # Length prefilter: a span half again longer or shorter than the
            # target cannot win, and SequenceMatcher is the expensive part.
            if len(text) > 2 * len(t):
                break
            if 2 * len(text) < len(t):
                continue
            r = SequenceMatcher(None, t, text).ratio()
            if r > best[0]:
                best = (r, list(acc))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml-dir", type=Path, default=SCRIPT_DIR / "data/udhr_xml")
    ap.add_argument("--tests", type=Path, default=REPO_ROOT / "test/tests.jsonl")
    ap.add_argument("--silver-items", type=Path,
                    default=REPO_ROOT / "silver/silver_items.json")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data")
    ap.add_argument("--report", action="store_true",
                    help="Also print cross-language paragraph-count divergence.")
    args = ap.parse_args()

    available = {p.stem[len("udhr_"):]: p for p in sorted(args.xml_dir.glob("udhr_*.xml"))}
    if not available:
        raise SystemExit(f"error: no UDHR files in {args.xml_dir} -- run fetch_udhr.py first")
    print(f"corpus: {len(available)} language files")

    parsed = {}
    for key, path in available.items():
        try:
            parsed[key] = parse_units(path)
        except ET.ParseError as e:
            print(f"  warning: {key} failed to parse ({e}); skipped")

    test_passages, english_passages = load_test_passages(args.tests, args.silver_items)

    # ---- which test items are UDHR, and which units they use -----------------
    eng_units = parsed["eng"][1]
    alignment = {}
    for item_id, passage in english_passages.items():
        ratio, units = best_span(passage, eng_units)
        if ratio >= 0.80:
            alignment[item_id] = {"units": units, "ratio": round(ratio, 4)}
    used_units = sorted({u for a in alignment.values() for u in a["units"]}, key=unit_sort_key)
    used_articles = sorted({u.split(".")[0] for u in used_units},
                           key=lambda a: (0 if a == "P" else int(a)))
    print(f"\ntest items that are UDHR passages: {len(alignment)}/{len(english_passages)}")
    groups: dict[tuple, list] = {}
    for item_id, a in alignment.items():
        groups.setdefault(tuple(a["units"]), []).append(item_id)
    for units, ids in sorted(groups.items(), key=lambda kv: unit_sort_key(kv[0][0])):
        mean = sum(alignment[i]["ratio"] for i in ids) / len(ids)
        print(f"  {'+'.join(units):24s} {len(ids):2d} items  mean ratio {mean:.4f}")
    print(f"  -> units used by the test set: {len(used_units)}")
    print(f"  -> articles touched: {', '.join(used_articles)}")

    # ---- decide the file for each MIST language ------------------------------
    # What script does the test set actually use for each context language?
    test_script = {}
    for (item_id, lang), passage in test_passages.items():
        test_script.setdefault(lang, collections.Counter()).update(script_histogram(passage))
    test_script = {k: v.most_common(1)[0][0] for k, v in test_script.items() if v}

    mapping = {}
    print(f"\n{'lang':5s} {'file':12s} {'decided by':16s} {'exact':>9s}  script")
    print("-" * 72)
    for lang in MIST_CONTEXT_LANGS:
        cands = [c for c in CANDIDATES.get(lang, [lang]) if c in parsed]
        if not cands:
            cands = [k for k, (a, _) in parsed.items() if a["iso639_3"] == lang]
        if not cands:
            raise SystemExit(f"error: no UDHR file for MIST language {lang!r}")

        # Test passages in this context language that we know are UDHR.
        probes = [(iid, alignment[iid]["units"]) for iid in alignment
                  if (iid, lang) in test_passages]

        scored = []
        for c in cands:
            units = parsed[c][1]
            n_exact = 0
            for iid, uids in probes:
                want = norm(test_passages[(iid, lang)])
                got = norm(" ".join(units.get(u, "") for u in uids))
                n_exact += (got == want and got != "")
            scored.append((n_exact, c))
        scored.sort(key=lambda t: (-t[0], cands.index(t[1])))
        n_exact, chosen = scored[0]

        units = parsed[chosen][1]
        corpus_script = dominant_script(" ".join(units.values()))
        want_script = test_script.get(lang)
        script_ok = (want_script is None or corpus_script == want_script)

        if probes and n_exact == len(probes):
            decided = "test exact-match"
        elif probes and n_exact:
            decided = "test partial"
        elif probes:
            decided = "SCRIPT ONLY*"
        else:
            decided = "script + default"

        mapping[lang] = {
            "udhr_key": chosen,
            "udhr_name": parsed[chosen][0]["name"],
            "iso15924": parsed[chosen][0]["iso15924"],
            "candidates": cands,
            "n_test_probes": len(probes),
            "n_test_exact": n_exact,
            "decided_by": decided,
            "corpus_script": corpus_script,
            "test_script": want_script,
            "script_matches_test": script_ok,
            "n_units": len(units),
        }
        flag = "" if script_ok else "   <-- SCRIPT MISMATCH"
        probe_s = f"{n_exact}/{len(probes)}" if probes else "-"
        print(f"{lang:5s} {chosen:12s} {decided:16s} {probe_s:>9s}  "
              f"{corpus_script or '?'} vs test {want_script or '?'}{flag}")

    n_proven = sum(1 for m in mapping.values() if m["decided_by"] == "test exact-match")
    n_bad_script = sum(1 for m in mapping.values() if not m["script_matches_test"])
    print(f"\n{n_proven}/{len(mapping)} languages decided by exact match against real test "
          f"passages; the rest by script + default")
    if n_bad_script:
        print(f"WARNING: {n_bad_script} languages' script disagrees with the test set")

    # ---- write ---------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    units_out = {lang: parsed[m["udhr_key"]][1] for lang, m in mapping.items()}
    (args.out_dir / "udhr_units.json").write_text(
        json.dumps(units_out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "udhr_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "udhr_alignment.json").write_text(
        json.dumps({
            "items": alignment,
            "used_units": used_units,
            "used_articles": used_articles,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out_dir / 'udhr_units.json'} "
          f"({len(units_out)} languages)")
    print(f"wrote {args.out_dir / 'udhr_mapping.json'}")
    print(f"wrote {args.out_dir / 'udhr_alignment.json'}")

    # ---- optional structural report -----------------------------------------
    if args.report:
        print("\nparagraph-count divergence from English (article: eng -> others)")
        eng_counts = collections.Counter(u.split(".")[0] for u in eng_units)
        for art in sorted(eng_counts, key=lambda a: (0 if a == "P" else int(a))):
            diverging = []
            for lang, m in mapping.items():
                c = collections.Counter(u.split(".")[0] for u in units_out[lang])
                if c[art] != eng_counts[art]:
                    diverging.append(f"{lang}:{c[art]}")
            if diverging:
                print(f"  art {art:>2s} (eng {eng_counts[art]}): {' '.join(diverging)}")

    print("\nNext: python3 data/udhr/build_passages.py")


if __name__ == "__main__":
    main()
