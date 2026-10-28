#!/usr/bin/env python3
"""Download the upstream corpora at the revisions pinned in sources.json into data/upstream/:
copenlu/answerable_tydiqa (train/validation parquet), pinzhenchen/wmt26-mist-sample, the test
release pinzhenchen/wmt26-mist-test (into data/test/) and facebook/belebele (27 languages). Files
already present are skipped unless --force.

    python3 data/fetch_sources.py
    python3 data/fetch_sources.py --only tydiqa mist-sample
"""

import argparse
import fnmatch
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

SCRIPT_DIR = Path(__file__).resolve().parent

# The 27 Belebele languages the MIST work uses. Belebele ships 122; pulling only
# these keeps the download to ~27 MB instead of ~120 MB.
MIST_BELEBELE_LANGUAGES = [
    "arb_Arab", "ben_Beng", "ces_Latn", "ckb_Arab", "deu_Latn", "eng_Latn",
    "fin_Latn", "fra_Latn", "hat_Latn", "hin_Deva", "ind_Latn", "ita_Latn",
    "jpn_Jpan", "kor_Hang", "mar_Deva", "pes_Arab", "por_Latn", "rus_Cyrl",
    "slk_Latn", "spa_Latn", "swh_Latn", "tel_Telu", "tha_Thai", "tur_Latn",
    "vie_Latn", "yor_Latn", "zho_Hans",
]

# (key, repo id, [(remote glob, local path relative to this directory)])
# The repo id must match a `name` in sources.json, which supplies the revision.
SOURCES = {
    "tydiqa": ("copenlu/answerable_tydiqa", [
        ("data/train-*.parquet", "upstream/answerable_tydiqa/train.parquet"),
        ("data/validation-*.parquet", "upstream/answerable_tydiqa/validation.parquet"),
    ]),
    "mist-sample": ("pinzhenchen/wmt26-mist-sample", [
        ("samples.jsonl", "upstream/wmt26-mist-sample/samples.jsonl"),
        ("README.md", "upstream/wmt26-mist-sample/README.md"),
    ]),
    "mist-test": ("pinzhenchen/wmt26-mist-test", [
        ("tests.jsonl", "test/tests.jsonl"),
    ]),
    "belebele": ("facebook/belebele", [
        (f"data/{lang}.jsonl", f"upstream/belebele/{lang}.jsonl")
        for lang in MIST_BELEBELE_LANGUAGES
    ]),
}


def load_revisions(sources_json: Path) -> dict[str, str]:
    spec = json.loads(sources_json.read_text(encoding="utf-8"))
    return {s["name"]: s["revision"] for s in spec["sources"] if s.get("revision")}


def resolve(remote_glob: str, available: list[str], repo: str) -> str:
    """Upstream shard names are content-hashed, so match by pattern."""
    hits = fnmatch.filter(available, remote_glob)
    if not hits:
        raise SystemExit(f"error: {repo} has no file matching {remote_glob!r}")
    if len(hits) > 1:
        raise SystemExit(f"error: {repo} pattern {remote_glob!r} is ambiguous: {hits}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources-json", type=Path, default=SCRIPT_DIR / "sources.json")
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR,
                    help="Root the local paths are relative to (default: this directory).")
    ap.add_argument("--only", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    ap.add_argument("--force", action="store_true", help="Re-download files already present.")
    args = ap.parse_args()

    revisions = load_revisions(args.sources_json)
    n_fetched = n_skipped = 0

    for key in args.only:
        repo, files = SOURCES[key]
        revision = revisions.get(repo)
        if revision is None:
            raise SystemExit(f"error: {repo} has no pinned revision in {args.sources_json}")
        print(f"\n{key}: {repo} @ {revision[:8]}")

        available = None
        for remote_glob, local_rel in files:
            dest = args.out_dir / local_rel
            if dest.exists() and not args.force:
                n_skipped += 1
                continue
            if available is None:  # one listing per repo, not per file
                available = list_repo_files(repo, repo_type="dataset", revision=revision)
            remote = resolve(remote_glob, available, repo)
            cached = hf_hub_download(repo, remote, repo_type="dataset", revision=revision)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)  # copy, not symlink: the cache may be pruned
            print(f"  {remote}  ->  {dest.relative_to(args.out_dir)}")
            n_fetched += 1

    print(f"\nfetched {n_fetched}, already present {n_skipped}"
          + ("  (use --force to re-download)" if n_skipped else ""))
    print("\nNext:")
    print("  python3 data/tydi/build_tydiqa.py                       # -> data/built/tydiqa_answerability/")
    print("  python3 data/translation/build_translation_jobs.py  # -> data/translated/qa_translation_jobs.jsonl")


if __name__ == "__main__":
    main()
