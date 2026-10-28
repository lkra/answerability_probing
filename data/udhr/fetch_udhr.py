#!/usr/bin/env python3
"""Fetch the "UDHR in Unicode" XML corpus (456 files) from the Internet Archive's copy of the
retired unicode.org URL, verified against the sha256 in data/udhr_source.json.
--verify-against-hf cross-checks the text against guipenedo/udhr. An already-verified corpus is
left alone unless --force.

    python3 data/udhr/fetch_udhr.py
"""

import argparse
import datetime
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# The Internet Archive's copy of https://unicode.org/udhr/assemblies/udhr_xml.zip.
# The `id_` modifier asks the Archive for the original bytes rather than its
# rewritten-for-display version -- without it the zip comes back corrupted.
ARCHIVE_URL = "https://web.archive.org/web/2023id_/https://unicode.org/udhr/assemblies/udhr_xml.zip"
EXPECTED_SHA256 = "18bf578bb837e0c6924ff389e607bfde5acc4d02412d8ecde989602ca8ff0514"
EXPECTED_SIZE = 2237381

# Live fallback, same 456 translations but flattened -- see the module docstring.
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/guipenedo/udhr/resolve/"
    "3b390cf842e9a55afa92440d47c427d71549240e/data/train-00000-of-00001.parquet"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "spinoza-mist/udhr_qa"})
    with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")


def verify_against_hf(xml_dir: Path) -> None:
    """Cross-check a few languages against the live HuggingFace copy.

    Only a sanity check that the archived zip is the corpus it claims to be.
    The HF copy is flattened, so this compares *content* (does every paragraph
    of our parse appear in their blob?), not structure.
    """
    import io
    import re
    import unicodedata
    import xml.etree.ElementTree as ET

    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("error: --verify-against-hf needs pandas")

    ns = "{http://www.unhchr.ch/udhr}"
    print("\ncross-checking against guipenedo/udhr (live HF copy)")
    with urllib.request.urlopen(
        urllib.request.Request(HF_PARQUET_URL, headers={"User-Agent": "spinoza-mist/udhr_qa"}),
        timeout=300,
    ) as r:
        df = pd.read_parquet(io.BytesIO(r.read()))

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()

    by_key = {row["lang_key"]: norm(row["text"]) for _, row in df.iterrows()}
    checked = mismatched = 0
    for path in sorted(xml_dir.glob("udhr_*.xml")):
        key = path.stem[len("udhr_"):]
        blob = by_key.get(key)
        if blob is None:
            continue
        root = ET.parse(path).getroot()
        paras = [norm("".join(p.itertext())) for p in root.iter(ns + "para")]
        missing = [p for p in paras if p and p not in blob]
        checked += 1
        if missing:
            mismatched += 1
            print(f"  {key}: {len(missing)}/{len(paras)} paragraphs not found in HF copy")
    print(f"  {checked - mismatched}/{checked} languages agree with the HF copy")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "data")
    ap.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    ap.add_argument("--verify-against-hf", action="store_true",
                    help="Also cross-check the extracted corpus against the live "
                         "HuggingFace copy of the same data.")
    args = ap.parse_args()

    xml_dir = args.out_dir / "udhr_xml"
    meta_path = args.out_dir / "udhr_source.json"

    if meta_path.exists() and xml_dir.exists() and not args.force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256") == EXPECTED_SHA256:
            n = len(list(xml_dir.glob("udhr_*.xml")))
            print(f"already present: {n} language files, sha256 pinned and matching")
            if args.verify_against_hf:
                verify_against_hf(xml_dir)
            return
        print("recorded sha256 does not match the pin; re-fetching")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / "udhr_xml.zip"
    if not zip_path.exists() or args.force:
        download(ARCHIVE_URL, zip_path)

    size = zip_path.stat().st_size
    digest = sha256_file(zip_path)
    if digest != EXPECTED_SHA256 or size != EXPECTED_SIZE:
        raise SystemExit(
            f"error: archive does not match the pin.\n"
            f"  expected sha256 {EXPECTED_SHA256} size {EXPECTED_SIZE}\n"
            f"  got      sha256 {digest} size {size}\n"
            f"The Internet Archive is serving different bytes than this pipeline was "
            f"built against. Do not proceed silently -- inspect the file first."
        )
    print(f"sha256 {digest}  OK")

    if xml_dir.exists() and args.force:
        shutil.rmtree(xml_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".xml") or n.endswith(".rnc") or n.endswith(".rng")]
        for n in names:
            target = xml_dir / Path(n).name
            with z.open(n) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    lang_files = sorted(xml_dir.glob("udhr_*.xml"))
    print(f"extracted {len(lang_files)} language files to {xml_dir}")

    meta_path.write_text(json.dumps({
        "url": ARCHIVE_URL,
        "upstream_url": "https://unicode.org/udhr/assemblies/udhr_xml.zip",
        "upstream_status": "retired by Unicode in January 2024; all data URLs 404",
        "sha256": digest,
        "size_bytes": size,
        "n_language_files": len(lang_files),
        "fetched": datetime.date.today().isoformat(),
        "copyright": "© The Office of the High Commissioner for Human Rights",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}")

    if args.verify_against_hf:
        verify_against_hf(xml_dir)

    print("\nNext: python3 data/udhr/parse_udhr.py")


if __name__ == "__main__":
    main()
