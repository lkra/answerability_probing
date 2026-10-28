#!/usr/bin/env python3
r"""Apply a probe fitted on the train cache(s), unchanged, to a cross-lingual cache. Reports AUROC per
(passage, question) language pair, marginals for question languages seen vs unseen in training, the
arb->pes pair separately, and two lexical baselines: overlap on the presented pair and overlap on the
source-language question. --mono-from-diagonal takes the monolingual reference from the cross-lingual
cache itself (BELEBELE); --layer-profile scores every saved layer.

    python3 probe/probing/eval_crosslingual.py --layer 19 --train-acts <train> [<udhr>] \
        --mono-acts <validation> --xl-acts <translation> --out <json>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_activations import pick_device
from train_probes import (accuracy, auroc, fit_logreg, layer_column, load_cache,
                          overlap_feature, saved_layers)

# The pair data/translation/README.md documents as systematically broken: 59.5% of Arabic "who is X?"
# questions come back as Persian "Who am I? X." statements. Kept in the cache and reported as its
# own line rather than silently dropped -- a probe that reads the question should visibly suffer
# here, which makes it a natural control rather than only a data defect.
BROKEN_PAIR = ("arb_Arab", "pes_Arab")


def load_passages(data_dir, split="validation"):
    """tydi_id -> (question, passage). Needed for both overlap baselines."""
    out = {}
    with open(Path(data_dir) / f"{split}.jsonl") as f:
        for line in f:
            o = json.loads(line)
            out[o["id"]] = (o["question"], o["context"])
    return out


def belebele_texts(path):
    """(id -> passage) and ((item_id, lang) -> question), for the two overlap baselines.

    Belebele is professionally translated and parallel across languages, so the *same* question
    exists in the passage's language. That gives a stronger `overlap_source` baseline than the
    translation set can: not "what an MT system would produce" but the actual human translation.
    """
    passage_of, question_of = {}, {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            parts = o["input"].split("\n")
            if len(parts) != 6:
                continue
            passage_of[o["id"]] = parts[0]
            question_of[(o["item_id"], o["question_lang"])] = parts[1]
    return passage_of, question_of


def pair_complete_ids(data_dir, split="validation"):
    """TyDi ids belonging to a question whose answerable *and* unanswerable row are both clean.

    TyDi pairs each question with two passages, one that answers it and one that does not, so the
    build is balanced by construction. Contamination filtering breaks that, because the organizer
    sample overlaps almost only the answerable half. Keeping whole pairs restores exact balance at
    the cost of dropping questions whose partner row was contaminated.
    """
    groups = defaultdict(list)
    with open(Path(data_dir) / f"{split}.jsonl") as f:
        for line in f:
            o = json.loads(line)
            groups[(o["lang_code"], o["question"])].append(o)
    keep = set()
    for rows in groups.values():
        clean = [o for o in rows if not o["in_organizer_sample"]]
        if len(clean) == 2 and {o["answerable"] for o in clean} == {0, 1}:
            keep.update(o["id"] for o in clean)
    return keep


def cell(probs, y):
    """Scores for one slice, always carrying the majority rate the accuracy has to beat."""
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return None
    frac_pos = float(np.mean(y))
    return {"n": int(len(y)), "acc": accuracy(probs, y), "auroc": auroc(probs, y),
            "majority": max(frac_pos, 1 - frac_pos), "frac_answerable": frac_pos}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-acts", type=Path, default=REPO_ROOT / "probe" / "caches" / "train")
    p.add_argument("--mono-acts", type=Path, default=REPO_ROOT / "probe" / "caches" / "validation")
    p.add_argument("--xl-acts", type=Path, default=REPO_ROOT / "probe" / "caches" / "translation")
    p.add_argument("--data-dir", type=Path,
                   default=REPO_ROOT / "data" / "built" / "tydiqa_answerability")
    p.add_argument("--layer", type=int, default=None, help="default: best_layer from --lolo")
    p.add_argument("--lolo", type=Path,
                   default=REPO_ROOT / "probe" / "runs" / "qwen35-4b" / "bare" / "lolo_train.json",
                   help="train_probes.py output; only used to read best_layer when --layer is "
                        "omitted. The Qwen3-4B lolo_train.json this defaulted to was removed "
                        "from the tree; it is still in git history.")
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--keep-contaminated", action="store_true")
    p.add_argument("--pair-complete", action="store_true",
                   help="keep only questions whose answerable and unanswerable row both survived "
                        "contamination filtering, giving an exactly balanced set. Costs ben_Beng "
                        "entirely and all but 6 kor_Hang rows.")
    p.add_argument("--mono-from-diagonal", action="store_true",
                   help="take the monolingual reference from same-language rows inside --xl-acts "
                        "rather than a separate cache. Required for Belebele, which has no TyDi "
                        "counterpart but does contain all 26 same-language pairs.")
    p.add_argument("--belebele", type=Path,
                   default=REPO_ROOT / "data" / "built" /
                    "belebele_answerability_delete_strict.jsonl",
                   help="source file for the overlap baselines when --xl-acts is a Belebele cache")
    p.add_argument("--layer-profile", action="store_true",
                   help="also score every layer the cross-lingual cache stores")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "probe" / "runs" / "qwen35-4b" / "bare" / "crosslingual.json")
    args = p.parse_args()

    args.device = pick_device(args.device)
    layer = args.layer
    if layer is None:
        if not args.lolo.exists():
            sys.exit(f"--layer not given and {args.lolo} does not exist to take best_layer from")
        layer = json.loads(args.lolo.read_text())["best_layer"]

    cfg_tr, meta_tr, acts_tr, idx_tr = load_cache(args.train_acts, args.keep_contaminated)
    cfg_xl, meta_xl, acts_xl, idx_xl = load_cache(args.xl_acts, args.keep_contaminated)
    if args.mono_from_diagonal:
        # Split the one cache in two: same-language rows are the reference, the rest is the test.
        diag = [i for i, m in enumerate(meta_xl) if m["passage_lang"] == m["question_lang"]]
        offd = [i for i, m in enumerate(meta_xl) if m["passage_lang"] != m["question_lang"]]
        if not diag:
            sys.exit("--mono-from-diagonal but the cross-lingual cache has no same-language rows")
        cfg_mo = cfg_xl
        meta_mo, idx_mo = [meta_xl[i] for i in diag], idx_xl[diag]
        meta_xl, idx_xl = [meta_xl[i] for i in offd], idx_xl[offd]
        print(f"reference from the diagonal: {len(meta_mo)} same-language rows, "
              f"{len(meta_xl)} cross-lingual rows")
    else:
        cfg_mo, meta_mo, acts_mo, idx_mo = load_cache(args.mono_acts, args.keep_contaminated)

    if args.pair_complete and cfg_xl.get("source") == "belebele":
        print("--pair-complete ignored: Belebele is balanced by construction and has no "
              "organizer-sample overlap")
        args.pair_complete = False
    if args.pair_complete:
        keep_ids = pair_complete_ids(args.data_dir)
        def restrict(meta, idx, key):
            m = [(i, r) for i, r in enumerate(meta) if r[key] in keep_ids]
            return [r for _, r in m], idx[[i for i, _ in m]]
        n0_mo, n0_xl = len(meta_mo), len(meta_xl)
        meta_mo, idx_mo = restrict(meta_mo, idx_mo, "id")
        meta_xl, idx_xl = restrict(meta_xl, idx_xl, "tydi_id")
        print(f"--pair-complete: monolingual {n0_mo} -> {len(meta_mo)}, "
              f"cross-lingual {n0_xl} -> {len(meta_xl)} rows (exactly balanced)")

    for name, c in (("mono", cfg_mo), ("cross-lingual", cfg_xl)):
        if (c["model"], c["pooling"]) != (cfg_tr["model"], cfg_tr["pooling"]):
            sys.exit(f"{name} cache was built with {c['model']}/{c['pooling']}, the training cache "
                     f"with {cfg_tr['model']}/{cfg_tr['pooling']} -- they cannot be compared.")

    y_tr = np.array([m["answerable"] for m in meta_tr])
    y_mo = np.array([m["answerable"] for m in meta_mo])
    y_xl = np.array([m["answerable"] for m in meta_xl])
    p_lang = np.array([m["passage_lang"] for m in meta_xl])
    q_lang = np.array([m["question_lang"] for m in meta_xl])
    mo_lang = np.array([m.get("passage_lang", m["lang_code"]) for m in meta_mo])

    train_langs = sorted({m["lang_code"] for m in meta_tr})
    p_langs = sorted(set(p_lang.tolist()))
    q_langs = sorted(set(q_lang.tolist()))
    seen_q = [l for l in q_langs if l in train_langs]
    unseen_q = [l for l in q_langs if l not in train_langs]

    print(f"probe: {cfg_tr['model']} layer {layer}, trained on {len(y_tr)} monolingual rows "
          f"({len(train_langs)} languages)")
    print(f"monolingual validation: {len(y_mo)} rows")
    print(f"cross-lingual:          {len(y_xl)} rows, {len(p_langs)} passage x {len(q_langs)} "
          f"question languages")
    print(f"  question languages the probe trained on: {', '.join(seen_q)}")
    print(f"  question languages never seen:           {', '.join(unseen_q)}\n")

    acts_mo_arr = acts_xl if args.mono_from_diagonal else acts_mo

    def fit_at(lay):
        Xtr = np.asarray(acts_tr[idx_tr, layer_column(cfg_tr, lay), :])
        Xmo = np.asarray(acts_mo_arr[idx_mo, layer_column(cfg_mo, lay), :])
        Xxl = np.asarray(acts_xl[idx_xl, layer_column(cfg_xl, lay), :])
        # One probe, fit once, applied to both evaluation sets unchanged.
        both = fit_logreg(Xtr, y_tr, np.concatenate([Xmo, Xxl]), args.device, l2=args.l2)
        return both[:len(y_mo)], both[len(y_mo):]

    pr_mo, pr_xl = fit_at(layer)

    # prompt_style is recorded so a renderer can refuse to splice a bare-prompt cross-lingual
    # result into a qa-context report. Model alone is not enough to tell them apart: both
    # variants of this rerun are the same model and differ only in the prompt.
    results = {"config": {"model": cfg_tr["model"], "pooling": cfg_tr["pooling"], "layer": layer,
                          "prompt_style": cfg_tr.get("prompt_style"),
                          "max_len": cfg_tr.get("max_len"),
                          "l2": args.l2, "train_rows": int(len(y_tr)),
                          "train_languages": train_langs,
                          "keep_contaminated": bool(args.keep_contaminated),
                          "pair_complete": bool(args.pair_complete)},
               "passage_languages": p_langs, "question_languages": q_langs,
               "seen_question_languages": seen_q, "unseen_question_languages": unseen_q}

    # ---- headline ------------------------------------------------------------------------
    results["monolingual"] = cell(pr_mo, y_mo)
    results["crosslingual"] = cell(pr_xl, y_xl)
    print("=== headline ===")
    print(f"{'monolingual validation':28s} n={results['monolingual']['n']:6d} "
          f"acc={results['monolingual']['acc']:.3f} AUROC={results['monolingual']['auroc']:.3f}")
    print(f"{'cross-lingual (all pairs)':28s} n={results['crosslingual']['n']:6d} "
          f"acc={results['crosslingual']['acc']:.3f} AUROC={results['crosslingual']['auroc']:.3f}")

    # ---- sensitivity: passages that also occur in the probe's training split --------------
    # The question differs even when the passage recurs, and the label depends on the
    # question-passage relation rather than the passage alone, so this should not leak. Reported
    # anyway, because "4% of your evaluation passages were in training" is the first thing a
    # reviewer will ask and it is cheaper to answer than to argue.
    shared_xl = np.array([bool(m.get("passage_in_other_split")) for m in meta_xl])
    shared_mo = np.array([bool(m.get("passage_in_other_split")) for m in meta_mo])
    if shared_xl.any() or shared_mo.any():
        results["excluding_shared_passages"] = {
            "monolingual": cell(pr_mo[~shared_mo], y_mo[~shared_mo]),
            "crosslingual": cell(pr_xl[~shared_xl], y_xl[~shared_xl]),
            "n_dropped_monolingual": int(shared_mo.sum()),
            "n_dropped_crosslingual": int(shared_xl.sum()),
        }
        e = results["excluding_shared_passages"]
        print(f"\n=== sensitivity: dropping passages that also appear in the train split ===")
        print(f"  dropped {e['n_dropped_crosslingual']} cross-lingual "
              f"({e['n_dropped_crosslingual'] / len(y_xl):.1%}) and "
              f"{e['n_dropped_monolingual']} monolingual rows")
        if e["monolingual"]:
            print(f"  monolingual  {results['monolingual']['auroc']:.3f} -> "
                  f"{e['monolingual']['auroc']:.3f}")
        if e["crosslingual"]:
            print(f"  crosslingual {results['crosslingual']['auroc']:.3f} -> "
                  f"{e['crosslingual']['auroc']:.3f}")

    # ---- baselines -----------------------------------------------------------------------
    ov_tr, ov_src = [], []
    if cfg_xl.get("source") == "belebele":
        passage_of, question_of = belebele_texts(args.belebele)
        for m in meta_xl:
            passage = passage_of[m["id"]]
            ov_tr.append(overlap_feature(m["question"], passage, m["question_lang"]))
            # the same question in the passage's language -- a human translation, not an MT one
            same = question_of.get((m["item_id"], m["passage_lang"]))
            ov_src.append(overlap_feature(same, passage, m["passage_lang"]) if same else 0.0)
        src_label = "overlap, question in the passage's language (human translation)"
    else:
        text = load_passages(args.data_dir)
        for m in meta_xl:
            _, passage = text[m["tydi_id"]]
            ov_tr.append(overlap_feature(m["question"], passage, m["question_lang"]))
            ov_src.append(overlap_feature(m["src_question"], passage, m["passage_lang"]))
        src_label = "overlap, original (untranslated) question"
    ov_tr, ov_src = np.array(ov_tr), np.array(ov_src)
    results["overlap_source_label"] = src_label

    # The reference rows need the same feature, so the same-language and cross-lingual slices can
    # be compared against the *same* baseline. With --mono-from-diagonal they live in meta_mo.
    ov_src_mo = None
    if cfg_xl.get("source") == "belebele":
        ov_src_mo = np.array([
            overlap_feature(question_of.get((m["item_id"], m["passage_lang"]), ""),
                            passage_of[m["id"]], m["passage_lang"]) or 0.0
            for m in meta_mo])

    # Fit each one-feature baseline on the monolingual training rows it corresponds to would need
    # the train passages; instead fit on half the cross-lingual rows and score the other half, which
    # is generous to the baseline (it is tuned on the same distribution it is tested on).
    rng = np.random.default_rng(0)
    half = rng.permutation(len(y_xl))
    a, b = half[:len(half) // 2], half[len(half) // 2:]
    results["baselines"] = {}
    for name, feat in (("overlap_translated", ov_tr), ("overlap_source", ov_src)):
        pr = np.empty(len(y_xl))
        pr[b] = fit_logreg(feat[a, None], y_xl[a], feat[b, None], args.device, l2=args.l2)
        pr[a] = fit_logreg(feat[b, None], y_xl[b], feat[a, None], args.device, l2=args.l2)
        results["baselines"][name] = cell(pr, y_xl)
        results["baselines"][name]["_probs"] = None
        print(f"{name:28s} n={len(y_xl):6d} acc={results['baselines'][name]['acc']:.3f} "
              f"AUROC={results['baselines'][name]['auroc']:.3f}")
        if name == "overlap_translated":
            print("   ^ the cue actually present cross-lingually; near chance is the expected result")
        else:
            print("   ^ what a lexical system with a perfect translator could do -- the bar to beat")
    for v in results["baselines"].values():
        v.pop("_probs", None)

    # ---- probe vs the lexical bar, per passage language ----------------------------------
    # The pooled comparison can hide that a probe beats the bar in some languages and loses in
    # others. On Belebele's delete-negatives the bar is unusually strong -- the sentence removed to
    # make a row unanswerable is exactly the one carrying the question's content words, so its
    # absence is directly detectable lexically -- and the probe does not necessarily clear it.
    pr_src = np.empty(len(y_xl))
    pr_src[b] = fit_logreg(ov_src[a, None], y_xl[a], ov_src[b, None], args.device, l2=args.l2)
    pr_src[a] = fit_logreg(ov_src[b, None], y_xl[b], ov_src[a, None], args.device, l2=args.l2)
    xl_only = p_lang != q_lang
    vs_bar = {}
    print(f"\n=== probe vs the lexical bar ({src_label}), cross-lingual rows only ===")
    print(f"{'passage':12s} {'probe':>7s} {'bar':>8s} {'margin':>8s}")
    for l in p_langs:
        sel = xl_only & (p_lang == l)
        cp, cb = cell(pr_xl[sel], y_xl[sel]), cell(pr_src[sel], y_xl[sel])
        if cp and cb:
            vs_bar[l] = {"probe_auroc": cp["auroc"], "bar_auroc": cb["auroc"],
                         "margin": cp["auroc"] - cb["auroc"], "n": cp["n"]}
            print(f"{l:12s} {cp['auroc']:7.3f} {cb['auroc']:8.3f} "
                  f"{cp['auroc'] - cb['auroc']:+8.3f}")
    if vs_bar:
        results["probe_vs_bar_by_passage_language"] = vs_bar
        n_beat = sum(1 for v in vs_bar.values() if v["margin"] > 0)
        results["n_languages_beating_bar"] = n_beat
        print(f"  probe beats the bar in {n_beat}/{len(vs_bar)} passage languages")

    # Splitting the same comparison by same- vs different-language separates two costs that are
    # otherwise confounded: how much the probe loses moving to this data at all (domain), and how
    # much it loses from the languages differing (cross-linguality). Only the second is what this
    # experiment is about, and a probe can clear the bar monolingually while failing cross-lingually.
    slices = []
    if (p_lang == q_lang).any():
        slices = [("same language", pr_xl, pr_src, y_xl, p_lang == q_lang)]
    elif ov_src_mo is not None:
        # --mono-from-diagonal moved the same-language rows into the reference set; fit the bar
        # there on its own rows so the two slices are scored the same way.
        pm = np.empty(len(y_mo))
        h = np.random.default_rng(0).permutation(len(y_mo))
        ha, hb = h[:len(h) // 2], h[len(h) // 2:]
        pm[hb] = fit_logreg(ov_src_mo[ha, None], y_mo[ha], ov_src_mo[hb, None], args.device, l2=args.l2)
        pm[ha] = fit_logreg(ov_src_mo[hb, None], y_mo[hb], ov_src_mo[ha, None], args.device, l2=args.l2)
        slices = [("same language", pr_mo, pm, y_mo, np.ones(len(y_mo), bool))]
    if slices:
        slices.append(("cross-lingual", pr_xl, pr_src, y_xl,
                       (p_lang != q_lang) if (p_lang == q_lang).any()
                       else np.ones(len(y_xl), bool)))
        split = {}
        for label, pp, pb, yy, sel in slices:
            cp, cb = cell(pp[sel], yy[sel]), cell(pb[sel], yy[sel])
            if cp and cb:
                split[label] = {"probe_auroc": cp["auroc"], "bar_auroc": cb["auroc"],
                                "margin": cp["auroc"] - cb["auroc"], "n": cp["n"]}
        if len(split) == 2:
            results["probe_vs_bar_by_lingual_split"] = split
            print(f"\n  {'slice':16s} {'probe':>7s} {'bar':>8s} {'margin':>8s}")
            for k, v in split.items():
                print(f"  {k:16s} {v['probe_auroc']:7.3f} {v['bar_auroc']:8.3f} "
                      f"{v['margin']:+8.3f}")
            pen = split["same language"]["probe_auroc"] - split["cross-lingual"]["probe_auroc"]
            results["crosslingual_penalty"] = pen
            print(f"  cross-lingual penalty: -{pen:.3f}")

    # ---- passage x question matrix -------------------------------------------------------
    matrix = defaultdict(dict)
    for pl in p_langs:
        for ql in q_langs:
            sel = (p_lang == pl) & (q_lang == ql)
            c = cell(pr_xl[sel], y_xl[sel])
            if c:
                matrix[pl][ql] = c
    results["matrix"] = {k: v for k, v in matrix.items()}

    # monolingual reference per passage language, from the same validation items
    mono_by_lang = {}
    for lg in p_langs:
        sel = mo_lang == lg
        c = cell(pr_mo[sel], y_mo[sel])
        if c:
            mono_by_lang[lg] = c
    results["monolingual_by_language"] = mono_by_lang

    print(f"\n=== AUROC by passage language (rows) x question language (columns), layer {layer} ===")
    hdr = f"{'passage':10s} {'mono':>6s} |" + "".join(f"{q[:3]:>6s}" for q in q_langs) + f" {'mean':>7s}"
    print(hdr); print("-" * len(hdr))
    for pl in p_langs:
        vals = [matrix[pl][q]["auroc"] for q in q_langs if q in matrix[pl]]
        cells = "".join(f"{matrix[pl][q]['auroc']:6.3f}" if q in matrix[pl] else "     -"
                        for q in q_langs)
        mono = f"{mono_by_lang[pl]['auroc']:6.3f}" if pl in mono_by_lang else "     -"
        rowmean = f"{np.mean(vals):7.3f}" if vals else f"{'-':>7s}"
        print(f"{pl:10s} {mono} |{cells} {rowmean}")
    print("-" * len(hdr))
    colmeans = []
    for q in q_langs:
        v = [matrix[pl][q]["auroc"] for pl in p_langs if q in matrix[pl]]
        colmeans.append(float(np.mean(v)) if v else None)
    print(f"{'mean':10s} {np.mean([c['auroc'] for c in mono_by_lang.values()]):6.3f} |"
          + "".join(f"{c:6.3f}" if c is not None else "     -" for c in colmeans))
    results["question_language_means"] = dict(zip(q_langs, colmeans))

    # ---- marginals -----------------------------------------------------------------------
    def marginal(mask_of, keys):
        out = {}
        for k in keys:
            sel = mask_of(k)
            c = cell(pr_xl[sel], y_xl[sel])
            if c:
                out[k] = c
        return out

    results["by_question_language"] = marginal(lambda k: q_lang == k, q_langs)
    results["by_passage_language"] = marginal(lambda k: p_lang == k, p_langs)

    print(f"\n=== by question language (the language the question was asked in) ===")
    print(f"{'question':10s} {'seen?':>6s} {'n':>7s} {'AUROC':>7s} {'acc':>7s}")
    print("-" * 40)
    for ql in sorted(results["by_question_language"],
                     key=lambda k: -results["by_question_language"][k]["auroc"]):
        c = results["by_question_language"][ql]
        print(f"{ql:10s} {'yes' if ql in seen_q else 'no':>6s} {c['n']:7d} "
              f"{c['auroc']:7.3f} {c['acc']:7.3f}")

    for label, group in (("trained on", seen_q), ("never seen", unseen_q)):
        sel = np.isin(q_lang, group)
        c = cell(pr_xl[sel], y_xl[sel])
        results.setdefault("by_seen_group", {})[label] = c
        print(f"  question language {label:11s}: n={c['n']:6d} AUROC={c['auroc']:.3f}")

    # ---- same script vs different script -------------------------------------------------
    # If the signal were lexical, sharing a script (where names, numbers and loanwords survive
    # translation verbatim) should help a lot. If it is semantic, it should barely matter.
    same_script = np.array([m["passage_lang"].split("_")[1] == m["question_lang"].split("_")[1]
                            for m in meta_xl])
    results["by_script"] = {}
    print(f"\n=== same script vs different script ===")
    for label, sel in (("same script", same_script), ("different script", ~same_script)):
        c = cell(pr_xl[sel], y_xl[sel])
        if c:
            results["by_script"][label] = c
            print(f"  {label:18s} n={c['n']:6d} AUROC={c['auroc']:.3f}")
    if len(results["by_script"]) == 2:
        d = (results["by_script"]["different script"]["auroc"]
             - results["by_script"]["same script"]["auroc"])
        print(f"  pooled difference: {d:+.3f}  <- CONFOUNDED, see below")

    # Pooling across passage languages is not a fair comparison: which passage languages have
    # same-script partners at all is not random. Latin-script passages dominate the same-script
    # pool and swh_Latn, the weakest passage language, is one of them -- so the pooled number
    # partly measures passage-language composition rather than script sharing. Compare within
    # each passage language instead, where composition is held fixed.
    within = {}
    for pl in p_langs:
        sc = pl.split("_")[1]
        cells_pl = matrix.get(pl, {})
        same = [c["auroc"] for q, c in cells_pl.items() if q.split("_")[1] == sc]
        diff = [c["auroc"] for q, c in cells_pl.items() if q.split("_")[1] != sc]
        if same and diff:
            within[pl] = {"same_script": float(np.mean(same)),
                          "different_script": float(np.mean(diff)),
                          "delta": float(np.mean(diff) - np.mean(same)),
                          "n_same_script_pairs": len(same)}
    if within:
        results["by_script_within_passage_language"] = within
        md = float(np.mean([v["delta"] for v in within.values()]))
        results["by_script_within_mean_delta"] = md
        print(f"  within passage language ({len(within)} languages have both): {md:+.3f}")
        print("  ^ this is the honest number; the pooled one above is confounded by which "
              "passage languages have same-script partners")

    # ---- the broken pair as a control ----------------------------------------------------
    src, tgt = BROKEN_PAIR
    sel_broken = ((p_lang == src) & (q_lang == tgt)
                  if cfg_xl.get("source") != "belebele" else np.zeros(len(y_xl), bool))
    sel_ctrl = (p_lang == src) & (q_lang != tgt)
    if sel_broken.any():
        results["broken_pair"] = {
            "pair": f"{src}->{tgt}",
            "broken": cell(pr_xl[sel_broken], y_xl[sel_broken]),
            "same_passage_other_targets": cell(pr_xl[sel_ctrl], y_xl[sel_ctrl]),
        }
        bp = results["broken_pair"]
        print(f"\n=== {src} -> {tgt}, the pair data/translation/README.md flags as broken ===")
        print(f"  {src}->{tgt:10s} n={bp['broken']['n']:6d} AUROC={bp['broken']['auroc']:.3f}")
        print(f"  {src}->all others  n={bp['same_passage_other_targets']['n']:6d} "
              f"AUROC={bp['same_passage_other_targets']['auroc']:.3f}")
        print("  same passages, same probe -- the difference is the translation")

    # ---- layer profile -------------------------------------------------------------------
    if args.layer_profile:
        common = [l for l in saved_layers(cfg_xl)
                  if l in saved_layers(cfg_tr) and l in saved_layers(cfg_mo)]
        print(f"\n=== layer profile ({len(common)} layers stored in all three caches) ===")
        print(f"{'layer':>5s} {'mono':>7s} {'cross':>7s} {'gap':>7s}")
        prof = {}
        for lay in common:
            a_mo, a_xl = fit_at(lay)
            m, x = auroc(a_mo, y_mo), auroc(a_xl, y_xl)
            prof[lay] = {"mono_auroc": m, "xl_auroc": x, "gap": x - m}
            print(f"{lay:5d} {m:7.3f} {x:7.3f} {x - m:+7.3f}")
        results["layer_profile"] = prof

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
