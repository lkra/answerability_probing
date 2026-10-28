# Probe Before you Answer!

Code and data for [*Probe Before you Answer! Answerability Probing for Abstention in Cross-Lingual
Context-Based Question Answering*](https://aclanthology.org/2026.wmt-main.129/) (WMT 2026), our
submission to Sub-task 1 of the [WMT26 Multilingual Instruction Shared Task](https://www2.statmt.org/wmt26/pdf/2026.wmt-1.50.pdf).

[Lea Krause](https://lkra.github.io)¹, [Stefan Schouten](https://stefanfs.me)¹,
[Michiel van der Meer](https://liacs.leidenuniv.nl/~meermtvander/)², [Piek Vossen](https://vossen.info)¹  
¹Vrije Universiteit Amsterdam, ²Leiden University

The system answers a question about a passage in two steps. A linear probe on layer 19 of the
frozen `unsloth/Qwen3.5-9B` decides whether the passage answers the question. If it does not, the
system returns the refusal string. Otherwise a LoRA adapter, trained only on answerable questions,
generates the answer. We submitted two adapters with the same probe.

```text
data/      datasets: builders, pinned sources, translation bundle, UDHR, silver labels
adapter/   the two submitted adapters: training and inference
probe/     activation extraction, probes, evaluation, abstention mask, submitted files
```

All commands run from the repository root.

## Data — [`data/README.md`](data/README.md)

```bash
pip install -r requirements-data.txt
python3 data/fetch_sources.py                 # pinned upstream corpora and the test release
curl -L -o answerability_probing_translated.tgz \
  "https://drive.usercontent.google.com/download?id=1N05_jmDFypFwOtSbAxUHvVdV7mVv_BZO&export=download&confirm=t"
tar xzf answerability_probing_translated.tgz -C data   # translation bundle -> data/translated/
data/build_all.sh
```

The translation bundle (44 MB) is also on
[Google Drive](https://drive.google.com/file/d/1N05_jmDFypFwOtSbAxUHvVdV7mVv_BZO/view?usp=sharing).

## Adapter — [`adapter/README.md`](adapter/README.md)

Adapter 1 (`TeamSpinoza-primary`) is trained on 100,466 TyDiQA/cross examples with short,
normalised targets. Adapter 2 (`TeamSpinoza-Variant_V4`) is trained on a 60,000-example mix of
TyDiQA/cross, BELEBELE and UDHR (40/50/10).

```bash
sbatch adapter/train/run_finetune.sh --variant v8 --phase train      # after filling in the launcher's <placeholders>
python3 adapter/infer/generate_test_qa_context.py --checkpoint <dir>
python3 adapter/infer/score_qa_context.py --predictions <file> --exclude-lang ckb
```

## Probe — [`probe/README.md`](probe/README.md)

The submitted probe is trained on TyDiQA and UDHR and uses one threshold for all languages. It
abstains on 1,768 of the 8,640 test questions. The probe results reported in the paper are in
`probe/runs/`.

```bash
bash probe/runs/reproduce.sh
python3 probe/submission/build_submission.py
```

## Environment

The code uses Python 3.10. Install `requirements-data.txt` to build the data and `requirements.txt`
for training and extraction.

## Citation

```bibtex
@inproceedings{krause-etal-2026-probe,
    title = "Probe Before you Answer! Answerability Probing for Abstention in Cross-Lingual Context-Based Question Answering",
    author = "Krause, Lea  and
      Schouten, Stefan  and
      Van Der Meer, Michiel  and
      Vossen, Piek",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Eleventh Conference on Machine Translation",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wmt-main.129/",
    pages = "1933--1948"
}
```

## Licences

Code: Apache-2.0 ([`LICENSE`](LICENSE)).

Data sources: `copenlu/answerable_tydiqa` Apache-2.0 · `facebook/belebele` CC BY-SA 4.0 ·
`FBK-MT/MCIF` CC BY 4.0 (via the organisers' sample) · UDHR text from the UN "UDHR in Unicode"
project · test release CC BY 4.0.
