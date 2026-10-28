r"""Prompt rendering shared by extraction and generation. bare: "{passage}\n\n{question}" inside the
chat template with no instruction; qa-context: the organisers' four-segment template joined by the
literal two-character sequence backslash-n backslash-n. enable_thinking=False throughout, so the
last-token position is the one the model decodes from.
"""

import json
from pathlib import Path

# The organizer's own rendering: build_tydiqa.py emits `input` == context + "\n\n" + question,
# verified byte-identical to the organizer's format on all 129,392 rows. Deliberately neutral --
# no task instruction and no abstention option, because the probe is meant to read a representation
# the model forms anyway, not one produced by prompting the model to self-report.
BARE_TEMPLATE = "{passage}\n\n{question}"

PROMPT_STYLES = ("bare", "qa-context")


def load_templates(path):
    """-> (templates, sep). `sep` is literal backslash-n backslash-n; see module docstring."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return spec["templates"], spec["sep"]


def render_qa_context(templates, question_lang, passage, question, sep):
    """The organizers' four-segment layout, verbatim.

    `question_lang` is the bare three-letter code ("arb"), not the FLORES code
    ("arb_Arab"), and it keys on the *question* language -- cross-lingually the
    passage is in a different one and the instruction must follow the question.
    """
    t = templates[question_lang]
    return sep.join([t["intro"], passage, t["q_prefix"] + question, t["instruction"]])


def render_content(style, passage, question, templates=None, question_lang=None, sep=None):
    """The user-turn content, before the chat template wraps it."""
    if style == "bare":
        return BARE_TEMPLATE.format(passage=passage, question=question)
    if style == "qa-context":
        if templates is None or question_lang is None or sep is None:
            raise ValueError("qa-context needs templates, question_lang and sep")
        return render_qa_context(templates, question_lang, passage, question, sep)
    raise ValueError(f"unknown prompt style {style!r}, expected one of {PROMPT_STYLES}")


def render_chat(tokenizer, content, enable_thinking=False):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking,
    )


def build_prompt(tokenizer, passage, question, max_len, style="bare", templates=None,
                 question_lang=None, sep=None, enable_thinking=False):
    """Render one row, truncating the passage only (never the question) -> (text, ids, truncated). The
    binary search runs on the fully rendered string, so template and instruction overhead count
    towards max_len; qa-context needs a larger --max-len than bare.
    """
    def render(psg):
        return render_chat(
            tokenizer,
            render_content(style, psg, question, templates, question_lang, sep),
            enable_thinking,
        )

    text = render(passage)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_len:
        return text, ids, False

    lo, hi = 0, len(passage)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n = len(tokenizer(render(passage[:mid]), add_special_tokens=False)["input_ids"])
        if n <= max_len:
            lo = mid
        else:
            hi = mid - 1
    text = render(passage[:lo])
    return text, tokenizer(text, add_special_tokens=False)["input_ids"], True
