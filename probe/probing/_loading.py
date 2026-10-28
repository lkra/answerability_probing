"""Load Qwen3 / Qwen3.5 checkpoints (causal or multimodal). load_backbone() returns the model
without its LM head, for activation extraction; load_generator() with it, for decoding.
"""

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def text_config(config):
    """The text stack's config, whether or not the checkpoint is multimodal.

    Multimodal configs nest the decoder under `text_config`, so reading
    `num_hidden_layers` off the top-level config returns the *vision* tower's
    depth -- or raises. Everything downstream (layer sweeps, `--save-layers`
    bounds) must key off this, not off the outer config.
    """
    return getattr(config, "text_config", config)


def load_tokenizer(name, padding_side):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _dtype_for(device, dtype):
    # startswith, not ==: pick_device returns an indexed device ("cuda:0"), and an
    # equality check here silently loaded the model in float32 and OOMed a 24 GB card.
    if not device.startswith("cuda") and dtype == "bfloat16":
        return torch.float32          # bf16 matmul on CPU is slow and patchily supported
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def load_backbone(name, device, dtype="bfloat16"):
    """Hidden states only, no LM head -> (model, n_text_layers + 1, hidden_size). On a
    *ForConditionalGeneration checkpoint AutoModel returns the multimodal base; with no image tokens
    output_hidden_states is the text stack, which verify_hidden_states asserts.
    """
    model = AutoModel.from_pretrained(
        name, dtype=_dtype_for(device, dtype), trust_remote_code=True,
    ).to(device)
    model.eval()
    tc = text_config(model.config)
    return model, tc.num_hidden_layers + 1, tc.hidden_size     # +1: embedding output is layer 0


def load_generator(name, device, dtype="bfloat16"):
    """With the LM head, for `.generate()`.

    Tries causal-LM first so nothing changes for Qwen3-4B and other text-only
    checkpoints, then the image-text-to-text head that Qwen3.5 needs. The generation
    call itself is identical either way: a text-only batch carries no image tokens,
    so the vision tower is never entered.
    """
    from transformers import AutoModelForCausalLM
    kw = dict(dtype=_dtype_for(device, dtype), trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, device_map=device, **kw)
    except (ValueError, KeyError, OSError) as e:
        try:
            from transformers import AutoModelForImageTextToText as Fallback
        except ImportError:
            raise e
        arch = getattr(AutoConfig.from_pretrained(name, trust_remote_code=True),
                       "architectures", None)
        print(f"AutoModelForCausalLM refused {name} (architectures={arch}); "
              f"loading as image-text-to-text instead")
        model = Fallback.from_pretrained(name, device_map=device, **kw)
    model.eval()
    return model


def verify_hidden_states(model, tokenizer, device, expect_layers, expect_hidden):
    """Assert the hidden states really are the text stack, before a multi-hour run.

    Cheap insurance against the failure this switch invites: a cache with the right
    shape built from the wrong tower, discovered only when every probe scores chance.
    """
    ids = tokenizer("hello world", return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    if len(hs) != expect_layers:
        raise SystemExit(
            f"expected {expect_layers} hidden states (text layers + embedding), got {len(hs)}. "
            f"The model is probably returning a different stack than the text decoder.")
    if hs[0].shape[-1] != expect_hidden:
        raise SystemExit(
            f"hidden size is {hs[0].shape[-1]}, expected {expect_hidden} -- wrong stack.")
    return True
