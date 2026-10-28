"""pick_device(): the CUDA device with the most memory among those whose compute capability the
installed torch supports; CPU otherwise.
"""

import re

import torch

# "sm_70" -> (7, 0); "sm_100" -> (10, 0); "sm_90a" -> (9, 0). The last digit is
# the minor version and everything before it the major, so a plain per-character
# parse breaks on the three-digit arches (sm_100, sm_120) that current wheels ship.
_ARCH_RE = re.compile(r"^sm_(\d+)(\d)[a-z]*$")


def compatible_cuda_devices() -> list[int]:
    """CUDA device indices this torch build has kernels for."""
    if not torch.cuda.is_available():
        return []
    try:
        supported = set()
        for arch in torch.cuda.get_arch_list():
            m = _ARCH_RE.match(arch)
            if m:
                supported.add((int(m.group(1)), int(m.group(2))))
    except Exception:
        supported = set()
    out = []
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        # A device runs if torch ships kernels for its major version at or below
        # its minor -- matching how CUDA selects a compatible cubin.
        if not supported or any(m == cap[0] and n <= cap[1] for m, n in supported):
            out.append(i)
    return out


def pick_device(prefer: str | None = None, require_gpu: bool = False) -> str:
    """Largest usable CUDA device, else CPU. `prefer` is an explicit override ("cuda:1", "cpu");
    "auto" means detect.
    """
    if prefer and prefer != "auto":
        return prefer
    usable = compatible_cuda_devices()
    if not usable:
        if require_gpu:
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
            raise SystemExit(
                f"error: --require-gpu, but none of the {n} visible CUDA device(s) is "
                f"usable by this torch build (supports {torch.cuda.get_arch_list() if n else 'n/a'})"
            )
        return "cpu"
    best = max(usable, key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    if len(usable) > 1 or torch.cuda.device_count() > len(usable):
        names = ", ".join(
            f"cuda:{i} {torch.cuda.get_device_properties(i).name}"
            f"{' (unusable)' if i not in usable else ''}"
            for i in range(torch.cuda.device_count())
        )
        print(f"CUDA devices: {names} -> using cuda:{best}")
    return f"cuda:{best}"
