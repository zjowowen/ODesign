from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn as nn


PROFILE_DETAIL_ATTR = "_odesign_profile_pairformer_detail"


def set_odesign_profile_detail_enabled(
    module: nn.Module, enabled: bool
) -> None:
    """Recursively enable optional ODesign record_function ranges."""
    for child in module.modules():
        setattr(child, PROFILE_DETAIL_ATTR, bool(enabled))


def odesign_record_function(module: nn.Module, name: str):
    if not getattr(module, PROFILE_DETAIL_ATTR, False):
        return nullcontext()
    return torch.autograd.profiler.record_function(name)


def odesign_profile_call(
    module: nn.Module,
    name: str,
    fn: Callable,
    *args,
    **kwargs,
):
    with odesign_record_function(module, name):
        return fn(*args, **kwargs)
