"""Stub torchvision so timm can import on ROCm wheels whose torchvision crashes.

Must be imported BEFORE `import timm`. Harmless on CUDA where real torchvision
already works -- we only install the stub if torchvision is missing or broken.
"""
from __future__ import annotations
import sys
import types

import torch.nn as nn


def install():
    try:
        import torchvision  # noqa: F401
        from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: F401
        return False
    except Exception:
        pass

    class _Lazy(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            sys.modules[name] = self

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            if name[:1].isupper():
                val = type(name, (), {})
                setattr(self, name, val)
                return val
            full = f"{self.__name__}.{name}"
            val = sys.modules.get(full) or _Lazy(full)
            setattr(self, name, val)
            return val

    tv = _Lazy("torchvision")
    for sub in (
        "torchvision.datasets", "torchvision.ops", "torchvision.ops.misc",
        "torchvision.models", "torchvision.models.feature_extraction",
        "torchvision.io", "torchvision.transforms",
        "torchvision.transforms.functional", "torchvision.utils",
    ):
        _Lazy(sub)

    class FrozenBatchNorm2d(nn.BatchNorm2d):
        pass

    class InterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    sys.modules["torchvision.ops.misc"].FrozenBatchNorm2d = FrozenBatchNorm2d
    sys.modules["torchvision.transforms"].InterpolationMode = InterpolationMode
    sys.modules["torchvision.transforms.functional"].InterpolationMode = InterpolationMode
    tv.ops = sys.modules["torchvision.ops"]
    tv.datasets = sys.modules["torchvision.datasets"]
    tv.transforms = sys.modules["torchvision.transforms"]
    tv.models = sys.modules["torchvision.models"]
    return True
