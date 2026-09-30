"""Portable SB3 archives, including compatibility with nested-ZIP readers on Windows."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import open_path


def portable_archive(source) -> io.BytesIO:
    """Preserve SB3 metadata and tensors; avoid a nested PyTorch ZIP stream.

    PyTorch's supported legacy tensor serialization avoids the ZipExtFile/miniz
    failure observed with SB3 2.9 / torch 2.14 on Windows. No global patches or
    changes to installed packages are required. Only load trusted model files.
    """
    target = io.BytesIO()
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as archive:
        for name in original.namelist():
            data = original.read(name)
            if name.endswith(".pth"):
                tensors = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                buffer = io.BytesIO()
                torch.save(tensors, buffer, _use_new_zipfile_serialization=False)
                data = buffer.getvalue()
            archive.writestr(name, data)
    target.seek(0)
    return target


class PortablePPO(PPO):
    def save(self, path, exclude=None, include=None) -> None:
        buffer = io.BytesIO()
        super().save(buffer, exclude=exclude, include=include)
        buffer.seek(0)
        portable = portable_archive(buffer)
        stream = open_path(path, "w", suffix="zip")
        try:
            stream.write(portable.getvalue())
        finally:
            if isinstance(path, (str, Path)):
                stream.close()


def load_ppo(path, **kwargs) -> PPO:
    """Read both existing nested archives and new portable ones without mutation."""
    return PPO.load(portable_archive(path), **kwargs)
