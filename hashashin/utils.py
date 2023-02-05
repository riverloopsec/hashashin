from __future__ import annotations
import glob
import json
import logging
import os
from pathlib import Path
from typing import Union, Iterable

import binaryninja  # type: ignore
import magic
import numpy as np
import numpy.typing as npt
from tqdm import tqdm  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hashashin")


def get_binaries(
    path: Union[Path, Iterable[Path]], bin_name=None, recursive=True, progress=False
) -> list[Path]:
    """Get all binaries in a directory"""
    globber = "*" + "*" * recursive
    if isinstance(path, Iterable):
        files = []
        for p in path:
            files.extend(get_binaries(p, bin_name, recursive, progress))
        return files
    elif os.path.isfile(path):
        files = [path]
    elif bin_name is None:
        files = list(path.glob(globber + "/*"))
    else:
        files = list(path.glob(globber + f"/{bin_name}"))
    if (Path(path) / ".bin.idx").is_file():
        logger.debug(f"Loading cached binary paths from {Path(path) / '.bin.idx'}")
        with open(Path(path) / ".bin.idx", "r") as f:
            binaries = [Path(b) for b in json.load(f)]
        return binaries
    binaries = []
    if not progress:
        if "progress_warning" not in dir(logger):
            logger.info(
                f"Iterating over {len(files)} files. If you see this, consider using --progress.",
            )
            logger.progress_warning = True
    elif len(files) == 1:
        progress = False
    for f in tqdm(
        files,
        disable=not progress,
        desc=f"Gathering binaries in {os.path.relpath(path)}",
    ):
        if f.is_file():
            if "ELF" in magic.from_file(f):
                binaries.append(f)
    if Path(path).is_dir():
        # cache list of binaries in directory
        with open(path / ".bin.idx", "w") as f:
            json.dump([str(b) for b in binaries], f)
    return binaries


def split_int_to_uint32(x: int, pad=None, wrap=False) -> npt.NDArray[np.uint32]:
    """Split very large integers into array of uint32 values. Lowest bits are first."""
    global logger
    _logger = logger.getChild("utils.split_int_to_uint32")
    if x < np.iinfo(np.uint32).max:
        if pad is None:
            return np.array([x], dtype=np.uint32)
        return np.pad([x], (0, pad - 1), "constant", constant_values=(0, 0))
    if pad is None:
        pad = int(np.ceil(len(bin(x)) / 32))
    elif pad < int(np.ceil(len(bin(x)) / 32)):
        if wrap:
            _logger.debug(f"Padding is too small for number {x}, wrapping")
            x = x % (2 ** (32 * pad))
        else:
            raise ValueError("Padding is too small for number")
    ret = np.array([(x >> (32 * i)) & 0xFFFFFFFF for i in range(pad)], dtype=np.uint32)
    assert merge_uint32_to_int(ret) == x, f"{merge_uint32_to_int(ret)} != {x}"
    if merge_uint32_to_int(ret) != x:
        _logger.warning(f"{merge_uint32_to_int(ret)} != {x}")
        raise ValueError("Splitting integer failed")
    return ret


def merge_uint32_to_int(x: npt.NDArray[np.uint32]) -> int:
    """Merge array of uint32 values into a single integer. Lowest bits first."""
    ret = 0
    for i, v in enumerate(x):
        ret |= int(v) << (32 * i)
    return ret


def bytes_distance(a: bytes, b: bytes) -> float:
    """Calculate the distance between two byte strings"""
    return np.mean(np.frombuffer(a, dtype=np.uint) != np.frombuffer(b, dtype=np.uint8))
