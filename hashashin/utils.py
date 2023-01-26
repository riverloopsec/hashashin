from __future__ import annotations
import glob
import json
import logging
import os
import zlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Iterator
from typing import Union
from typing import TYPE_CHECKING

import binaryninja  # type: ignore
import magic
import numpy as np
import numpy.typing as npt
from binaryninja import BinaryView
from binaryninja import ConstantReference
from binaryninja import Function
from binaryninja import HighLevelILConst
from binaryninja import HighLevelILConstPtr
from binaryninja import HighLevelILFloatConst
from binaryninja import HighLevelILImport
from binaryninja import HighLevelILInstruction
from binaryninja import HighLevelILStructField
from binaryninja import SSAVariable
from binaryninja import Variable
from binaryninja import enums
from tqdm import tqdm  # type: ignore

logger = logging.getLogger(os.path.basename(__name__))

def get_binaries(path, bin_name=None, recursive=True, progress=False):
    """Get all binaries in a directory"""
    if os.path.isfile(path):
        files = [path]
    elif bin_name is None:
        files = glob.glob(f"{path}/**", recursive=recursive)
    else:
        files = glob.glob(f"{path}/**/{bin_name}", recursive=recursive)
    binaries = []
    if not progress:
        print(
            f"Iterating over {len(files)} files. If you see this, consider using --progress.",
            end="\r",
        )
    elif len(files) == 1:
        progress = False
    for f in tqdm(
        files,
        disable=not progress,
        desc=f"Gathering binaries in {os.path.relpath(path)}",
    ):
        if os.path.isfile(f):
            if "ELF" in magic.from_file(f):
                binaries.append(f)
    return binaries

def split_int_to_uint32(x: int, pad=None, wrap=False) -> npt.NDArray[np.uint32]:
    """Split very large integers into array of uint32 values. Lowest bits are first."""
    if x < np.iinfo(np.uint32).max:
        return np.array([x], dtype=np.uint32)
    if pad is None:
        pad = int(np.ceil(len(bin(x)) / 32))
    elif pad < int(np.ceil(len(bin(x)) / 32)):
        if wrap:
            logger.warning(f"Padding is too small for number {x}, wrapping")
            x = x % (2 ** (32 * pad))
        else:
            raise ValueError("Padding is too small for number")
    ret = np.array([(x >> (32 * i)) & 0xFFFFFFFF for i in range(pad)], dtype=np.uint32)
    assert merge_uint32_to_int(ret) == x, f"{merge_uint32_to_int(ret)} != {x}"
    if merge_uint32_to_int(ret) != x:
        logger.warning(f"{merge_uint32_to_int(ret)} != {x}")
        raise ValueError("Splitting integer failed")
    return ret


def merge_uint32_to_int(x: npt.NDArray[np.uint32]) -> int:
    """Merge array of uint32 values into a single integer. Lowest bits first."""
    ret = 0
    for i, v in enumerate(x):
        ret |= int(v) << (32 * i)
    return ret