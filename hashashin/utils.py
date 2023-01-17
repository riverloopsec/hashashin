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

if TYPE_CHECKING:
    from hashashin.types import BinSig, FuncSig

logger = logging.getLogger(os.path.basename(__name__))


def func2str(func: binaryninja.Function) -> str:
    """
    Convert a function to a string.
    :param func: Function to convert
    :return: String representation of the function
    """
    return f"{func} @ 0x{func.start:X}"


def str2func(func_str: str, bv: BinaryView) -> Function:
    """
    Convert a function string to a function.
    :param func_str: the function string
    :param bv: the binary view

    :return (Function): the function
    """
    func_name, func_addr = func_str.split(" @ 0x")
    func = bv.get_function_at(int(func_addr, 16))
    if func.name not in func_name:
        raise ValueError(f"Function name mismatch: {func_name} != {func.name}")
    return func


def serialize_features(features: Dict[Function, FuncSig]) -> Dict[str, str]:
    return {func2str(f): vec2hex(v) for f, v in features.items()}


def deserialize_features(
        features: Dict[str, str], bv: BinaryView
) -> Dict[Function, np.ndarray]:
    return {str2func(f, bv): hex2vec(v) for f, v in features.items()}


# def vec2hex(vector: np.ndarray) -> str:
#     """
#     Convert a vector to a hex string.
#
#     :param vector: the vector to convert
#     :return: a hex string representing the vector
#     """
#     assert isinstance(vector, np.ndarray)
#     return "".join([f"{x:08x}" for x in vector])
#
#
# def hex2vec(hex_str: str) -> npt.NDArray[np.int64]:
#     """
#     Convert a hex string to a vector.
#
#     :param hex_str: the hex string to convert
#     :return: a vector representing the hex string
#     """
#     assert isinstance(hex_str, str)
#     return np.array([int(hex_str[i: i + 8], 16) for i in range(0, len(hex_str), 8)])


# def encode_feature(feature: str) -> str:
#     """Encode a feature string"""
#     return zlib.compress(feature.encode()).hex()
#
#
# def encode_feature_dict(features: Dict[str, str]) -> Dict[str, str]:
#     """Encode values of dictionary using zlib"""
#     return {k: encode_feature(v) for k, v in features.items()}
#
#
# def decode_feature(feature: str) -> str:
#     """Decode a feature string"""
#     return zlib.decompress(bytes.fromhex(feature)).decode()
#
#
# def decode_feature_dict(features: dict[str, str]) -> dict[str, str]:
#     """Decode values of dictionary using zlib"""
#     return {k: decode_feature(v) for k, v in features.items()}


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
            f"Iterating over {len(files)} files. If you see this, consider using --progress.", end='\r'
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


def zero_constants(feat):
    mask = np.ones(580, dtype=np.uint32)
    mask[4:68] = 0
    return vec2hex(hex2vec(feat) * mask)


def minhash_similarity(a: Union[np.ndarray, str], b: Union[np.ndarray, str]) -> float:
    """
    Compute the minhash similarity between two vectors.

    Args:
        a: vector a
        b: vector b

    Returns:
        (float): minhash similarity
    """
    if isinstance(a, str):
        a = hex2vec(a)
    if isinstance(b, str):
        b = hex2vec(b)
    return (a == b).sum() / len(a)


def jaccard_similarity(a: Union[Set, Dict], b: Union[Set, Dict]) -> float:
    """
    Compute the Jaccard similarity between two sets.

    Args:
        a: set a
        b: set b

    Returns:
        (float): Jaccard similarity
    """
    if isinstance(a, dict):
        a = set(a.values())
    if isinstance(b, dict):
        b = set(b.values())
    return len(a.intersection(b)) / len(a.union(b))


def distance(a: Union[Set, Dict], b: Union[Set, Dict]) -> int:
    """
    Compute the distance between two sets.

    Args:
        a: set a
        b: set b

    Returns:
        (int): distance
    """
    if isinstance(a, dict):
        a = set(a.values())
    if isinstance(b, dict):
        b = set(b.values())
    return np.sum(np.dist(map(hex2vec, a), map(hex2vec, b)))


def _grab_constant_from_list(l: List) -> Iterator[ConstantReference]:
    """
    Recursively grab all constants from a list.
    :param l: List of HLIL operands
    :return: generator of ConstantReferences
    """
    for param in l:
        if isinstance(param, list):
            yield from _grab_constant_from_list(param)
        elif isinstance(param, (Variable, SSAVariable)):
            continue
        elif isinstance(
                param, (HighLevelILConst, HighLevelILConstPtr, HighLevelILFloatConst)
        ):
            is_ptr = isinstance(param, HighLevelILConstPtr)
            yield ConstantReference(
                param.constant, size=4, pointer=is_ptr, intermediate=not is_ptr
            )
        elif "operands" in dir(param):
            # TODO: Determine if we want to include DerefField offset value
            # skip things we know aren't constants
            if not isinstance(param, (HighLevelILImport, HighLevelILStructField)):
                yield from _grab_constant_from_list(param.operands)
        elif isinstance(param, int):
            yield ConstantReference(param, size=4, pointer=False, intermediate=True)
        elif not param:
            continue
        else:
            print(f"Could not get constant from {param}")


def hlil_operation_get_consts_recursive(operation: HighLevelILInstruction, silent=True):
    """Extract constants recursively using HLIL operations.

    Args:
        operation (binaryninja.highlevelil.HighLevelILInstruction):
            Seed initially with an HLIL instruction.
    """
    if not hasattr(operation, "operands"):
        if isinstance(operation, int):
            return [operation]
        if isinstance(operation, list):
            ops = []
            for op in operation:
                ops += hlil_operation_get_consts_recursive(op, silent)
            return ops
        if isinstance(operation, float):
            if float(int(operation)) == operation:
                return [int(operation)]
            #  return [operation]  # TODO: C++ backend expects a set of ints, and fails to handle floats
            return []
        if isinstance(operation, binaryninja.variable.Variable):
            return []
        if isinstance(operation, binaryninja.lowlevelil.ILIntrinsic):
            # TODO: sometimes HLIL instructions encapsulate IL intrinsics - look into if they may have meaningful inputs
            return []
        if operation is None:
            return []
    if operation.operation == enums.HighLevelILOperation.HLIL_VAR:
        #  return [operation.value.value] # TODO: we seem to get a lot of inaccurate values from this, disabling for now
        return []

    operands = filter_address_operations(operation)
    ops = []
    for oper in operands:
        if isinstance(oper, int):
            if len(operation.function.view.get_sections_at(oper)) == 0:
                ops.append(oper)
            else:
                if not silent:
                    print(oper)
        ops += hlil_operation_get_consts_recursive(oper, silent)
    if not silent and ops:
        print(ops)
    return ops


def filter_address_operations(
        instr: HighLevelILInstruction,
) -> List[HighLevelILInstruction]:
    op = instr.operation

    addr_only = [
        enums.HighLevelILOperation.HLIL_GOTO,
        enums.HighLevelILOperation.HLIL_LABEL,
        enums.HighLevelILOperation.HLIL_JUMP,
        enums.HighLevelILOperation.HLIL_IMPORT,
        # NOTE: HLIL_DEREF will miss constants in the form of `*(var + 0x30)` - these are likely not "interesting" constants
        enums.HighLevelILOperation.HLIL_DEREF,
        enums.HighLevelILOperation.HLIL_CONST_PTR,
        enums.HighLevelILOperation.HLIL_EXTERN_PTR,
        enums.HighLevelILOperation.HLIL_VAR_DECLARE,
        # enums.HighLevelILOperation.HLIL_RET,
        enums.HighLevelILOperation.HLIL_BLOCK,  # should never come up, but filter out blocks to avoid double counting
    ]
    const_first = [
        enums.HighLevelILOperation.HLIL_IF,
        enums.HighLevelILOperation.HLIL_WHILE,
        enums.HighLevelILOperation.HLIL_FOR,
        enums.HighLevelILOperation.HLIL_SWITCH,
        enums.HighLevelILOperation.HLIL_CASE,
        enums.HighLevelILOperation.HLIL_DO_WHILE,
    ]
    const_second = [enums.HighLevelILOperation.HLIL_VAR_INIT]

    if op in addr_only:
        return []
    if op in const_first:
        return [instr.operands[0]]
    if op in const_second:
        return [instr.operands[1]]
    # calls are a special case, since second operand is already a list
    if (op == enums.HighLevelILOperation.HLIL_CALL) or (
            op == enums.HighLevelILOperation.HLIL_TAILCALL
    ):
        return instr.operands[1]
    return instr.operands
