import json
import logging
import os
from typing import Dict
from typing import List
from typing import Set
from typing import Union
from typing import Tuple

import hashashin
import binaryninja
from binaryninja import BinaryView
from binaryninja import ConstantReference
from binaryninja import HighLevelILInstruction
from binaryninja import enums
import numpy as np
import zlib

logger = logging.getLogger(os.path.basename(__name__))


def vec2hex(vector: np.ndarray) -> str:
    """
    Convert a vector to a hex string.

    :param vector: the vector to convert
    :return: a hex string representing the vector
    """
    assert isinstance(vector, np.ndarray)
    return "".join([f"{x:08x}" for x in vector])


def hex2vec(hex_str: str) -> np.ndarray:
    """
    Convert a hex string to a vector.

    :param hex_str: the hex string to convert
    :return: a vector representing the hex string
    """
    assert isinstance(hex_str, str)
    return np.array([int(hex_str[i: i + 8], 16) for i in range(0, len(hex_str), 8)])


def encode_feature_dict(features: Dict[str, str]) -> Dict[str, str]:
    """Encode values of dictionary using zlib"""
    return {k: zlib.compress(v.encode()).hex() for k, v in features.items()}


def decode_feature_dict(features: Dict[str, str]) -> Dict[str, str]:
    """Decode values of dictionary using zlib"""
    return {k: zlib.decompress(bytes.fromhex(v)).decode() for k, v in features.items()}


def cache_hash(bv_or_file: Union[BinaryView, str], progress=False, overwrite=False) -> Tuple[str, Dict[str, str]]:
    """Cache hash of a binary view"""
    if isinstance(bv_or_file, str):
        if os.path.exists(bv_or_file + ".hash.json") and not overwrite:
            print(f"Skipping {bv_or_file} (already exists)")
            return load_hash(bv_or_file + ".hash.json")
        bv = binaryninja.open_view(bv_or_file)
    else:
        bv = bv_or_file
    signature, features = hashashin.hash_all(bv, return_serializable=True, show_progress=progress)
    write_hash(bv.file.filename, signature, features, overwrite=overwrite)
    return signature, features


def write_hash(filepath: str, signature: str, features: Dict[str, str], overwrite=False) -> None:
    """Write hash to a file"""
    if not filepath.endswith(".hash.json"):
        filepath += ".hash.json"
    if os.path.exists(filepath) and not overwrite:
        print(f"Skipping {filepath} (already exists)")
        return
        # raise FileExistsError(f"Hash file already exists: {filepath}")
    with open(filepath, "w") as f:
        json.dump({"signature": signature, "features": encode_feature_dict(features)}, f)


def load_hash(filepath: str, generate=False, regenerate=False, progress=False) -> Tuple[str, Dict[str, str]]:
    """Load hash from a file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input path not found: {filepath}")
    if not filepath.endswith(".hash.json"):
        filepath += ".hash.json"
    if regenerate or (not os.path.exists(filepath) and generate):
        filepath = filepath.replace(".hash.json", "")  # get binary path if hash path is given
        return cache_hash(filepath, progress=progress)
    try:
        with open(filepath, "r") as f:
            binhash = json.load(f)
            return binhash["signature"], decode_feature_dict(binhash["features"])
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Invalid hash file: {filepath}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Hash file not found: {filepath}")


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


def get_cyclomatic_complexity(fn: binaryninja.function.Function) -> int:
    """
    Annotates functions with a cyclomatic complexity calculation.
    See [Wikipedia](https://en.wikipedia.org/wiki/Cyclomatic_complexity) for more information.

    Args:
        fn (binaryninja.Function): binaryninja function

    Returns:
        (int): cyclomatic complexity
    """
    n_edges = sum([len(bb.outgoing_edges) for bb in fn.basic_blocks])
    return max(1, n_edges - len(fn.basic_blocks) + 2)


def get_strings(binary_view: BinaryView, min_length: int = 5) -> Dict[int, Set[str]]:
    """
    Extracts strings used within a function.

    Args:
        binary_view: Binary Ninja binaryview to extract strings from
        min_length: Minimum length of string to extract

    Returns:
        (Dict): Dictionary of unique strings per function keyed by function address
    """
    mapping = {fn.start: set() for fn in binary_view.functions}
    for bs in binary_view.get_strings():
        if len(bs.value) < min_length:
            continue
        for ref in binary_view.get_code_refs(bs.start):
            for fn in binary_view.get_functions_containing(ref.address):
                if fn.start not in mapping:
                    logger.debug("String found in unknown function: %s", fn)
                    mapping[fn.start] = set()
                mapping[fn.start].add(bs.value)
    return mapping


def get_constants(fn: binaryninja.function.Function) -> Set[ConstantReference]:
    """
    Extracts constants used within a function.

    Args:
        fn: Binary Ninja function to extract constants from

    Returns:
        (set): set of constants
    """
    consts = []
    if not fn.hlil:
        if fn.analysis_skipped:
            fn.analysis_skipped = False
            fn.view.update_analysis_and_wait()
            assert len(list(fn.hlil.instructions)) > 0
        else:
            raise ValueError("Function has no high level IL")
    for instr in fn.hlil.instructions:
        if fn.get_constants_referenced_by(instr.address):
            consts += fn.get_constants_referenced_by_address_if_available(instr.address)
    return set(consts)


def hlil_operation_get_consts_recursive(operation: HighLevelILInstruction):
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
                ops += hlil_operation_get_consts_recursive(op)
            return ops
        if isinstance(operation, float):
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
                print(oper)
        ops += hlil_operation_get_consts_recursive(oper)
    if ops:
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
