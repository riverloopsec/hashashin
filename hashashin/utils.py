import json
import logging
import os
from typing import Dict
from typing import List
from typing import Set
from typing import Union
from typing import Tuple
from typing import Optional

from binaryninja import HighLevelILLabel
from binaryninja import HighLevelILStructField

import hashashin
import binaryninja
from binaryninja import BinaryView
from binaryninja import ConstantReference
from binaryninja import HighLevelILInstruction
from binaryninja import HighLevelILCall
from binaryninja import HighLevelILConst
from binaryninja import HighLevelILDeref
from binaryninja import HighLevelILConstPtr
from binaryninja import HighLevelILFloatConst
from binaryninja import HighLevelILAdd
from binaryninja import HighLevelILAnd
from binaryninja import HighLevelILGoto
from binaryninja import HighLevelILImport
from binaryninja import Variable
from binaryninja import SSAVariable
from binaryninja import enums
from binaryninja import Function
import numpy as np
import zlib

logger = logging.getLogger(os.path.basename(__name__))
HASH_EXT = ".hash.json"


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


def serialize_features(features: Dict[Function, np.ndarray]) -> Dict[str, str]:
    return {func2str(f): vec2hex(v) for f, v in features.items()}


def deserialize_features(
        features: Dict[str, str], bv: BinaryView
) -> Dict[Function, np.ndarray]:
    return {str2func(f, bv): hex2vec(v) for f, v in features.items()}


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


def encode_feature(feature: str) -> str:
    """Encode a feature string"""
    return zlib.compress(feature.encode()).hex()


def encode_feature_dict(features: Dict[str, str]) -> Dict[str, str]:
    """Encode values of dictionary using zlib"""
    return {k: encode_feature(v) for k, v in features.items()}


def decode_feature(feature: str) -> str:
    """Decode a feature string"""
    return zlib.decompress(bytes.fromhex(feature)).decode()


def decode_feature_dict(features: Dict[str, str]) -> Dict[str, str]:
    """Decode values of dictionary using zlib"""
    return {k: decode_feature(v) for k, v in features.items()}


def features_to_dict(features: Union[np.ndarray, str]) -> Dict[str, Union[int, str]]:
    """
    Convert a numpy array of features to a dictionary mapping feature names to their values.

    :param features: a numpy array of features
    :return: a dictionary mapping feature names to their values
    """
    if isinstance(features, str):
        try:
            features = decode_feature(features)
        except zlib.error:
            pass
        features = hex2vec(features)
    return {
        "cyclomatic_complexity": features[0],
        "num_instructions": features[1],
        "num_strings": features[2],
        "max_string_length": features[3],
        "constants": ",".join([hex(features[i]) for i in range(4, 68) if i == 4 or features[i] != 0]),
        "strings": "".join([chr(c) for c in features[68:580] if c != 0]),
    }


def dict_to_features(features: Dict[str, Union[int, str, list]]) -> np.ndarray:
    """
    Convert a dictionary of features to a numpy array of features.

    :param features: a dictionary mapping feature names to their values
    :return: a numpy array of features
    """
    vector = np.zeros(580, dtype=np.uint32)
    vector[0] = features["cyclomatic_complexity"]
    vector[1] = features["num_instructions"]
    vector[2] = features["num_strings"]
    vector[3] = features["max_string_length"]
    if isinstance(features["constants"], str):
        constants = features["constants"].split(",")
    else:
        constants = features["constants"]
    for i, c in enumerate(constants):
        vector[4 + i] = int(c, 16)
    for i, c in enumerate(features["strings"]):
        vector[68 + i] = ord(c)
    return vector


def cache_hash(
        bv_or_file: Union[BinaryView, str],
        progress=False,
        overwrite=False,
        deserialize=False,
) -> Tuple[str, Dict[str, str]]:
    """Cache hash of a binary view"""
    if isinstance(bv_or_file, str):
        if os.path.exists(bv_or_file + HASH_EXT) and not overwrite:
            print(f"Skipping {bv_or_file} (already exists)")
            return load_hash(bv_or_file + HASH_EXT)
        bv = binaryninja.open_view(bv_or_file)
    else:
        bv = bv_or_file
    signature, features = hashashin.hash_all(
        bv, return_serializable=not deserialize, show_progress=progress
    )
    write_hash(bv.file.filename, signature, features, overwrite=overwrite)
    return signature, features


def write_hash(
        filepath: str,
        signature: str,
        features: Union[Dict[str, str], Dict[Function, np.ndarray]],
        overwrite=False,
) -> None:
    """Write hash to a file"""
    if not filepath.endswith(HASH_EXT):
        filepath += HASH_EXT
    if os.path.exists(filepath) and not overwrite:
        print(f"Skipping {filepath} (already exists)")
        return
        # raise FileExistsError(f"Hash file already exists: {filepath}")
    if any(not isinstance(dv, str) for dv in features.values()):
        features = serialize_features(features)
    with open(filepath, "w") as f:
        json.dump(
            {"signature": signature, "features": encode_feature_dict(features)}, f
        )


def zero_constants(feat):
    mask = np.ones(580, dtype=np.uint32)
    mask[4:68] = 0
    return vec2hex(hex2vec(feat) * mask)


def load_hash(
        filepath: str,
        generate=False,
        regenerate=False,
        progress=False,
        deserialize=False,
        bv: Optional[BinaryView] = None,
) -> Tuple[str, Union[Dict[Function, np.ndarray], Dict[str, str]]]:
    """Load hash from a file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input path not found: {filepath}")
    bin_path = filepath
    if not filepath.endswith(HASH_EXT):
        filepath += HASH_EXT
    else:
        bin_path = filepath[: -len(HASH_EXT)]
    if regenerate or (not os.path.exists(filepath) and generate):
        return cache_hash(
            bin_path if not bv else bv, progress=progress, deserialize=deserialize, overwrite=True
        )
    try:
        with open(filepath, "r") as f:
            binhash = json.load(f)
            binhash["features"] = decode_feature_dict(binhash["features"])
            # binhash["features"] = {k: zero_constants(v) for k, v in binhash["features"].items()}
            if deserialize:
                binhash["features"] = deserialize_features(
                    binhash["features"],
                    binaryninja.open_view(bin_path) if not bv else bv,
                )
            return binhash["signature"], binhash["features"]
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


def _grab_constant_from_list(l: List) -> List[ConstantReference]:
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
        elif isinstance(param, (HighLevelILConst, HighLevelILConstPtr, HighLevelILFloatConst)):
            is_ptr = isinstance(param, HighLevelILConstPtr)
            yield ConstantReference(param.constant, size=4, pointer=is_ptr, intermediate=not is_ptr)
        elif 'operands' in dir(param):
            # TODO: Determine if we want to include DerefField offset value
            # skip things we know aren't constants
            if not isinstance(param, (HighLevelILImport, HighLevelILStructField)):
                yield from _grab_constant_from_list(param.operands)
        elif isinstance(param, int):
            yield ConstantReference(param, size=4, pointer=False, intermediate=True)
        elif not param:
            continue
        else:
            print(f'Could not get constant from {param}')


def get_constants(fn: binaryninja.function.Function) -> Set[ConstantReference]:
    """
    Extracts constants used within a function.

    Args:
        fn: Binary Ninja function to extract constants from

    Returns:
        (set): set of unique constant values
    """
    # TODO: NOTE: This is currently non-deterministic due to the way the high level IL is generated
    consts = list()
    if not fn.hlil:
        if fn.analysis_skipped:
            fn.analysis_skipped = False
            fn.view.update_analysis_and_wait()
            assert len(list(fn.hlil.instructions)) > 0
        else:
            raise ValueError("Function has no high level IL")
    for instr in fn.hlil.instructions:
        # consts += hlil_operation_get_consts_recursive(instr)
        if fn.get_constants_referenced_by(instr.address):
            consts += [c.value for c in fn.get_constants_referenced_by_address_if_available(instr.address)]
        # Fix up the binja errors, where it doesn't find constants in some cases
        # TODO: Determine if we want HighLevelILStructField operand
        # if 'operands' in dir(instr) and not isinstance(instr, (HighLevelILGoto, HighLevelILLabel)):
        #     consts += list(_grab_constant_from_list(instr.operands))
        # if not set(x.value for x in fn.get_constants_referenced_by(instr.address)).issubset(set(x.value for x in consts)):
        #     print('Might be missing constants',
        #           set(x.value for x in fn.get_constants_referenced_by(instr.address)) -
        #           set(x.value for x in consts),
        #           f'for {instr} @ {instr.address:x}')
    consts = set(consts)
    for c in list(consts):  # resolve consts set so it can be changed in the loop
        if c > 0xFFFFFFFF:
            consts.remove(c)
            consts.add(c >> 32)
            consts.add(c & 0xFFFFFFFF)
    return set(consts)


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
