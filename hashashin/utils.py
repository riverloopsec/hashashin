import json
import logging
import os
import time
import zlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import binaryninja
import numpy as np
import magic
import glob
from tqdm import tqdm
from binaryninja import BasicBlock
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

import hashashin

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
    return np.array([int(hex_str[i : i + 8], 16) for i in range(0, len(hex_str), 8)])


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


def get_binaries(path, bin_name=None, recursive=True, progress=False):
    """Get all binaries in a directory"""
    if os.path.isfile(path):
        files = [path]
    elif bin_name is None:
        files = glob.glob(f"{path}/**", recursive=recursive)
    else:
        files = glob.glob(f"{path}/**/{bin_name}", recursive=recursive)
    binaries = []
    for f in tqdm(files, disable=not progress):
        if os.path.isfile(f):
            if "ELF" in magic.from_file(f):
                binaries.append(f)
    return binaries


def split_int_to_uint32(x: int, pad=None, wrap=False) -> np.ndarray[int]:
    """Split very large integers into array of uint32 values. Lowest bits are first."""
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


def merge_uint32_to_int(x: np.ndarray[int]) -> int:
    """Merge array of uint32 values into a single integer. Lowest bits first."""
    ret = 0
    for i, v in enumerate(x):
        ret |= int(v) << (32 * i)
    return ret


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
    ilen = len(enums.MediumLevelILOperation.__members__)
    return {
        "cyclomatic_complexity": features[0],
        "num_instructions": features[1],
        "num_strings": features[2],
        "max_string_length": features[3],
        "constants": ",".join(
            [hex(features[i]) for i in range(4, 68) if i == 4 or features[i] != 0]
        ),
        "strings": "".join([chr(c) for c in features[68:580] if c != 0]),
        "histogram": ",".join(map(str, features[580 : 580 + ilen])),
        "dominator_sig": hex(
            merge_uint32_to_int(features[580 + ilen : 580 + ilen + 32])
        ),
        "vertex_hist": {
            "Entry": features[580 + ilen + 32],
            "Exit": features[580 + ilen + 33],
            "Normal": features[580 + ilen + 34],
        },
        "edge_hist": {
            "Basis": features[580 + ilen + 35],
            "Back": features[580 + ilen + 36],
            "Forward": features[580 + ilen + 37],
            "Cross": features[580 + ilen + 38],
        },
    }


def dict_to_features(features: Dict[str, Union[str, list]]) -> np.ndarray:
    """
    Convert a dictionary of features to a numpy array of features.

    :param features: a dictionary mapping feature names to their values
    :return: a numpy array of features
    """
    ilen = len(enums.MediumLevelILOperation.__members__)
    vector = np.zeros(580 + ilen + 32, dtype=np.uint32)
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
    for i, c in enumerate(features["histogram"]):
        vector[580 + i] = c
    for i, c in enumerate(features["dominator_sig"]):
        vector[580 + ilen + i] = c
    vector[612 + ilen] = features["vertex_hist"]["Entry"]
    vector[613 + ilen] = features["vertex_hist"]["Exit"]
    vector[614 + ilen] = features["vertex_hist"]["Normal"]
    vector[615 + ilen] = features["edge_hist"]["Basis"]
    vector[616 + ilen] = features["edge_hist"]["Back"]
    vector[617 + ilen] = features["edge_hist"]["Forward"]
    vector[618 + ilen] = features["edge_hist"]["Cross"]
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
            bin_path if not bv else bv,
            progress=progress,
            deserialize=deserialize,
            overwrite=True,
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


def get_constants(fn: binaryninja.function.Function) -> Set[ConstantReference]:
    """
    Extracts constants used within a function.

    Args:
        fn: Binary Ninja function to extract constants from

    Returns:
        (set): set of unique constant values
    """
    consts = list()
    for instr in fn.instructions:
        # consts += hlil_operation_get_consts_recursive(instr)
        if fn.get_constants_referenced_by(instr[1]):
            consts += [
                c.value
                for c in fn.get_constants_referenced_by_address_if_available(instr[1])
            ]
    consts = set(consts)
    for c in list(consts):  # split consts into 4 byte chunks
        if c > 0xFFFFFFFF:
            consts.remove(c)
            consts.add(c >> 32)
            consts.add(c & 0xFFFFFFFF)
    return set(consts)


def compute_instruction_histogram(
    fn: binaryninja.Function, timeout: int = 15
) -> np.ndarray[int, ...]:
    """
    Computes the instruction histogram for a function.
    :param fn: function to compute histogram for
    :param timeout: timeout for analysis in seconds
    :return: vector of instruction counts by type
    """
    vector = np.zeros(
        len(binaryninja.enums.MediumLevelILOperation.__members__), dtype=np.uint32
    )
    if not fn.mlil:
        fn.analysis_skipped = False
        start = time.time()
        fn.view.update_analysis()
        while (
            fn.view.analysis_progress.state != binaryninja.enums.AnalysisState.IdleState
            and fn.mlil is None
            and time.time() - start < timeout
        ):
            time.sleep(0.1)
        if not fn.mlil:
            raise ValueError(
                f"MLIL not available for function or analysis exceeded {timeout}s timeout."
            )
    for instr in fn.mlil_instructions:
        if vector[instr.operation.value] >= np.iinfo(np.uint32).max:
            logger.warning("Instruction count overflow for %s", instr.operation.name)
        vector[instr.operation.value] += 1
    return vector


def walk(block, seen=None):
    """
    Recursively walks the basic block graph starting at the given block.
    :param block: starting block
    :param seen: set of blocks already seen
    :return: generator representing the dfs order of the graph
    """
    if seen is None:
        seen = set()
    if block not in seen:
        seen.add(block)
        yield 1
    else:
        yield 0
    for child in block.dominator_tree_children:
        yield from walk(child, seen)
        yield 0


def dominator_signature(entry: Union[BasicBlock, Function]) -> int:
    """
    Computes the dominator signature for a function or basic block.
    :param entry: entry block to compute signature for
    :return: dominator signature as integer
    """
    if isinstance(entry, Function):
        entry = entry.basic_blocks[0]
    assert (
        len(entry.dominators) == 1 and entry.dominators[0] == entry
    ), f"Entry block should not be dominated: {entry}\t{entry.dominators}"
    return int("".join(map(str, walk(entry))), 2)


def compute_vertex_taxonomy_histogram(fn: binaryninja.Function) -> np.ndarray[int, ...]:
    """
    Computes the vertex taxonomy histogram for a function.
    :param fn: function to compute histogram for
    :return: vector of vertex taxonomy counts by type
        0: Entry
        1: Exit
        2: Normal
    """
    vector = np.zeros(3, dtype=np.uint32)
    for block in fn.basic_blocks:
        if block.dominators == [block]:
            vector[0] += 1
        elif len(block.dominator_tree_children) == 0:
            vector[1] += 1
        else:
            vector[2] += 1
    return vector


def recursive_edge_count(block, start_time=None, end_time=None, seen=None, dfs_time=0):
    """
    Yields the type of edges in the dominator tree rooted at block
    https://www.geeksforgeeks.org/tree-back-edge-and-cross-edges-in-dfs-of-graph/
    :param block: block to start counting from
    :param start_time: time at start of dfs for vertex
    :param end_time: time at end of dfs for vertex
    :param seen: track if the vertex has been seen
    :param dfs_time: starting time for dfs
    :return: generator of each edge's corresponding classification
        0: Basis edge
        1: Back edge
        2: Forward edge
        3: Cross edge
    """
    if not start_time:
        start_time = {}
    if not end_time:
        end_time = {}
    if not seen:
        seen = set()
    seen.add(block)
    start_time[block] = dfs_time
    dfs_time += 1
    for child in block.dominator_tree_children:
        if child not in seen:
            yield 0  # basis edge
            yield from recursive_edge_count(child, start_time, end_time, seen, dfs_time)
        else:
            # parent traversed after child
            if (
                start_time[block] > start_time[child]
                and end_time[block] < end_time[child]
            ):
                yield 1
            # child not part of dfs tree
            elif (
                start_time[block] < start_time[child]
                and end_time[block] > end_time[child]
            ):
                yield 2
            # parent and child have no ancestor-descendant relationship
            elif (
                start_time[block] > start_time[child]
                and end_time[block] > end_time[child]
            ):
                yield 3
            else:
                raise ValueError(f"Unknown edge classification for {block} -> {child}")
        end_time[block] = dfs_time
        dfs_time += 1


def compute_edge_taxonomy_histogram(fn: binaryninja.Function) -> np.ndarray[int, ...]:
    """
    Computes the edge taxonomy histogram for a function.
    Gets the basic block at the start of the function
    and walks the dominator tree to count the number of
    each type of edge.
    :param fn: function to compute histogram for
    :return: vector of edge taxonomy counts by type
    """
    entry = fn.get_basic_block_at(fn.start)
    return np.bincount(list(recursive_edge_count(entry)), minlength=4)


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
