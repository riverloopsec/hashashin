from __future__ import annotations
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable
from typing import Union
from typing import Annotated
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from binaryninja import BasicBlock  # type: ignore
from binaryninja import BinaryView  # type: ignore
from binaryninja import Function  # type: ignore
from binaryninja import enums  # type: ignore

if TYPE_CHECKING:
    from hashashin.types import FuncSig

logger = logging.getLogger(os.path.basename(__name__))

NUM_INSTR_CATEGORIES = len(enums.MediumLevelILOperation.__members__)


@dataclass
class Feature:
    """Dataclass to represent a feature extractor."""

    def _default_repr(
        self, features: FuncSig
    ) -> tuple[str, Optional[Union[int, np.ndarray]]]:
        """
        Default string representation of a feature vector.
        :param features: feature vector
        :return: string representation (name, repr)
        """
        f_idx = FEATURES.index(self)
        offset = sum([f.size for f in FEATURES[:f_idx]])
        if features.features is None:
            return self.name, None
        if isinstance(features.features, str):
            raise
        if self.size == 1:
            return self.name, features.features[offset]
        return self.name, features.features[offset : offset + self.size]

    def _default_repr_wrapper(self, features: FuncSig) -> tuple[str, str]:
        ret = self._default_repr(features)
        return ret[0], str(ret[1])

    name: str
    size: int
    extractor: Callable[[Function], np.ndarray]
    repr: Callable[["Feature", FuncSig], tuple[str, str]] = _default_repr_wrapper

    def __call__(self, func: Function) -> np.ndarray:
        return self.extractor(func)


def split_int_to_uint32(x: int, pad=None, wrap=False) -> npt.NDArray[np.uint32]:
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


def merge_uint32_to_int(x: npt.NDArray[np.uint32]) -> int:
    """Merge array of uint32 values into a single integer. Lowest bits first."""
    ret = 0
    for i, v in enumerate(x):
        ret |= int(v) << (32 * i)
    return ret


def get_cyclomatic_complexity(fn: Function) -> int:
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


def get_strings(binary_view: BinaryView, min_length: int = 5) -> dict[int, set[str]]:
    """
    Extracts strings used within a function. Stores them in binaryview.session_data.

    Args:
        binary_view: Binary Ninja binaryview to extract strings from
        min_length: Minimum length of string to extract

    Returns:
        (Dict): Dictionary of unique strings per function keyed by function address
    """
    if "STRING_MAP" in binary_view.session_data:
        return binary_view.session_data.STRING_MAP
    mapping: dict[int, set[str]] = {fn.start: set() for fn in binary_view.functions}
    for bs in binary_view.get_strings():
        if len(bs.value) < min_length:
            continue
        for ref in binary_view.get_code_refs(bs.start):
            for fn in binary_view.get_functions_containing(ref.address):
                if fn.start not in mapping:
                    logger.debug("String found in unknown function: %s", fn)
                    mapping[fn.start] = set()
                mapping[fn.start].add(bs.value)
    binary_view.session_data["STRING_MAP"] = mapping
    return mapping


def get_constants(fn: Function) -> set[int]:
    """
    Extracts constants used within a function.

    Args:
        fn: Binary Ninja function to extract constants from

    Returns:
        (set): set of unique constant values
    """
    consts = set()
    for instr in fn.instructions:
        # consts += hlil_operation_get_consts_recursive(instr)
        if fn.get_constants_referenced_by(instr[1]):
            consts.update(
                [
                    c.value
                    for c in fn.get_constants_referenced_by_address_if_available(
                        instr[1]
                    )
                ]
            )
    for c in list(consts):  # split consts into 4 byte chunks for uint32
        if c > 0xFFFFFFFF:
            consts.remove(c)
            consts.add(c >> 32)
            consts.add(c & 0xFFFFFFFF)
        elif c < 0:  # wrap negative values
            consts.remove(c)
            consts.add(c % 0xFFFFFFFF)
    return set(consts)


def compute_instruction_histogram(
    fn: Function, timeout: int = 15
) -> npt.NDArray[np.uint32]:
    """
    Computes the instruction histogram for a function.
    :param fn: function to compute histogram for
    :param timeout: timeout for analysis in seconds
    :return: vector of instruction counts by type
    """
    vector = np.zeros(len(enums.MediumLevelILOperation.__members__), dtype=np.uint32)
    if not fn.mlil:
        fn.analysis_skipped = False
        start = time.time()
        fn.view.update_analysis()
        while (
            fn.view.analysis_progress.state != enums.AnalysisState.IdleState
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


def compute_vertex_taxonomy_histogram(fn: Function) -> npt.NDArray[np.uint32]:
    """
    Computes the vertex taxonomy histogram for a function.
    :param fn: function to compute histogram for
    :return: vector of vertex taxonomy counts by type
        0: Entry
        1: Exit
        2: Normal
    """
    # TODO: add wrap check
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


def compute_edge_taxonomy_histogram(fn: Function) -> npt.NDArray:
    """
    Computes the edge taxonomy histogram for a function.
    Gets the basic block at the start of the function
    and walks the dominator tree to count the number of
    each type of edge.
    :param fn: function to compute histogram for
    :return: vector of edge taxonomy counts by type
    """
    # TODO: fix return type hint
    entry = fn.get_basic_block_at(fn.start)
    return np.bincount(list(recursive_edge_count(entry)), minlength=4)


def _constants_repr(self: Feature, features: FuncSig) -> tuple[str, str]:
    """
    Returns a string representation of the constants stored in a feature.
    :param features: feature vector representing first 64 bytes of constants
    :return: string representation
    """
    ret = self._default_repr(features)
    if not isinstance(ret[1], np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(ret[1])}")
    return ret[0], ",".join(
        [hex(ret[1][i]) for i in range(len(ret[1])) if i == 0 or ret[1][i] != 0]
    )


def _strings_repr(self: Feature, features: FuncSig) -> tuple[str, str]:
    """
    Returns a string representation of the strings stored in a feature.
    :param features: feature vector representing first 64 bytes of strings
    :return: string representation
    """
    ret = self._default_repr(features)
    if not isinstance(ret[1], np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(ret[1])}")
    return ret[0], "".join([chr(c) for c in ret[1] if c != 0])


def _dominator_sig_repr(self: Feature, features: FuncSig) -> tuple[str, str]:
    """
    Returns a string representation of the dominator signature stored in a feature.
    :param features: feature vector representing dominator signature
    :return: string representation
    """
    ret = self._default_repr(features)
    if not isinstance(ret[1], np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(ret[1])}")
    return ret[0], hex(merge_uint32_to_int(ret[1]))


def _vertex_taxonomy_repr(self: Feature, features: FuncSig) -> tuple[str, str]:
    """
    Returns a string representation of the vertex taxonomy histogram stored in a feature.
    :param features: feature vector representing vertex taxonomy histogram
    :return: string representation
    """
    ret = self._default_repr(features)
    if not isinstance(ret[1], np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(ret[1])}")
    return ret[0], str(
        {
            "Entry": ret[1][0],
            "Exit": ret[1][1],
            "Normal": ret[1][2],
        }
    )


def _edge_taxonomy_repr(self: Feature, features: FuncSig) -> tuple[str, str]:
    """
    Returns a string representation of the edge taxonomy histogram stored in a feature.
    :param features: feature vector representing edge taxonomy histogram
    :return: string representation
    """
    ret = self._default_repr(features)
    if not isinstance(ret[1], np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(ret[1])}")
    return ret[0], str(
        {
            "Basis": ret[1][0],
            "Back": ret[1][1],
            "Forward": ret[1][2],
            "Cross": ret[1][3],
        }
    )


FEATURES = [
    Feature(
        "Cyclomatic Complexity", 1, lambda f: np.array(get_cyclomatic_complexity(f))
    ),
    Feature("Instruction Count", 1, lambda f: np.array(len(f.instructions))),
    Feature("String Count", 1, lambda f: np.array(len(get_strings(f.view)[f.start]))),
    Feature(
        "Maximum String Length",
        1,
        lambda f: np.array(max(get_strings(f.view)[f.start], key=len)),
    ),
    Feature(
        "Constants",
        64,
        lambda f: np.array(sorted(get_constants(f)), dtype=np.uint32)[:64],
        _constants_repr,
    ),
    Feature(
        "Strings",
        512,
        lambda f: np.frombuffer(
            "".join(sorted(get_strings(f.view)[f.start])).encode("utf-8"), dtype=np.byte
        )[:512],
        _strings_repr,
    ),
    Feature(
        "Instruction Histogram", NUM_INSTR_CATEGORIES, compute_instruction_histogram
    ),
    Feature(
        "Dominator Signature",
        32,
        lambda f: split_int_to_uint32(dominator_signature(f), pad=32, wrap=True),
        _dominator_sig_repr,
    ),
    Feature(
        "Vertex Histogram", 3, compute_vertex_taxonomy_histogram, _vertex_taxonomy_repr
    ),
    Feature("Edge Histogram", 4, compute_edge_taxonomy_histogram, _edge_taxonomy_repr),
]


def extract_features(fn: Function) -> FuncSig:
    """
    Computes the feature vector for a function.
    :param fn: function to compute feature vector for
    :return: feature vector
    """
    return FuncSig(fn, np.concatenate([f(fn) for f in FEATURES]))
