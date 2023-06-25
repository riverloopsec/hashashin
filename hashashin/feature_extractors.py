from __future__ import annotations

import os
import time
from typing import Union

import numpy as np
from binaryninja import BasicBlock  # type: ignore
from binaryninja import BinaryView  # type: ignore
from binaryninja import Function as BinaryNinjaFunction
from binaryninja import enums  # type: ignore

import logging

logger = logging.getLogger(__name__)

NUM_INSTR_CATEGORIES = len(enums.MediumLevelILOperation.__members__)
VERTICES = 3
EDGES = 4


class BinjaTimeoutException(Exception):
    pass


def compute_cyclomatic_complexity(fn: BinaryNinjaFunction) -> int:
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


def get_fn_strings(fn: BinaryNinjaFunction) -> set[str]:
    """Wrapper function to call get_strings for a single function"""
    return get_strings(fn.view)[fn.start]


def compute_constants(fn: BinaryNinjaFunction) -> set[int]:
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


def _compute_instruction_histogram(
    fn: BinaryNinjaFunction, timeout: int = 15
) -> list[int]:
    """
    Computes the instruction histogram for a function.
    :param fn: function to compute histogram for
    :param timeout: timeout for analysis in seconds
    :return: vector of instruction counts by type
    """
    vector: list[int] = [0] * len(enums.MediumLevelILOperation.__members__)
    if not fn.mlil:
        logger.debug(
            f"MLIL not available for function {fn.name}, rerunning analysis..."
        )
        start = time.time()
        fn.analysis_skipped = False
        fn.reanalyze()
        while (
            time.time() - start < timeout
            and fn.view.analysis_progress.state != enums.AnalysisState.IdleState
        ):
            if logger.getEffectiveLevel() <= logging.DEBUG:
                print(
                    f"Waiting for analysis to complete... {time.time() - start:.1f}/{timeout}s",
                    end="\r",
                )
            time.sleep(0.1)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            print()
        if fn.view.analysis_progress.state != enums.AnalysisState.IdleState:
            logger.debug(f"Binaryview not idle: {fn.view.analysis_progress}")
            logger.debug(f"fn.mlil: {fn.mlil_if_available}")
        if not fn.mlil_if_available:
            raise BinjaTimeoutException(
                f"MLIL not available for {fn.name} or analysis exceeded {timeout}s timeout."
            )
    for instr in fn.mlil.instructions:
        # https://youtrack.jetbrains.com/issue/PY-55734/IntEnum.value-is-not-recognized-as-a-property
        vector[instr.operation.value] += 1  # type: ignore
    return vector


def compute_instruction_histogram(
    fn: BinaryNinjaFunction, timeout: int = 15
) -> list[int]:
    try:
        return _compute_instruction_histogram(fn, timeout)
    except BinjaTimeoutException as e:
        logger.warning(e)
        return [0] * len(enums.MediumLevelILOperation.__members__)


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


def compute_dominator_signature(entry: Union[BasicBlock, BinaryNinjaFunction]) -> int:
    """
    Computes the dominator signature for a function or basic block.
    :param entry: entry block to compute signature for
    :return: dominator signature as integer
    """
    if isinstance(entry, BinaryNinjaFunction):
        entry = entry.basic_blocks[0]
    assert (
        len(entry.dominators) == 1 and entry.dominators[0] == entry
    ), f"Entry block should not be dominated: {entry}\t{entry.dominators}"
    return int("".join(map(str, walk(entry))), 2)


def compute_vertex_taxonomy_histogram(fn: BinaryNinjaFunction) -> list[int]:
    """
    Computes the vertex taxonomy histogram for a function.
    :param fn: function to compute histogram for
    :return: vector of vertex taxonomy counts by type
        0: Entry
        1: Exit
        2: Normal
    """
    vector = [0] * VERTICES
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


def compute_edge_taxonomy_histogram(fn: BinaryNinjaFunction) -> list[int]:
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
    return list(np.bincount(list(recursive_edge_count(entry)), minlength=EDGES))
