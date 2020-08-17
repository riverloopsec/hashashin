# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import binaryninja as binja
from typing import Dict
import numpy as np
import hashlib
import re

# type aliases
Function = binja.function.Function
Basic_Block = binja.basicblock.BasicBlock
Binary_View = binja.binaryview.BinaryView
Vector = np.ndarray


def hash_tagged(bv: Binary_View) -> Dict[str, Function]:
    """
    Iterate over tagged functions in the binary and calculate their hash.

    :param bv: binary view encapsulating the binary
    :return: a dictionary mapping hashes to functions
    """
    sigs = {}
    for function in bv.functions:
        if len(function.address_tags) == 0:
            continue
        sigs[hash_function(function)] = function
    return sigs


def hash_all(bv: Binary_View) -> Dict[str, Function]:
    """
    Iterate over every function in the binary and calculate its hash.

    :param bv: binary view encapsulating the binary
    :return: a dictionary mapping hashes to functions
    """
    sigs = {}
    for function in bv.functions:
        sigs[hash_function(function)] = function
    return sigs


def hash_function(function: Function) -> str:
    """
    Hash a given function by "bucketing" basic blocks to capture a high level overview of their functionality, then
    performing a variation of the Weisfeiler Lehman graph similarity test on the labeled CFG.
    For more information on this process, see
    https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23.

    :param function: the function to hash
    :return: a deterministic hash of the function
    """
    # generate hyper planes to (roughly) evenly split all vectors
    h_planes = gen_planes()

    # generate vectors for each basic block
    bb_hashes = {}
    for bb in function.mlil:
        bb_hashes[bb] = bucket(vectorize(bb), h_planes)
    return weisfeiler_lehman(bb_hashes)


def weisfeiler_lehman(bbs: Dict[Basic_Block, int], iterations: int = 1) -> str:
    """
    Hash each function using a variation of the Weisfeiler-Lehman kernel.
    This allows us to account for not only the contents of each basic block, but also the overall "structure" of the CFG

    See https://blog.quarkslab.com/weisfeiler-lehman-graph-kernel-for-binary-function-analysis.html for more info.

    :param bbs: the dictionary mapping basic blocks in the function to their calculated "bucket"
    :param iterations: the number of levels of "neighbors" to account for in the Weisfeiler-Lehman kernel
    :return: a string representing the function's hash
    """
    old_labels = bbs

    # TODO: experiment with different # iterations to find a reasonable default
    new_labels = {}
    for _ in range(iterations):
        for bb in bbs:
            new_labels[bb] = ''
            incoming_bbs = [edge.source for edge in bb.incoming_edges]
            outgoing_bbs = [edge.target for edge in bb.outgoing_edges]

            for incoming in incoming_bbs:
                new_labels[bb] += old_labels[incoming]

            for outgoing in outgoing_bbs:
                new_labels[bb] += old_labels[outgoing]

        old_labels = new_labels

    # The set of labels associated with each basic block now captures the relationship between basic blocks in the CFG,
    # however, a function with many CFGs will have a very long set of labels. Hash this list again for hash consistency
    long_hash = ''.join(sorted(new_labels.values()))
    m = hashlib.sha256()
    m.update(long_hash.encode('utf-8'))
    return m.digest().hex()


def bucket(vector: Vector, h_planes: Vector) -> str:
    """
    Encode a vector's position relative to each hyper plane such that similar vectors will land in the same "bucket".

    :param vector: the vector to encode the position of
    :param h_planes: a numpy array of hyperplanes
    :return: a hex string representing the "bucket" the given vector lands in
    """
    bools = [str(int(np.dot(vector, h_plane) > 0)) for h_plane in h_planes]
    return hex(int(''.join(bools), 2))


def gen_planes(num_planes: int = 100) -> Vector:
    """
    Generates a series of arbitrary hyper planes to be used for "bucketing" in our LSH.

    :param num_planes: the number of hyper planes to generate
    :return: a numpy array containing num_planes pythonic arrays
    """
    # TODO: look into alternate (deterministic) methods of generating uniformly distributed hyperplanes
    h_planes = np.array([[(-1 ^ int((i * j) / 2)) * (i ^ j) % (int(i / (j + 1)) + 1)
                        for j in range(len(binja.MediumLevelILOperation.__members__))]
                         for i in range(num_planes)])

    return h_planes


def vectorize(bb: Basic_Block) -> Vector:
    """
    Converts a basic block into a vector representing the number of occurrences of each "type" of MLIL instruction,
    effectively forming a "bag of words".

    :param bb: the basic block to construct a vector representation of
    :return: a numpy array representing the number of occurrences of each MLIL instruction type
    """
    vector = dict((key, 0) for key in range(len(binja.MediumLevelILOperation.__members__)))
    for instr in bb:
        vector[instr.operation.value] += 1
    return np.fromiter(vector.values(), dtype=int)


def brittle_hash(bv: Binary_View, bb: Basic_Block) -> str:
    # operands are only available on an IL, ensure we're working with one
    if not bb.is_il:
        bb = bb.function.hlil.basic_blocks[bb.index]

    disassembly_text = ''.join([str(instr.operation) for instr in bb])

    # substitute out names tainted by addresses/etc.
    function_pattern = 'sub_[0-9, a-z]{6}'
    data_pattern = 'data_[0-9, a-z]{6}'
    label_pattern = 'label_[0-9, a-z]{6}'

    re.sub(function_pattern, 'function', disassembly_text)
    re.sub(data_pattern, 'data', disassembly_text)
    re.sub(label_pattern, 'label', disassembly_text)

    anchors = []
    for instr in bb:
        # TODO: construct disassembly_text inside existing bb loop here
        if instr.operation == binja.HighLevelILOperation.HLIL_ASSIGN:
            src = instr.src

            # check constant strings
            if src.operation == binja.HighLevelILOperation.HLIL_CONST_PTR:
                address = src.constant

                # filter out false positives/pointers to other data types
                anchor = bv.get_ascii_string_at(address)
                if anchor is not None:
                    anchors.append(str(anchor))

            # check arguments to function call
            elif src.operation == binja.HighLevelILOperation.HLIL_CALL:
                args = src.operands[1]  # isolate arguments from callee function
                for argument in args:
                    if argument.operation == binja.HighLevelILOperation.HLIL_CONST_PTR:
                        # filter out false positives/pointers to other data types
                        anchor = bv.get_ascii_string_at(argument.constant)
                        if anchor is not None:
                            anchors.append(str(anchor))

        elif instr.operation == binja.HighLevelILOperation.HLIL_CALL:
            args = instr.operands[1]  # isolate arguments from callee function
            for argument in args:
                if argument.operation == binja.HighLevelILOperation.HLIL_CONST_PTR:
                    # filter out false positives/pointers to other data types
                    anchor = bv.get_ascii_string_at(argument.constant)
                    if anchor is not None:
                        anchors.append(str(anchor))

    # sort anchors to guarantee consistency
    anchors.sort()
    anchor_text = ''.join(anchors)
    disassembly_text += anchor_text
    m = hashlib.sha256()
    m.update(disassembly_text.encode('utf-8'))
    return m.digest().hex()
