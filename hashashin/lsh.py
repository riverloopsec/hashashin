# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Authors Rylan O'Connell and Jonathan Prokos

import hashlib
import logging
import os
import re
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import binaryninja as binja
import numpy as np
from binaryninja import BasicBlock
from binaryninja import BinaryView
from binaryninja import Function
from binaryninja import enums
from binaryninja import open_view
from tqdm import tqdm

from hashashin.utils import compute_edge_taxonomy_histogram
from hashashin.utils import compute_instruction_histogram
from hashashin.utils import compute_vertex_taxonomy_histogram
from hashashin.utils import dominator_signature
from hashashin.utils import func2str
from hashashin.utils import get_constants
from hashashin.utils import get_cyclomatic_complexity
from hashashin.utils import get_strings
from hashashin.utils import serialize_features
from hashashin.utils import split_int_to_uint32
from hashashin.utils import vec2hex
from hashashin.utils import write_hash
from hashashin.utils import features_to_dict
from hashashin.utils import load_hash
from hashashin.utils import get_binaries

logger = logging.getLogger(os.path.basename(__name__))
SIGNATURE_LEN = 20


def hash_tagged(bv: BinaryView) -> Dict[str, Function]:
    """
    Iterate over tagged functions in the binary and calculate their hash.

    :param bv: binary view encapsulating the binary
    :return: a dictionary mapping hashes to functions
    """
    raise NotImplementedError()
    sigs = {}
    h_planes = gen_planes()
    for function in bv.functions:
        if len(function.address_tags) == 0:
            continue
        sigs[hash_function(function, h_planes)] = function
    return sigs


def hash_all(
    bv: BinaryView,
    return_serializable: bool = False,
    show_progress: bool = False,
    save_to_file: bool = False,
) -> Tuple[str, Union[Dict[Function, np.ndarray], Dict[str, str]]]:
    """
    Iterate over every function in the binary and calculate its hash.

    :param bv: binary view encapsulating the binary
    :param return_serializable: if true, return a serializable dictionary mapping function name and address to hash
    :param show_progress: if true, show a progress bar while hashing functions
    :param save_to_file: if true, save the hashes to a file
    :return: a dictionary mapping signatures to sets of functions and a dictionary mapping functions to feature maps
    """
    string_map = get_strings(bv)
    features = {}
    h_planes = gen_planes(20)
    # baseline_sig, baseline_feats = hashashin.utils.load_hash(
    #     bv.file.filename, deserialize=True, bv=bv
    # )
    print(
        "Hashing functions... (if you are seeing this, you should try showing progress with --progress)",
        end="\r",
    )
    for function in (pbar := tqdm(bv.functions, disable=not show_progress)):
        pbar.set_description(f"Hashing {func2str(function)}")
        feature = hash_function(function, h_planes, string_map=string_map)
        features[function] = feature
        # if vec2hex(features[function]) != vec2hex(baseline_feats[function]):
        #     print("ok this is the problem")
    signature = min_hash(np.stack(features.values()))
    if return_serializable:
        features = serialize_features(features)
    if save_to_file:
        write_hash(bv.file.filename, signature, features)
    return signature, features


def min_hash(
    features: np.ndarray, sig_len: int = SIGNATURE_LEN, seed: int = 2022
) -> str:
    """
    Generate a minhash signature for a given set of features.

    :param features: a matrix of vectorized features
    :param sig_len: the length of the minhash signature
    :param seed: a seed for the random number generator
    :return: a string representing the minhash signature
    """
    np.random.seed(seed)
    # h(x) = (ax + b) % c
    a = np.random.randint(0, 2**32 - 1, size=sig_len)
    b = np.random.randint(0, 2**32 - 1, size=sig_len)
    c = 4297922131  # prime number above 2**32-1

    b = np.stack([np.stack([b] * features.shape[0])] * features.shape[1]).T
    hashed_features = (np.tensordot(a, features, axes=0) + b) % c
    minhash = hashed_features.min(axis=(1, 2))
    return vec2hex(minhash)


def hash_function(
    function: Function,
    h_planes: Optional[np.ndarray] = None,
    string_map=None,
) -> np.ndarray:
    """
    Hash a given function by "bucketing" basic blocks to capture a high level overview of their functionality, then
    performing a variation of the Weisfeiler Lehman graph similarity test on the labeled CFG.
    For more information on this process, see
    https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23.

    :param string_map: a dictionary of strings in each function
    :param function: the function to hash
    :param h_planes: a numpy array of hyperplanes (if none are provided the default values will be used)
    :return: a deterministic hash of the function
    """
    if not string_map:
        string_map = get_strings(function.view)
    if h_planes is None:
        h_planes = gen_planes(20)
    features = extract_features(function, string_map=string_map, h_planes=h_planes)
    # if function.start == 141836:
    #     print('hey now')
    # if function.start == 141836 and vec2hex(features) != '':
    #     print('wtf')
    return features


def extract_features(
    function: Function, string_map=None, h_planes=None
) -> np.ndarray[int, ...]:
    """
    Extract features from a function to be used in the hashing process.
    This includes ideas from https://docs.google.com/document/d/1ZphECbrEUTw4WNepQQ2KrABll0JXHUjElEfXVjbV4Jg/edit
    Sizes (uint32):
        - 1: cyclomatic complexity
        - 1: number of instructions
        - 1: number of strings
        - 1: maximum string length
        - 64: constants
        - 512: strings
        - len(enums.MediumLevelILOperation.__members__) [132]: instruction histogram
        - 32: dominator tree signature broken into 32 32-bit integers
        - 3: taxonomy of vertices
        - 4: taxonomy of edges

    :param string_map: dictionary of strings in each function
    :param function: the function to extract features from
    :return: a vector representing the function's features (np.int32, 1xN)
    """
    if not string_map:
        string_map = get_strings(function.view)
    if h_planes is None:
        h_planes = gen_planes(20)
    num_instr_categories = len(enums.MediumLevelILOperation.__members__)
    features = np.zeros(
        4 + 64 + 512 + num_instr_categories + 32 + 3 + 4, dtype=np.uint32
    )
    strings = sorted(string_map[function.start])

    offset = 0
    features[offset] = get_cyclomatic_complexity(function)
    offset += 1
    features[offset] = len(list(function.instructions))
    offset += 1
    features[offset] = len(strings)  # number of strings
    offset += 1
    features[offset] = (
        len(max(strings, key=len)) if strings else 0
    )  # maximum string length
    offset += 1

    constants = get_constants(function)
    constants = np.array(sorted(constants), dtype=np.int32)
    if len(constants) > 64:
        constants = constants[:64]
    features[offset : offset + 64] = np.pad(
        constants, (0, 64 - len(constants)), "constant", constant_values=0
    )
    offset += 64

    # add first 512 bytes of strings
    strings = "".join(strings)
    if len(strings) > 512:
        strings = strings[:512]
    strings = np.frombuffer(strings.encode("utf-8"), dtype=np.byte)
    features[offset : offset + 512] = np.pad(
        strings, (0, 512 - len(strings)), "constant", constant_values=0
    )
    offset += 512

    # instruction histogram
    features[offset : offset + num_instr_categories] = compute_instruction_histogram(
        function
    )
    offset += num_instr_categories

    # If it is a tiny function just stop early to save compute
    if len(function.basic_blocks) == 1:
        return features

    # dominator signature
    sig = dominator_signature(function)
    features[offset : offset + 32] = split_int_to_uint32(sig, pad=32, wrap=True)
    offset += 32

    # vertex taxonomy histogram
    features[offset : offset + 3] = compute_vertex_taxonomy_histogram(function)
    offset += 3

    # edge taxonomy histogram
    features[offset : offset + 4] = compute_edge_taxonomy_histogram(function)
    offset += 4

    return features


def weisfeiler_lehman(bbs: Dict[BasicBlock, int], iterations: int = 1) -> str:
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
            new_labels[bb] = ""
            incoming_bbs = [edge.source for edge in bb.incoming_edges]
            outgoing_bbs = [edge.target for edge in bb.outgoing_edges]

            for incoming in incoming_bbs:
                new_labels[bb] += old_labels[incoming]

            for outgoing in outgoing_bbs:
                new_labels[bb] += old_labels[outgoing]

        old_labels = new_labels

    # The set of labels associated with each basic block now captures the relationship between basic blocks in the CFG,
    # however, a function with many CFGs will have a very long set of labels. Hash this list again for hash consistency
    long_hash = "".join(sorted(new_labels.values()))
    m = hashlib.sha256()
    m.update(long_hash.encode("utf-8"))
    return m.digest().hex()


def hash_basic_block(bb: BasicBlock, h_planes: Optional[np.ndarray] = None) -> str:
    """
    Wrapper function to generate a fuzzy hash for a basic block

    :param bb: the basic block to hash
    :param h_planes: a numpy array of hyperplanes (if none are provided the default values will be used)
    :return: a string representing a fuzzy hash of the basic block
    """
    if not bb.is_il:
        index = bb.index
        bb = bb.function.mlil.basic_blocks[index]

    if h_planes is None:
        h_planes = gen_planes()
    return bucket(vectorize(bb), h_planes)


def bucket(vector: np.ndarray, h_planes: np.ndarray) -> str:
    """
    Encode a vector's position relative to each hyper plane such that similar vectors will land in the same "bucket".

    :param vector: the vector to encode the position of
    :param h_planes: a numpy array of hyperplanes
    :return: a hex string representing the "bucket" the given vector lands in
    """
    bools = [str(int(np.dot(vector, h_plane) > 0)) for h_plane in h_planes]
    return hex(int("".join(bools), 2))[2:]


def gen_planes(num_planes: int = 100) -> np.ndarray:
    """
    Generates a series of arbitrary hyper planes to be used for "bucketing" in our LSH.

    :param num_planes: the number of hyper planes to generate
    :return: a numpy array containing num_planes pythonic arrays
    """
    # TODO: look into alternate (deterministic) methods of generating uniformly distributed hyperplanes
    h_planes = np.array(
        [
            [
                (-1 ^ int((i * j) / 2)) * (i ^ j) % (int(i / (j + 1)) + 1)
                for j in range(len(binja.MediumLevelILOperation.__members__))
            ]
            for i in range(num_planes)
        ]
    )

    return h_planes


def vectorize(bb: BasicBlock) -> np.ndarray:
    """
    Converts a basic block into a vector representing the number of occurrences of each "type" of MLIL instruction,
    effectively forming a "bag of words".

    :param bb: the basic block to construct a vector representation of
    :return: a numpy array representing the number of occurrences of each MLIL instruction type
    """
    vector = dict(
        (key, 0) for key in range(len(binja.MediumLevelILOperation.__members__))
    )
    for instr in bb:
        vector[instr.operation.value] += 1
    return np.fromiter(vector.values(), dtype=int)


def brittle_hash(bv: BinaryView, bb: BasicBlock) -> str:
    # operands are only available on an IL, ensure we're working with one
    if bb is None or not bb.is_il:
        return

    disassembly_text = "".join([str(instr.operation) for instr in bb])

    # substitute out names tainted by addresses/etc.
    function_pattern = "sub_[0-9, a-z]{6}"
    data_pattern = "data_[0-9, a-z]{6}"
    label_pattern = "label_[0-9, a-z]{6}"

    re.sub(function_pattern, "function", disassembly_text)
    re.sub(data_pattern, "data", disassembly_text)
    re.sub(label_pattern, "label", disassembly_text)

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
    anchor_text = "".join(anchors)
    disassembly_text += anchor_text
    m = hashlib.sha256()
    m.update(disassembly_text.encode("utf-8"))
    return m.digest().hex()


def main():
    import argparse
    import pprint as pp

    parser = argparse.ArgumentParser()
    # parse binary file path
    parser.add_argument(
        "-b", "--binary", type=str, required=True, help="Path to binary file(s)"
    )
    parser.add_argument(
        "-f",
        "--function",
        type=str,
        required=False,
        help="Optional name or address of function to hash.",
    )
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache results to disk. Cannot be used when specifying a function.",
    )
    parser.add_argument(
        "--load", action="store_true", help="Load cached results from disk."
    )
    args = parser.parse_args()

    bins = get_binaries(args.binary)
    if len(bins) == 0:
        raise ValueError(f"No binaries found at path: {args.binary}")
    if len(bins) > 1 and args.function is not None:
        raise ValueError("Cannot specify a function when hashing multiple binaries.")
    for b in bins:
        logger.info(f"Hashing {b}")
        bv = open_view(b)
        if args.load:
            try:
                load_hash(args.binary, bv=bv)
            except (ValueError, FileNotFoundError) as e:
                logger.error(
                    f"Error loading cached results: {e}.\nPlease use --cache to generate a cache file."
                )
        if args.function is not None:
            if args.cache:
                raise ValueError(
                    "Must hash full binary to cache results, rerun without --cache or --function."
                )
            if args.function.isdigit():
                function = bv.get_function_at(int(args.function))
            elif args.function.startswith("0x"):
                function = bv.get_function_at(int(args.function, 16))
            else:
                function = bv.get_functions_by_name(args.function)
                if len(function) > 1:
                    raise ValueError(
                        "Multiple functions with the same name found, please specify an address"
                    )
                function = function[0]
            print(
                f"Features for {function}:\n{pp.pformat(features_to_dict(hash_function(function)), sort_dicts=False)}"
            )
            continue
        signature, features = hash_all(
            bv,
            return_serializable=True,
            show_progress=args.progress,
            save_to_file=args.cache,
        )
        print(f"Signature for {b}:\n{signature}")
    return


if __name__ == "__main__":
    main()
