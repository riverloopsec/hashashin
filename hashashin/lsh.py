raise NotImplementedError()

import logging
import os
import re
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import TYPE_CHECKING

import binaryninja as binja  # type: ignore
import numpy as np
from binaryninja import BasicBlock
from binaryninja import BinaryView
from binaryninja import Function
from binaryninja import enums
from binaryninja import open_view
from tqdm import tqdm  # type: ignore

from hashashin.utils import func2str
from hashashin.utils import serialize_features
from hashashin.utils import get_binaries
from hashashin.feature_extractors import extract_features
from hashashin.feature_extractors import get_strings

if TYPE_CHECKING:
    from hashashin.classes import BinSig, FuncSig

logging.basicConfig()
logger = logging.getLogger(os.path.basename(__name__))
SIGNATURE_LEN = 20


def hash_tagged(bv: BinaryView) -> Dict[str, Function]:
    """
    Iterate over tagged functions in the binary and calculate their hash.

    :param bv: binary view encapsulating the binary
    :return: a dictionary mapping hashes to functions
    """
    raise NotImplementedError()


def hash_basic_block(bb: BasicBlock, h_planes: Optional[np.ndarray] = None) -> str:
    """
    Wrapper function to generate a fuzzy hash for a basic block

    :param bb: the basic block to hash
    :param h_planes: a numpy array of hyperplanes (if none are provided the default values will be used)
    :return: a string representing a fuzzy hash of the basic block
    """
    raise NotImplementedError()


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
    print(
        "Hashing functions... (if you are seeing this, you should try showing progress with --progress)",
        end="\r",
    )
    import shutil
    ts = shutil.get_terminal_size(fallback=(120, 50)).columns // 2
    for function in (pbar := tqdm(bv.functions, disable=not show_progress)):
        pbar.set_description(f"Hashing {func2str(function):{ts}.{ts}}")
        feature = hash_function(function)
        features[function] = feature
        # if vec2hex(features[function]) != vec2hex(baseline_feats[function]):
        #     print("ok this is the problem")
    signature = min_hash(np.stack(features.values()))
    if return_serializable:
        features = serialize_features(features)
    if save_to_file:
        write_hash(bv.file.filename, signature, features)
    return signature, features


def hash_function(
    function: Function,
    h_planes: Optional[np.ndarray] = None,
    string_map=None,
) -> "FuncSig":
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
    features = extract_features(function)
    # if function.start == 141836:
    #     print('hey now')
    # if function.start == 141836 and vec2hex(features) != '':
    #     print('wtf')
    return features


def get_func_from_unknown(bv: BinaryView, func: Union[str, int]) -> Function:
    if func.isdigit():
        return bv.get_function_at(int(func))
    elif func.startswith("0x"):
        return bv.get_function_at(int(func, 16))
    else:
        function = bv.get_functions_by_name(func)
        if len(function) > 1:
            raise ValueError(
                "Multiple functions with the same name found, please specify an address"
            )
        return function[0]


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
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--load", action="store_true", help="Load cached results from disk."
    )
    load_group.add_argument(
        "--force-load",
        action="store_true",
        help="Used to force hash generation if --load fails.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress output to stdout."
    )
    args = parser.parse_args()
    if args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    if args.function and os.path.isdir(args.binary):
        logger.error(
            "Cannot specify a function when hashing multiple binaries. Exiting."
        )
        return
    bins = get_binaries(os.path.expanduser(args.binary), progress=args.progress)
    if len(bins) == 0:
        raise ValueError(f"No binaries found at path: {args.binary}")
    if len(bins) > 1:
        if args.function is not None:
            raise ValueError(
                "Cannot specify a function when hashing multiple binaries."
            )
        logger.info(f"Hashing {len(bins)} binaries...")
    for b in bins:
        b = os.path.relpath(b)
        logger.info(f"Hashing {b}")
        if args.load or args.force_load:
            try:
                signature, features = load_hash(
                    b,
                    generate=args.force_load,
                    regenerate=False,
                    progress=args.progress,
                    only_signature=args.function is None,
                )
                logger.info(f"Cached signature for {b}:\n{signature}")
                if args.function:
                    try:
                        func = next(f for f in features if args.function in f)
                    except StopIteration:
                        func = func2str(get_func_from_unknown(open_view(b), args.function))
                    logger.info(f"Hash for {args.function}:\n{pp.pformat(features_to_dict(features[func]), sort_dicts=False)}")
                continue
            except (ValueError, FileNotFoundError) as e:
                logger.error(
                    f"Error loading cached results: {e}.\nPlease use --cache to generate a cache file."
                )
                return
        bv = open_view(b)
        if args.function is not None:
            if args.cache:
                raise ValueError(
                    "Must hash full binary to cache results, rerun without --cache or --function."
                )
            function = get_func_from_unknown(bv, args.function)
            logger.info(
                f"Features for {function}:\n{pp.pformat(features_to_dict(hash_function(function)), sort_dicts=False)}"
            )
            continue
        signature, features = hash_all(
            bv,
            return_serializable=True,
            show_progress=args.progress,
            save_to_file=args.cache,
        )
        logger.info(f"Signature for {b}:\n{signature}")
    return


if __name__ == "__main__":
    main()
