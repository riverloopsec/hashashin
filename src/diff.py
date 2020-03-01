#!/usr/bin/env python3

# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import binaryninja as binja

import argparse
import os
import sys
import time
from typing import Dict, Tuple

from generate_signatures import generate
from parsing import read_json
from lsh import brittle_hash, hash_function

Binary_View = binja.binaryview.BinaryView


def diff(bndb_path, binary_path, sig_path) -> None:
    bndb_bv = binja.BinaryViewType.get_view_of_file(bndb_path)
    binary_bv = binja.BinaryViewType.get_view_of_file(binary_path)
    signatures = read_json(sig_path)

    # align functions for diffing
    for function in bndb_bv.functions:
        function_hash = hash_function(function)
        if function_hash in signatures:
            for bb in function.basic_blocks:
                bb_hash = brittle_hash(bndb_bv, bb)
                # basic block matches a block in the source
                if bb_hash in signatures[function_hash]:
                    bb.set_user_highlight(binja.highlight.HighlightStandardColor.GreenHighlightColor)
                # basic block differs, but function is similar
                else:
                    bb.set_user_highlight(binja.highlight.HighlightStandardColor.RedHighlightColor)
        # if function can't be aligned for comparison, assume all basic blocks are unique
        else:
            for bb in function:
                bb.set_user_highlight(binja.highlight.HighlightStandardColor.RedHighlightColor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark accuracy and performance on a pair of binaries.')
    parser.add_argument('bndb', type=str,
                        help='The Binary Ninja database to learn tags from.')
    parser.add_argument('binary', type=str,
                        help='The binary to compare with.')
    parser.add_argument('--reuse-sig', action='store_true',
                        help='Typically for development/debugging, reuse a signature file that was created.')
    args = parser.parse_args()

    if not os.path.isfile(args.bndb):
        print("Must provide valid path to Binary Ninja database.")
        sys.exit(-1)

    if not os.path.isfile(args.binary):
        print("Must provide valid path to binary.")
        sys.exit(-2)

    # Create a new path name for saving the signatures to:
    tmp_path = os.path.join(os.path.dirname(args.bndb), 'benchmarking_{}.sig'.format(os.path.basename(args.bndb)))
    if args.reuse_sig and not os.path.isfile(tmp_path):
        print("If --reuse-sig is set, the signature file {} must exist.".format(tmp_path))
        sys.exit(-3)
    if not args.reuse_sig and os.path.exists(tmp_path):
        print("Cannot write signature file to {} as it already exists.".format(tmp_path))
        sys.exit(-3)
    if not args.reuse_sig:
        print("Writing signature file to {}".format(tmp_path))

        print('Starting signature generation...')
        num_sigs = generate(args.bndb, tmp_path)
        print('Signatures generated for {}...'.format(args.bndb))
    else:
        print('Using signature file from {}'.format(tmp_path))

    print('Starting diffing...')
    diff(args.bndb, args.binary, tmp_path)
