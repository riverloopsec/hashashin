#!/usr/bin/env python3

# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import binaryninja as binja

import argparse
import os
import sys
from typing import Dict, Tuple

from utils.lsh import brittle_hash, hash_function

Binary_View = binja.binaryview.BinaryView


def diff(dst_path: str, signatures: Dict[str, Tuple[str]]) -> None:
    dst_bv = binja.BinaryViewType.get_view_of_file(dst_path)

    mismatched_tt = dst_bv.create_tag_type('Difference', '🚫')
    new_function_tt = dst_bv.create_tag_type('New function', '➕')

    # align functions for diffing
    for function in dst_bv.functions:
        function_hash = hash_function(function)
        if function_hash in signatures:
            print('aligned {}...'.format(function.name))
            for bb in function.hlil.basic_blocks:
                bb_hash = brittle_hash(dst_bv, bb)
                # basic block matches a block in the source
                if bb_hash in signatures[function_hash]:
                    bb.set_user_highlight(binja.highlight.HighlightStandardColor.GreenHighlightColor)
                # basic block differs, but function is similar
                else:
                    print('tagging mismatch...')
                    tag = function.create_tag(mismatched_tt, '')
                    function.add_user_address_tag(bb.start, tag)
                    bb.set_user_highlight(binja.highlight.HighlightStandardColor.RedHighlightColor)

        # if function can't be aligned for comparison, assume all basic blocks are unique
        else:
            tag = function.create_tag(new_function_tt, 'New function')
            function.add_user_address_tag(function.start, tag)

            for bb in function:
                bb.set_user_highlight(binja.highlight.HighlightStandardColor.RedHighlightColor)

    output_bndb = os.path.join(os.getcwd(), dst_path + '.bndb')
    print("Writing output Binary Ninja database at {}".format(output_bndb))
    if not os.path.isfile(output_bndb):
        print('writing view to database...')
        dst_bv.create_database(output_bndb)
    else:
        print('updating database...')
        dst_bv.save(output_bndb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify differences between two binaries')
    parser.add_argument('src', type=str,
                        help='Source binary (or bndb)')
    parser.add_argument('dst', type=str,
                        help='Destination binary (or bndb)')
    args = parser.parse_args()

    if not (os.path.isfile(args.src) and os.path.isfile(args.dst)):
        print("Must provide valid path to binary or bndb")
        sys.exit(-1)

    # compute function and basic block hashes for the source binary
    bv = binja.BinaryViewType.get_view_of_file(args.src)
    signatures = {}
    for function in bv.functions:
        function_hash = hash_function(function)
        bb_hashes = []
        for bb in function.hlil.basic_blocks:
            bb_hashes.append(brittle_hash(bv, bb))

        signatures[function_hash] = bb_hashes

    print('{} ingested...'.format(args.src))

    print('Starting diffing...')
    diff(args.dst, signatures)
