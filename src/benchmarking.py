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
from apply_signatures import apply

Binary_View = binja.binaryview.BinaryView


def get_tag_count(bv: Binary_View) -> Dict[Tuple[str, binja.TagType], int]:
    """
    Counts all occurrences of each tag type

    :param bv: binaryview of file to analyze
    :return: dict of the form {(tag_name, TagType): count}
    """
    tags = dict((tag_name, 0) for tag_name in bv.tag_types)
    for function in bv.functions:
        for tag in function.address_tags:
            tag_name = tag[2].type.name
            tags[tag_name] += 1
    return tags


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
        gen_start_time = time.time()
        num_sigs = generate(args.bndb, tmp_path)
        gen_end_time = time.time()
        print('Signature generation done after {}s, generated {} function signatures'.format(gen_end_time - gen_start_time, num_sigs))
    else:
        print('Using signature file from {}'.format(tmp_path))

    print('Starting signature application...')
    apply_start_time = time.time()
    num_func_sigs_applied, output_bndb = apply(args.binary, tmp_path)
    apply_end_time = time.time()
    print('Signature application done after {}s, applied {} function signatures'.format(apply_end_time - apply_start_time, num_func_sigs_applied))

    # Get total tags in source:
    src_bv = binja.BinaryViewType.get_view_of_file(args.bndb)
    src_tags = get_tag_count(src_bv)

    # Get total tags in destination:
    dst_bv = binja.BinaryViewType.get_view_of_file(output_bndb)
    dst_tags = get_tag_count(dst_bv)

    for tag in src_tags.keys():
        if tag in dst_tags.keys():
            print('\t{}: {}/{}'.format(tag, dst_tags[tag], src_tags[tag]))

    total_src_tags = 0
    total_dst_tags = 0
    for tag in src_tags:
        if tag in dst_tags.keys():
            total_src_tags += src_tags[tag]
    for tag in dst_tags:
        if tag in src_tags.keys():
            total_dst_tags += dst_tags[tag]

    print('Total: {}%'.format((total_dst_tags / float(total_src_tags)) * 100))
