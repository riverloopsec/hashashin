#!/usr/bin/env python3

# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import sys
import os
import argparse

import binaryninja as binja
from lsh import hash_all
from parsing import write_json
from tagging import read_tags


def generate(bndb: str, sig_path: str):
    """
    Create signatures for all functions which have tagged basic blocks within the BNDB given.

    :param bndb: path to BNDB input file that is tagged
    :param sig_path: path to output JSON signature file to
    :return: number of signatures generated
    """
    bv = binja.BinaryViewType.get_view_of_file(bndb)
    print("Loaded BNDB {} into Binary Ninja.".format(bndb))
    hashes = hash_all(bv)
    print("{} functions in binary have been hashed.".format(len(hashes)))

    signatures = read_tags(bv, hashes)
    print("{} signatures have been created based on the present tags.".format(len(signatures)))
    write_json(signatures, sig_path)
    print("Signature file written to {}.".format(sig_path))
    return len(signatures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture tags from a Binary Ninja database and save them along with anchoring signatures to a signature_file.')
    parser.add_argument('bndb', type=str,
                        help='The Binary Ninja database to learn tags from.')
    parser.add_argument('signature_file', type=str,
                        help='The JSON signature file to output.')
    args = parser.parse_args()

    if not os.path.isfile(args.bndb):
        print("Must provide valid path to Binary Ninja database.")
        sys.exit(-1)
    if os.path.exists(args.signature_file):
        print("Must provide a signature_file path that does not exist.")
        sys.exit(-2)

    generate(args.bndb, args.signature_file)

