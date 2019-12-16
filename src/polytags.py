#!/usr/bin/env python3

# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import binaryninja as binja

import json
import os
import sys
from typing import Dict, List


def find_leaves(tag_tree: Dict, functions: List[str] = None) -> Dict[str, str]:
    """
    Recursively visits all subEls in polymerge output, finding the bottom tags in the hierarchy

    :param tag_tree: nested subEls dictionary
    :param functions: only top level subEls dictionary has a list of associated functions - pass this list to children
    :return: a mapping of function names to tag names
    """
    if functions is None:
        functions = []

    sub_elements = tag_tree['subEls']

    # not all subEls have functions?
    if 'functions' in tag_tree.keys():
        functions = tag_tree['functions']

    # if subEls is empty, we must have hit a leaf
    if len(sub_elements) == 0:
        return dict([(function, tag_tree['name']) for function in functions])

    else:
        tags = {}
        for sub_tag in sub_elements:
            # TODO: check for multiple tags applied to a function
            tags.update(find_leaves(sub_tag, functions))
        return tags


def read_polytags(file: str) -> Dict[str, str]:
    """
    Driver function to recover function to tag mapping from the JSON file produced by polymerge

    :param file: path to polymerge output file
    :return: a mapping of function names to tag names
    """
    if os.path.exists(file):
        with open(file, 'r') as input_file:
            poly_data = json.load(input_file)
            # TODO: compare hashes as sanity check
            tag_dict = {}
            for file in poly_data['struc']:
                for element in file['subEls']:
                    leaves = find_leaves(element)
                    if leaves is not None:
                        tag_dict.update(leaves)
            return tag_dict
    else:
        print('Invalid signature file.')
        sys.exit(-1)


def open_binary_in_triage(binary_path: str):
    print("Starting to load binary {} and set triage settings".format(binary_path))
    view = binja.BinaryView.open(binary_path)
    if view is None:
        print("Failed to load binary.")
    bns = binja.settings.Settings()
    print("Starting to load and analyze binary {}...".format(binary_path))
    bv = None
    for available in view.available_view_types:
        if available.name != 'Raw':
            bv = available.open(binary_path)
            break
    bns.set_string('analysis.mode', 'controlFlow', bv)
    bns.set_integer('analysis.limits.maxFunctionAnalysisTime', 2, bv)
    bns.set_integer('analysis.maxFunctionAnalysisTime', 2, bv)
    bns.set_integer('analysis.maxFunctionSize', 500, bv)
    bns.set_integer('analysis.maxFunctionUpdateCount', 2, bv)
    bns.set_integer('analysis.limits.cacheSize', 64, view)
    bns.set_bool("analysis.linearSweep.controlFlowGraph", True, bv)
    bns.set_bool("analysis.linearSweep.autorun", True, bv)
    bv.update_analysis_and_wait()
    print("Loaded binary {} into Binary Ninja: {}.".format(binary_path, bv))
    return bv


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Apply signatures that were captured from a signature file to a binary.')
    parser.add_argument('binary',
                        help='The executable file to attempt to apply signatures onto.')
    parser.add_argument("--polymerge", required=True,
                        help="Reads JSON emitted by polymerge tool")
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print("Must provide valid path to binary.")
        sys.exit(-1)
    if not os.path.isfile(args.polymerge):
        print("Must provide valid path to polymerge JSON file.")
        sys.exit(-2)

    #bv = open_binary_in_triage(args.binary)
    functions = read_polytags(args.polymerge)
    for function, tag in functions.items():
        print("{: <40}\t{}".format(function.replace('dfs$',''), tag))
