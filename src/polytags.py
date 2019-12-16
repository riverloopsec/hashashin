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
