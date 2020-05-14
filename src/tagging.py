# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import binaryninja as binja
#from annotations import Annotations
from . import annotations
from typing import Dict

# type aliases
Function = binja.function.Function
Basic_Block = binja.basicblock.BasicBlock
Binary_View = binja.binaryview.BinaryView


def tag_function(bv: Binary_View, function: Function, sig: str,  signatures: Dict[str, annotations.Annotations]) -> None:
    """
    Port tags for each basic block from signatures dictionary into current binary.

    :param bv: BinaryView that tags will be applied to
    :param function: function to tag
    :param sig: sig of function as generated by hashing.hash_function()
    :param signatures: dictionary mapping function hashes to Annotation objects
    """
    tag_types = {}

    annotations = signatures[sig]
    for bb_index in annotations.blocks():
        bb = function.basic_blocks[int(bb_index)]

        for tag_name in signatures[sig][bb_index]:
            if tag_name == '':
                continue

            if tag_name not in tag_types:
                tag_types[tag_name] = bv.create_tag_type(
                    tag_name, tag_name[0].capitalize())

            tag_data = signatures[sig][bb_index][tag_name]
            tag = function.create_tag(tag_types[tag_name], tag_data)
            function.add_user_address_tag(bb.start, tag)


def read_tags(bv: Binary_View, hashes: Dict[str, Function]) -> Dict[str, annotations.Annotations]:
    """
    Gathers tag locations from every function in the binary.

    :param bv: BinaryView that contains the analysis results
    :param hashes: a dictionary mapping hashes to their functions
    :return: dictionary representing all tags in the current binary
    """
    tagged_dict = {}

    # TODO: switch to use GetAllTagReferences once it's available in the python API for O(1) access times
    for hash_value in hashes:
        function = hashes[hash_value]
        tagged_dict[hash_value] = annotations.Annotations(function=function, bv=bv)
    return tagged_dict
