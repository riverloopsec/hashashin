import binaryninja as binja
from json import JSONEncoder, JSONDecoder
from typing import Dict

# type aliases
Function = binja.function.Function
Basic_Block = binja.basicblock.BasicBlock
Binary_View = binja.binaryview.BinaryView
Tag_Type = binja.binaryview.TagType


class Annotations:
    """
    Helper class to organize tags
    """

    def __init__(self, function: Function = None, bv: Binary_View = None, raw_data: Dict[str, Dict[str, str]] = None):
        self.tagged_dict = {}
        if (function is not None) and (bv is not None):
            function_tags = function.address_tags

            # to preserve annotation granularity, we store tags by their basic block # inside the function
            for _, addr, label in function_tags:
                bb = bv.get_basic_blocks_at(addr)[0]

                # initialize tag mapping if it does not already exist
                if bb.index not in self.tagged_dict.keys():
                    self.tagged_dict[bb.index] = {}
                self.tagged_dict[bb.index][label.type.name] = label.data

        elif raw_data is not None:
            self.decode(raw_data)

        else:
            self.tagged_dict = {}

    def __getitem__(self, item):
        return self.tagged_dict[str(item)]

    def encode(self) -> Dict[str, Dict[str, str]]:
        """
        Converts Annotations object into a JSON encodable dictionary

        :return: dictionary representing all annotations contained in object
        """
        return self.tagged_dict

    def decode(self, raw_data: Dict[str, Dict[str, str]]) -> None:
        """
        Constructs Annotations object from nested dictionary

        :param raw_data:
        """
        for bb_num in raw_data:
            bb_tags = {}
            for tag_name in raw_data[bb_num]:
                bb_tags[tag_name] = raw_data[bb_num][tag_name]

            self.tagged_dict[bb_num] = bb_tags

    def blocks(self):
        """
        Helper function to access all basic blocks

        :return: List of all basic blocks contained
        """
        return list(self.tagged_dict.keys())
