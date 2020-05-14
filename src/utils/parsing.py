# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import json
from .annotations import Annotations
import os
import sys
from typing import Dict


def write_json(data: Dict[str, Annotations], file: str):
    """
    Dumps generated signatures to disk for comparision with other binaries

    :param data: dictionary mapping hashes to dictionary mapping basic block index to tag data
    :param file: full path to json file to be dumped
    """
    if not os.path.exists(file):
        with open(file, 'w', encoding='utf-8') as output_file:
            json_data = dict(map(lambda kv: (kv[0], kv[1].encode()), data.items()))
            json.dump(json_data, output_file, ensure_ascii=False, indent=4)
    else:
        print('Signature file `{}` already exists.'.format(file))
        sys.exit(-1)


def read_json(file: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Reads previously generated signature file into dictionary for analysis

    :param file: path to json file
    :return: dictionary representation of the contents of file specified
    """
    if os.path.exists(file):
        with open(file, 'r') as input_file:
            return json.load(input_file)
    else:
        print('Invalid signature file.')
        sys.exit(-1)
