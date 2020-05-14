#!/usr/bin/env python3

from . import lsh
from . import parsing
from . import tagging

from binaryninja.plugin import PluginCommand
from binaryninja.plugin import BackgroundTaskThread
from binaryninja.interaction import get_open_filename_input, get_save_filename_input


class GenFunctionSigInBackground(BackgroundTaskThread):

    def __init__(self, function, bv, msg):
        BackgroundTaskThread.__init__(self, msg, True)
        self.function = function
        self.bv = bv

    def run(self):
        hash = lsh.hash_function(self.function)
        print("Hash of {} is {}".format(hex(self.function.lowest_address), hash))

        signatures = tagging.read_tags(self.bv, {hash: self.function})
        sig_path = get_save_filename_input("Signature File")
        parsing.write_json(signatures, sig_path)

def hash_in_background(bv, function):
    hash_task = GenFunctionSigInBackground(function, bv, "Hashing function at {}...".format(hex(function.lowest_address)))
    hash_task.start()


PluginCommand.register_for_function("Generate Signatures", "Produce fuzzy hash of current function",
                                    hash_in_background)
