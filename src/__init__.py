#!/usr/bin/env python3

from .utils import lsh, parsing, tagging, Annotations, tag_function

from binaryninja.plugin import PluginCommand
from binaryninja.plugin import BackgroundTaskThread
from binaryninja.interaction import get_save_filename_input, get_open_filename_input


class GenFunctionSigInBackground(BackgroundTaskThread):

    def __init__(self, function, bv, msg):
        BackgroundTaskThread.__init__(self, msg, True)
        self.function = function
        self.bv = bv

    def run(self):
        function_hash = lsh.hash_function(self.function)
        print("Hash of {} is {}".format(hex(self.function.lowest_address), function_hash))

        signatures = tagging.read_tags(self.bv, {function_hash: self.function})
        sig_path = get_save_filename_input("Signature File")
        parsing.write_json(signatures, sig_path)


class ApplyFunctionSigInBackground(BackgroundTaskThread):

    def __init__(self, function, bv, msg):
        BackgroundTaskThread.__init__(self, msg, True)
        self.function = function
        self.bv = bv

    def run(self):
        function_hash = lsh.hash_function(self.function)
        print("Hash of {} is {}".format(hex(self.function.lowest_address), function_hash))

        sig_path = get_open_filename_input("Signature File")
        data = parsing.read_json(sig_path)
        signatures = {}
        for raw_hash in data:
            # only bother with functions that actually have tags
            if len(data[raw_hash]) > 0:
                signatures[raw_hash] = Annotations(raw_data=data[raw_hash])

        print("Signature file {} loaded into memory.".format(sig_path))

        if function_hash in signatures:
            tag_function(self.bv, self.function, function_hash, signatures)
            print('Located a match at {}!'.format(function_hash))



def gen_sigs_in_background(bv, function):
    gen_sigs_task = GenFunctionSigInBackground(function, bv, "Generating signatures for function at {}...".format(
        hex(function.lowest_address)))
    gen_sigs_task.start()


def apply_sigs_in_background(bv, function):
    apply_sigs_task = ApplyFunctionSigInBackground(function, bv, "Applying signatures to function at {}...".format(
        hex(function.lowest_address)))
    apply_sigs_task.start()


PluginCommand.register_for_function("Generate Signatures", "Generate signature for the current function",
                                    gen_sigs_in_background)

PluginCommand.register_for_function("Apply Signatures", "Apply signatures from file to current function",
                                    apply_sigs_in_background)
