#!/usr/bin/env python

from lsh import hash_function

from binaryninja.plugin import PluginCommand
from binaryninja.plugin import BackgroundTaskThread


class HashFunctionInBackground(BackgroundTaskThread):
    def __init__(self, function, msg):
        BackgroundTaskThread.__init__(self, msg, True)
        self.function = function

    def run(self):
        print(hash_function(self.function))


def hash_in_background(function):
    background_task = HashFunctionInBackground(function, "Hashing function")
    background_task.start()


PluginCommand.register_for_function("Hash", "Produce fuzzy hash of current function",
                                    hash_in_background)
