#!/bin/env python3

from binaryninja import PluginCommand
from hashashin.lsh import hash_function, hash_all
from hashashin.utils import func2str, features_to_dict, vec2hex, dict_to_features
import pprint as pp
import ast


def get_features(view, func):
    if func.start in func.comments:
        try:
            print(func.comments[func.start])
            func_hash = dict_to_features(ast.literal_eval(func.comments[func.start]))
            print(f"Cached Hash for {func2str(func)}:\n{vec2hex(func_hash)}")
            return
        except Exception as e:
            print(f"Error parsing comment: {e}")
    func_hash = hash_function(func)
    func.set_comment_at(func.start, pp.pformat(features_to_dict(func_hash), sort_dicts=False))
    print(f"Hash for {func2str(func)}:\n{vec2hex(func_hash)}")


def get_signature(view):
    sig, _ = hash_all(view)
    print(f"Signature for {view.file.filename}:\n{sig}")


PluginCommand.register_for_function("Hashashin Feature Extraction",
                                    "Run hashashin's feature extraction for a given function.",
                                    get_features)
PluginCommand.register("Hashashin Signature Generation",
                       "Run hashashin's signature generation for a given binary view.",
                       get_signature)
