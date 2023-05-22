#!/bin/env python3

from binaryninja import PluginCommand  # type: ignore
from binaryninja.plugin import BackgroundTaskThread  # type: ignore
from hashashin.classes import BinjaFeatureExtractor
import pprint as pp


def apply_comment(func, function_features):
    func.comment = pp.pformat(function_features.get_feature_dict(), sort_dicts=False)


def get_features(view, func):
    apply_comment(func, ff := BinjaFeatureExtractor().extract(func))
    print(f"Signature for {ff.function.name}:\n{ff.signature}")


# class BackgroundSignatureGenerator(BackgroundTaskThread):
#     def __init__(self, view):
#         super().__init__(f"Computing {len(view.functions)} function features...")
#         self.view = view
#
#     def _helper(self, func):
#         apply_comment(func, ff := BinjaFeatureExtractor().extract(func))
#         self.progress
#
#     def run(self):
#         for func in self.view.functions:



def get_signature(view):
    task = BackgroundTaskThread(f"Computing {len(view.functions)} function features...")
    def extract_signature():
        view.session_data["BinarySignature"] = (bs := BinjaFeatureExtractor().extract_from_bv(view))
        for ff in bs.functionFeatureList:
            apply_comment(ff.function.function, ff)
        print(f"Signature for {view.file.filename}:\n{bs.signature}")
    task.run(extract_signature)


PluginCommand.register_for_function(
    "Hashashin Feature Extraction",
    "Run hashashin's feature extraction for a given function.",
    get_features,
)
PluginCommand.register(
    "Hashashin Signature Generation",
    "Run hashashin's signature generation for a given binary view.",
    get_signature,
)
