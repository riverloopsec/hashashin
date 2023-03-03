#!/bin/env python3

from binaryninja import PluginCommand  # type: ignore
from hashashin.classes import BinjaFeatureExtractor
import pprint as pp


def get_features(view, func):
    ff = BinjaFeatureExtractor().extract(func)
    func.comment = pp.pformat(ff.get_feature_dict(), sort_dicts=False)
    print(f"Signature for {ff.function.name}:\n{ff.signature}")


def get_signature(view):
    bs = BinjaFeatureExtractor().extract_from_bv(view)
    print(f"Signature for {view.file.filename}:\n{bs.signature}")


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
