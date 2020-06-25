#!/usr/bin/env python3

# Copyright 2019 River Loop Security LLC, All Rights Reserved
# Author Rylan O'Connell

import binaryninja as binja

import argparse
import os
import sys
import math
import networkx as nx
from typing import Dict, Tuple, List

from utils.lsh import brittle_hash, hash_function

Binary_View = binja.binaryview.BinaryView


def diff(dst_path: str, pairings: List[nx.DiGraph]) -> None:
    dst_bv = binja.BinaryViewType.get_view_of_file(dst_path)

    mismatched_tt = dst_bv.create_tag_type('Difference', '🚫')
    new_function_tt = dst_bv.create_tag_type('New function', '➕')

    # align functions for diffing
    # TODO: exclude thunks/etc.
    for function in dst_bv.functions:
        # ignore small functions to avoid false positives
        if len(function.basic_blocks) < 5:
            continue

        hash_cfg = function_graph(dst_bv, function.hlil)
        min_pairing, distance = get_min_pair(hash_cfg, pairings)

        # if pairing failed, the function must be new to this binary
        if min_pairing is None:
            print('No suitable function pairing for {}'.format(function.name))
            tag = function.create_tag(new_function_tt, 'New function')
            function.add_user_address_tag(function.start, tag)

            for bb in function.hlil:
                for instr in bb:
                    function.set_user_instr_highlight(
                        instr.address,
                        binja.highlight.HighlightStandardColor.RedHighlightColor
                    )
            continue

        if distance > 0:
            print('Successfully aligned {} to {} (delta: {})'.format(function.name, min_pairing.name, distance))

        for bb in function.hlil.basic_blocks:
            # TODO: optmize to avoid second hashing
            bb_hash = brittle_hash(dst_bv, bb)

            # basic block matches a block in the source
            if min_pairing.has_node(bb_hash):
                for instr in bb:
                    function.set_user_instr_highlight(
                        instr.address,
                        binja.highlight.HighlightStandardColor.GreenHighlightColor
                    )

            # basic block differs, but function is similar
            else:
                print('tagging mismatch at {}...'.format(hex(bb.start + function.start)))
                tag = function.create_tag(mismatched_tt, '')
                function.add_user_address_tag(bb.start + function.start, tag)
                for instr in bb:
                    function.set_user_instr_highlight(
                        instr.address,
                        binja.highlight.HighlightStandardColor.RedHighlightColor
                    )

    output_bndb = os.path.join(os.getcwd(), dst_path + '.bndb')
    print("Writing output Binary Ninja database at {}".format(output_bndb))
    if not os.path.isfile(output_bndb):
        print('writing view to database...')
        dst_bv.create_database(output_bndb)
    else:
        print('updating database...')
        dst_bv.save(output_bndb)


def get_min_pair(function: nx.DiGraph, pairings: List[nx.DiGraph]) -> Tuple[nx.DiGraph, float]:
    min_distance = math.inf
    min_pairing = None

    for pairing in pairings:
        distance = function_difference(function, pairing)
        # only accept pairings "close" to the original (accounting for function size)
        if (distance < min_distance) and \
                (distance < 0.40 * (function.number_of_nodes() + .1 * function.number_of_edges())):
            min_distance = distance
            min_pairing = pairing

    return min_pairing, min_distance


def function_difference(f1: nx.DiGraph, f2: nx.DiGraph) -> float:
    distance = 0.0

    for block in f1.nodes:
        if not f2.has_node(block):
            distance += 1
    for block in f2.nodes:
        if not f1.has_node(block):
            distance += 1

    for edge in f1.edges:
        if not f2.has_edge(edge[0], edge[1]):
            distance += 0.1
    for edge in f2.edges:
        if not f1.has_edge(edge[0], edge[1]):
            distance += 0.1

    return distance


def function_graph(bv: binja.binaryview.BinaryView, function: binja.highlevelil.HighLevelILFunction) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.name = function.source_function.name

    bbs = {}
    for bb in function:
        bb_hash = brittle_hash(bv, bb)
        graph.add_node(bb_hash)
        bbs[bb] = bb_hash

    for bb in function:
        bb_hash = bbs[bb]
        outgoing = bb.outgoing_edges
        for edge in outgoing:
            target_hash = bbs[edge.target]
            graph.add_edge(bb_hash, target_hash)

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Identify differences between two binaries')
    parser.add_argument('src', type=str,
                        help='Source binary (or bndb)')
    parser.add_argument('dst', type=str,
                        help='Destination binary (or bndb)')
    args = parser.parse_args()

    if not (os.path.isfile(args.src) and os.path.isfile(args.dst)):
        print("Must provide valid path to binary or bndb")
        sys.exit(-1)

    # compute function and basic block hashes for the source binary
    bv = binja.BinaryViewType.get_view_of_file(args.src)
    functions = []

    print('Ingesting {}...'.format(args.src))
    # TODO: exclude thunks/etc.
    for function in bv.functions:
        # ignore small functions to avoid false positives
        if len(function.basic_blocks) < 5:
            continue

        hash_cfg = function_graph(bv, function.hlil)
        functions.append(hash_cfg)

    print('Starting diffing...')
    diff(args.dst, functions)
