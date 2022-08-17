import unittest

import hashashin
import binaryninja
import os
import cProfile

from ground_truths import *
from hashashin.lsh import f2str

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_DIR = os.path.join(TEST_DIR, 'binary_data')


def hash_things():
    bv = binaryninja.open_view(f'{BINARY_DIR}/echo')
    print(hashashin.hash_all(bv))


class TestCases(unittest.TestCase):
    def setUp(self):
        self.binary_path = None

    def runSetup(self, binary):
        if self.binary_path == f'{BINARY_DIR}/{binary}':
            print(f'Binary Ninja already loaded for {self.binary_path}')
            return
        self.binary_path = f'{BINARY_DIR}/{binary}'
        print(f'Binary Ninja loading {self.binary_path}...')
        self.bv = binaryninja.open_view(self.binary_path)

    def test_echo_full_hash(self):
        self.runSetup('echo')
        echo_hash = hashashin.hash_all(self.bv, return_serializable=True, show_progress=True)
        self.assertDictEqual(echo_ground_truth, echo_hash)

    def test_echo_main_hash(self):
        self.runSetup('echo')
        main_func = self.bv.get_functions_by_name('main')[0]
        main_hash = hashashin.hash_function(self.bv.get_functions_by_name('main')[0])
        self.assertEqual(echo_ground_truth[f2str(main_func)], main_hash)

    def test_echo_main_bb_hash(self):
        self.runSetup('echo')
        first_bb = self.bv.get_functions_by_name('main')[0].basic_blocks[0]
        main_bb_hash = hashashin.hash_basic_block(first_bb)
        self.assertEqual('888888888888', main_bb_hash)

    def test_busybox_full_hash(self):
        self.runSetup('busybox')
        busybox_hash = hashashin.hash_all(self.bv, return_serializable=True, show_progress=True)
        self.assertDictEqual(busybox_ground_truth, busybox_hash)

    def test_busybox_main_hash(self):
        self.runSetup('busybox')
        main_func = self.bv.get_functions_by_name('main')[0]
        main_hash = hashashin.hash_function(self.bv.get_functions_by_name('main')[0])
        self.assertEqual(busybox_ground_truth[f2str(main_func)], main_hash)

    def test_busybox_main_bb_hash(self):
        self.runSetup('busybox')
        first_bb = self.bv.get_functions_by_name('main')[0].basic_blocks[0]
        main_bb_hash = hashashin.hash_basic_block(first_bb)
        self.assertEqual('2e196bef7f9beffa99ffbf', main_bb_hash)
