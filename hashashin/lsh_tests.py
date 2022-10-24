import unittest

import numpy as np

import hashashin
import binaryninja
import os
import magic
import glob
from tqdm import tqdm
from multiprocessing import Pool

from hashashin.lsh import FUNC_TO_STR as f2str
from hashashin.utils import minhash_similarity, jaccard_similarity, load_hash, cache_hash, hex2vec, vec2hex

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_DIR = os.path.join(TEST_DIR, "binary_data")


def get_binaries(path, recursive=True):
    files = glob.glob(f"{path}/**", recursive=recursive)
    binaries = []
    for f in files:
        if os.path.isfile(f):
            if "ELF" in magic.from_file(f):
                binaries.append(f)
    return binaries


def compute_metrics(similarity_matrix):
    # Compute metrics
    tp = np.trace(similarity_matrix > 0.9) / len(similarity_matrix)
    fn = np.trace(similarity_matrix <= 0.9) / len(similarity_matrix)
    tn = (np.sum(similarity_matrix < 0.5) - np.trace(similarity_matrix < 0.5)) / (
                len(similarity_matrix) ** 2 - len(similarity_matrix))
    fp = (np.sum(similarity_matrix >= 0.5) - np.trace(similarity_matrix >= 0.5)) / (
                len(similarity_matrix) ** 2 - len(similarity_matrix))
    # Compute the precision
    precision = tp / (tp + fp)
    # Compute the recall
    recall = tp / (tp + fn)
    # Compute the F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_matrices(base_binary):
    version_paths = glob.glob(f"{BINARY_DIR}/{base_binary}/*[0-9].[0-9]*")
    binaries = set()
    for v in version_paths:
        bins = get_binaries(v)
        print(f"Hashing {len(bins)} binaries in {v}")
        for b in tqdm(bins):
            load_hash(b, progress=False, generate=False)
            binaries.add(b.replace(v, ""))
    binaries = sorted(list(binaries))
    minhash_similarities = np.zeros((len(binaries), len(binaries)))
    jaccard_similarities = np.zeros((len(binaries), len(binaries)))

    print(f'Computing similarity matrix for {base_binary}:')
    print(','.join(binaries))
    print()
    for i, j in tqdm(np.ndindex(len(binaries), len(binaries)), total=len(binaries) ** 2):
        a = f'{version_paths[0]}/{binaries[i]}'
        b = f'{version_paths[1]}/{binaries[j]}'
        sig_a, feat_a = load_hash(a, generate=False, progress=False)
        sig_b, feat_b = load_hash(b, generate=False, progress=False)
        minhash_similarities[i, j] = minhash_similarity(sig_a, sig_b)
        jaccard_similarities[i, j] = jaccard_similarity(feat_a, feat_b)
    return minhash_similarities, jaccard_similarities


class TestCases(unittest.TestCase):
    def setUp(self):
        self.binary_path = None

    def runSetup(self, binary, binary_dir=BINARY_DIR):
        if self.binary_path == f"{binary_dir}/{binary}":
            print(f"Binary Ninja already loaded for {self.binary_path}")
            return
        self.binary_path = f"{binary_dir}/{binary}"
        print(f"Binary Ninja loading {self.binary_path}...")
        self.bv = binaryninja.open_view(self.binary_path)

    def test_vector_conversion(self):
        vec = np.random.randint(0, 2**32-1, 1000, dtype=np.uint32)
        post_vec = hex2vec(vec2hex(vec))
        self.assertTrue(all(vec == post_vec))

    def test_echo_hash(self):
        self.runSetup("echo")
        sig, feats = hashashin.hash_all(self.bv, return_serializable=True)
        stored_sig, stored_feats = load_hash(f'{BINARY_DIR}/echo', generate=False)
        self.assertEqual(sig, stored_sig)
        self.assertDictEqual(feats, stored_feats)

    def test_echo_full_hash(self):
        self.runSetup("echo")
        echo_sig, echo_feats = hashashin.hash_all(
            self.bv, return_serializable=True, show_progress=True
        )
        sig, feats = load_hash(self.binary_path, generate=False)
        self.assertEqual(sig, echo_sig)
        self.assertDictEqual(feats, echo_feats)

    def test_echo_main_hash(self):
        self.runSetup("echo")
        main_func = self.bv.get_functions_by_name("main")[0]
        main_sig, main_feat = hashashin.hash_function(
            self.bv.get_functions_by_name("main")[0]
        )
        sigs, feats = load_hash(self.binary_path, generate=False)
        self.assertIn(f2str(main_func), sigs[main_sig])
        self.assertEqual(feats[f2str(main_func)], main_feat)

    @unittest.skip("Not implemented")
    def test_echo_main_bb_hash(self):
        raise NotImplementedError()
        self.runSetup("echo")
        first_bb = self.bv.get_functions_by_name("main")[0].basic_blocks[0]
        main_bb_hash = hashashin.hash_basic_block(first_bb)
        self.assertEqual("888888888888", main_bb_hash)

    def test_busybox_full_hash(self):
        self.runSetup("busybox")
        busybox_sig, busybox_features = hashashin.hash_all(
            self.bv, return_serializable=True, show_progress=True
        )
        sig, feats = load_hash(self.binary_path, generate=False)
        self.assertEqual(sig, busybox_sig)
        self.assertDictEqual(feats, busybox_features)

    def test_busybox_main_hash(self):
        self.runSetup("busybox")
        main_func = self.bv.get_functions_by_name("main")[0]
        main_hash = hashashin.hash_function(main_func)
        sigs, feats = load_hash(self.binary_path, generate=True)
        main_feats = feats[f2str(main_func)]
        self.assertTrue(all(hex2vec(main_feats) == main_hash))

    @unittest.skip("Not implemented")
    def test_busybox_main_bb_hash(self):
        raise NotImplementedError()
        self.runSetup("busybox")
        first_bb = self.bv.get_functions_by_name("main")[0].basic_blocks[0]
        signature, features = hashashin.hash_basic_block(first_bb)
        self.assertEqual("2e196bef7f9beffa99ffbf", main_bb_hash)

    def test_snmp_snmpd(self):
        base_binary = "net-snmp"
        versions = ('v5.9.2', 'v5.9.3')
        a = f'{BINARY_DIR}/{base_binary}/{versions[0]}/sbin/snmpd'
        b = f'{BINARY_DIR}/{base_binary}/{versions[1]}/sbin/snmpd'
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_snmp_agentxtrap(self):
        base_binary = "net-snmp"
        versions = ('v5.9.2', 'v5.9.3')
        a = f'{BINARY_DIR}/{base_binary}/{versions[0]}/bin/agentxtrap'
        b = f'{BINARY_DIR}/{base_binary}/{versions[1]}/bin/agentxtrap'
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_snmp_encode_keychange(self):
        base_binary = "net-snmp"
        versions = ('v5.9.2', 'v5.9.3')
        a = f'{BINARY_DIR}/{base_binary}/{versions[0]}/bin/encode_keychange'
        b = f'{BINARY_DIR}/{base_binary}/{versions[1]}/bin/encode_keychange'
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_snmp_snmptrapd(self):
        base_binary = "net-snmp"
        versions = ('v5.9.2', 'v5.9.3')
        a = f'{BINARY_DIR}/{base_binary}/{versions[0]}/sbin/snmptrapd'
        b = f'{BINARY_DIR}/{base_binary}/{versions[1]}/sbin/snmptrapd'
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_difference(self):
        base_binary = "net-snmp"
        a = f'{BINARY_DIR}/{base_binary}/v5.9.2/sbin/snmpd'
        b = f'{BINARY_DIR}/{base_binary}/v5.9.2/sbin/snmptrapd'
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(jaccard_sim, 0.5)
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_net_snmp(self):
        minhash_similarities, jaccard_similarities = compute_matrices("net-snmp")
        minhash_metrics = compute_metrics(minhash_similarities)
        print(f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}")
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}")
        self.assertGreaterEqual(minhash_metrics[2], 0.9)

    def test_openssl(self):
        minhash_similarities, jaccard_similarities = compute_matrices("openssl")
        print('\n'.join([','.join([str(y) for y in list(x)]) for x in minhash_similarities]))
        print()
        print('\n'.join([','.join([str(y) for y in list(x)]) for x in jaccard_similarities]))
        print()

        minhash_metrics = compute_metrics(minhash_similarities)
        print(f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}")
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}")
        self.assertGreaterEqual(minhash_metrics[2], 0.9)

    def test_libcurl(self):
        minhash_similarities, jaccard_similarities = compute_matrices("libcurl")
        print('\n'.join([','.join([str(y) for y in list(x)]) for x in minhash_similarities]))
        print()
        print('\n'.join([','.join([str(y) for y in list(x)]) for x in jaccard_similarities]))
        print()

        minhash_metrics = compute_metrics(minhash_similarities)
        print(f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}")
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}")
        self.assertGreaterEqual(minhash_metrics[2], 0.9)
