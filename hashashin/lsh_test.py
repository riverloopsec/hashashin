import glob
import os
import unittest

import binaryninja
import magic
import numpy as np
from tqdm import tqdm

import hashashin
from hashashin.utils import deserialize_features
from hashashin.utils import func2str
from hashashin.utils import hex2vec
from hashashin.utils import jaccard_similarity
from hashashin.utils import load_hash
from hashashin.utils import minhash_similarity
from hashashin.utils import serialize_features
from hashashin.utils import vec2hex
from hashashin.utils import features_to_dict
from hashashin.utils import dict_to_features

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_DIR = os.path.join(TEST_DIR, "binary_data")


def get_binaries(path, bin_name=None, recursive=True, progress=False):
    if os.path.isfile(path):
        files = [path]
    elif bin_name is None:
        files = glob.glob(f"{path}/**", recursive=recursive)
    else:
        files = glob.glob(f"{path}/**/{bin_name}", recursive=recursive)
    binaries = []
    for f in tqdm(files, disable=not progress):
        if os.path.isfile(f):
            if "ELF" in magic.from_file(f):
                binaries.append(f)
    return binaries


def const_to_numpy(constants):
    return np.array(sorted([c.value for c in constants]), dtype=np.int32)


def compute_metrics(similarity_matrix):
    # Compute metrics
    tp = np.trace(similarity_matrix > 0.9) / len(similarity_matrix)
    fn = np.trace(similarity_matrix <= 0.9) / len(similarity_matrix)
    tn = (np.sum(similarity_matrix < 0.5) - np.trace(similarity_matrix < 0.5)) / (
        len(similarity_matrix) ** 2 - len(similarity_matrix)
    )
    fp = (np.sum(similarity_matrix >= 0.5) - np.trace(similarity_matrix >= 0.5)) / (
        len(similarity_matrix) ** 2 - len(similarity_matrix)
    )
    # Compute the precision
    precision = tp / (tp + fp)
    # Compute the recall
    recall = tp / (tp + fn)
    # Compute the F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def fmask(mask: slice, content: str):
    '''Mask a feature string with a slice. Use step=1 to return the masked feature.'''
    zeros = ['0'] * len(content)
    _content = list(content)
    invert = mask.step is not None
    mask = slice(mask.start * 8, mask.stop * 8)
    if invert:
        zeros[mask] = _content[mask]
        return ''.join(zeros)
    else:
        _content[mask] = zeros[mask]
        return ''.join(_content)


def compute_matrices(base_binary, generate=True, regenerate=False, version_paths=None, hash_progress=False, _feature_mask=None):
    if version_paths is None:
        version_paths = glob.glob(f"{BINARY_DIR}/{base_binary}/*[0-9].[0-9]*")
    elif isinstance(version_paths, str):
        print(f"Globbing {BINARY_DIR}/{base_binary}/{version_paths}")
        version_paths = glob.glob(f"{BINARY_DIR}/{base_binary}/{version_paths}")
    else:
        assert isinstance(version_paths, list), "version_paths must be a string regex or a list of paths"
    binaries = set()
    for v in version_paths:
        bins = get_binaries(v)
        print(f"Hashing {len(bins)} binaries in {v}: {[b.replace(v, '') for b in bins]}")
        for b in tqdm(bins, disable=hash_progress):
            load_hash(b, progress=hash_progress, generate=generate, regenerate=regenerate)
            binaries.add(b.replace(v, ""))
    binaries = sorted(list(binaries))
    minhash_similarities = np.zeros((len(binaries), len(binaries)))
    jaccard_similarities = np.zeros((len(binaries), len(binaries)))

    print(f"Computing similarity matrix for {base_binary}:")
    print(",".join(binaries))
    print()
    for i, j in tqdm(
        np.ndindex(len(binaries), len(binaries)), total=len(binaries) ** 2
    ):
        a = f"{version_paths[0]}/{binaries[i]}"
        b = f"{version_paths[1]}/{binaries[j]}"
        sig_a, feat_a = load_hash(a, generate=False, progress=False)
        sig_b, feat_b = load_hash(b, generate=False, progress=False)
        if _feature_mask is not None:
            feat_a = {k: fmask(_feature_mask, v) for k, v in feat_a.items()}
            feat_b = {k: fmask(_feature_mask, v) for k, v in feat_b.items()}
        minhash_similarities[i, j] = minhash_similarity(sig_a, sig_b)
        jaccard_similarities[i, j] = jaccard_similarity(feat_a, feat_b)
    return minhash_similarities, jaccard_similarities, binaries


def compute_single_bin_matrices(base_binary, binary, generate=True, regenerate=False, hash_progress=False):
    binaries = sorted(get_binaries(f"{BINARY_DIR}/{base_binary}", bin_name=binary))
    for b in tqdm(binaries, disable=hash_progress):
        load_hash(b, progress=hash_progress, generate=generate, regenerate=regenerate)
    minhash_similarities = np.zeros((len(binaries), len(binaries)))
    jaccard_similarities = np.zeros((len(binaries), len(binaries)))

    for i, j in tqdm(
        np.ndindex(len(binaries), len(binaries)), total=len(binaries) ** 2
    ):
        a = binaries[i]
        b = binaries[j]
        sig_a, feat_a = load_hash(a, generate=False, progress=False)
        sig_b, feat_b = load_hash(b, generate=False, progress=False)
        minhash_similarities[i, j] = minhash_similarity(sig_a, sig_b)
        jaccard_similarities[i, j] = jaccard_similarity(feat_a, feat_b)
    return minhash_similarities, jaccard_similarities, binaries


def print_similarity_matrix(matrix: list[list], labels: list):
    assert len(matrix) == len(matrix[0]) == len(labels)
    print(",".join([''] + labels))
    print("\n".join(",".join(str(x) for x in labels[i:i+1] + list(matrix[i])) for i in range(len(matrix))))


class TestCases(unittest.TestCase):
    def setUp(self):
        self.binary_path = None

    def runSetup(self, binary, binary_dir=BINARY_DIR):
        if self.binary_path == f"{binary_dir}/{binary}":
            print(f"Binary Ninja already loaded for {self.binary_path}")
            return
        self.binary_path = f"{binary_dir}/{binary}"
        print(f"Binary Ninja loading {self.binary_path}...")
        self.bv = binaryninja.open_view(self.binary_path, options={"analysis.experimental.gratuitousFunctionUpdate": True})

    def test_vector_conversion(self):
        vec = np.random.randint(0, 2**32 - 1, 1000, dtype=np.uint32)
        post_vec = hex2vec(vec2hex(vec))
        self.assertTrue(all(vec == post_vec))

    def test_serializer(self):
        self.runSetup("echo")
        sig, feats = hashashin.hash_all(self.bv, return_serializable=False)
        a = b = feats
        b = deserialize_features(serialize_features(b), self.bv)
        self.assertEqual(a.keys(), b.keys())
        self.assertTrue(all([all(a[k] == b[k]) for k in a.keys()]))

    def test_features_to_dict(self):
        self.runSetup("busybox")
        func = self.bv.get_functions_by_name("main")[0]
        feat = hashashin.hash_function(func)
        print(features_to_dict(feat))
        print(features_to_dict(dict_to_features(features_to_dict(feat))))
        self.assertTrue(all(feat == dict_to_features(features_to_dict(feat))))

    def test_get_constants(self):
        self.runSetup("busybox")
        function_addr = 0x7e1c4
        binaryninja.open_view(self.binary_path, options={"analysis.experimental.gratuitousFunctionUpdate": True})
        f = self.bv.get_function_at(function_addr)
        constants = hashashin.get_constants(f)
        for _ in tqdm(range(10)):
            _view = binaryninja.open_view(self.binary_path, options={"analysis.experimental.gratuitousFunctionUpdate": True})
            new_constants = hashashin.get_constants(_view.get_function_at(function_addr))
            if constants != new_constants:
                print("Constants changed between views")
                hashashin.get_constants(_view.get_function_at(function_addr))
            _view.update_analysis_and_wait()
            new_constants = hashashin.get_constants(_view.get_function_at(function_addr))
            if constants != new_constants:
                print("Constants changed between views and analysis")
                hashashin.get_constants(_view.get_function_at(function_addr))
        test = [
            constants
            == hashashin.get_constants(
                binaryninja.open_view(self.binary_path).get_function_at(0x22A0C)
            )
            for _ in tqdm(range(10))
        ]
        if not all(test):
            print("Failed to get constants consistently")
        self.assertTrue(all(test))

    def test_echo_hash(self):
        self.runSetup("echo")
        sig, feats = hashashin.hash_all(self.bv, return_serializable=True)
        stored_sig, stored_feats = load_hash(f"{BINARY_DIR}/echo", generate=True)
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
        main_feat = hashashin.hash_function(self.bv.get_functions_by_name("main")[0])
        _, feats = load_hash(self.binary_path, generate=False)
        self.assertTrue(all(hex2vec(feats[func2str(main_func)]) == main_feat))

    @unittest.skip("Not implemented")
    def test_echo_main_bb_hash(self):
        raise NotImplementedError()
        self.runSetup("echo")
        first_bb = self.bv.get_functions_by_name("main")[0].basic_blocks[0]
        main_bb_hash = hashashin.hash_basic_block(first_bb)
        self.assertEqual("888888888888", main_bb_hash)

    def test_echo_deterministic(self):
        self.runSetup("echo")
        sig, feats = hashashin.hash_all(
            self.bv, return_serializable=False, show_progress=True
        )
        sig2, feats2 = hashashin.hash_all(
            self.bv, return_serializable=False, show_progress=True
        )
        self.assertEqual(sig, sig2)
        self.assertEqual(feats.keys(), feats2.keys())
        self.assertTrue(all([all(feats[k] == feats2[k]) for k in feats.keys()]))

    def test_file_push(self):
        self.runSetup("busybox")
        sig, feats = load_hash(
            self.binary_path,
            regenerate=True,
            progress=True,
            deserialize=True,
            bv=self.bv,
        )
        sig2, feats2 = hashashin.hash_all(
            self.bv, return_serializable=False, show_progress=True
        )
        self.assertEqual(sig, sig2)
        self.assertEqual(feats.keys(), feats2.keys())
        self.assertTrue(all([all(feats[k] == feats2[k]) for k in feats.keys()]))

    def test_busybox_deterministic(self):
        self.runSetup("busybox")
        sig, feats = hashashin.hash_all(
            self.bv, return_serializable=False, show_progress=True
        )
        self.runSetup("busybox")
        sig2, feats2 = hashashin.hash_all(
            self.bv, return_serializable=False, show_progress=True
        )
        self.assertEqual(sig, sig2)
        self.assertEqual(feats.keys(), feats2.keys())
        self.assertTrue(all([all(feats[k] == feats2[k]) for k in feats.keys()]))
        # k = list(feats.keys())[  # use for testing
        #     [all(feats[k] == feats2[k]) for k in feats.keys()].index(False)
        # ]
        # print(k, hex(k.start))
        # print(features_to_dict(feats[k]))
        # print(features_to_dict(feats2[k]))

    def test_busybox_full_hash(self):
        self.runSetup("busybox")
        busybox_sig, busybox_features = hashashin.hash_all(
            self.bv,
            return_serializable=False,
            show_progress=True,
        )
        sig, feats = load_hash(
            self.binary_path, generate=True, progress=True, deserialize=True, bv=self.bv
        )
        self.assertEqual(sig, busybox_sig)
        self.assertEqual(feats.keys(), busybox_features.keys())
        print(all([all(feats[k] == busybox_features[k]) for k in feats.keys()]))
        self.assertTrue(
            all([all(feats[k] == busybox_features[k]) for k in feats.keys()])
        )

    def test_busybox_main_hash(self):
        self.runSetup("busybox")
        main_func = self.bv.get_functions_by_name("main")[0]
        main_hash = hashashin.hash_function(main_func)
        sigs, feats = load_hash(self.binary_path, generate=True)
        main_feats = feats[func2str(main_func)]
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
        versions = ("v5.9.2", "v5.9.3")
        a = f"{BINARY_DIR}/{base_binary}/{versions[0]}/sbin/snmpd"
        b = f"{BINARY_DIR}/{base_binary}/{versions[1]}/sbin/snmpd"
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_snmp_agentxtrap(self):
        base_binary = "net-snmp"
        versions = ("v5.9.2", "v5.9.3")
        a = f"{BINARY_DIR}/{base_binary}/{versions[0]}/bin/agentxtrap"
        b = f"{BINARY_DIR}/{base_binary}/{versions[1]}/bin/agentxtrap"
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_snmp_encode_keychange(self):
        base_binary = "net-snmp"
        versions = ("v5.9.2", "v5.9.3")
        a = f"{BINARY_DIR}/{base_binary}/{versions[0]}/bin/encode_keychange"
        b = f"{BINARY_DIR}/{base_binary}/{versions[1]}/bin/encode_keychange"
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_snmp_snmptrapd(self):
        base_binary = "net-snmp"
        versions = ("v5.9.2", "v5.9.3")
        a = f"{BINARY_DIR}/{base_binary}/{versions[0]}/sbin/snmptrapd"
        b = f"{BINARY_DIR}/{base_binary}/{versions[1]}/sbin/snmptrapd"
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_difference(self):
        base_binary = "net-snmp"
        a = f"{BINARY_DIR}/{base_binary}/v5.9.2/sbin/snmpd"
        b = f"{BINARY_DIR}/{base_binary}/v5.9.2/sbin/snmptrapd"
        sig_a, feat_a = load_hash(a, generate=False)
        sig_b, feat_b = load_hash(b, generate=False)
        minhash_sim = minhash_similarity(sig_a, sig_b)
        jaccard_sim = jaccard_similarity(feat_a, feat_b)
        print(f"Minhash similarity: {minhash_sim}")
        print(f"Jaccard similarity: {jaccard_sim}")
        self.assertLess(jaccard_sim, 0.5)
        self.assertLess(abs(minhash_sim - jaccard_sim), 0.1)

    def test_net_snmp(self):
        minhash_similarities, jaccard_similarities, binaries = compute_matrices("net-snmp")
        print_similarity_matrix(minhash_similarities, binaries)
        print_similarity_matrix(jaccard_similarities, binaries)
        minhash_metrics = compute_metrics(minhash_similarities)
        print(
            f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
        )
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(
            f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
        )
        self.assertGreaterEqual(minhash_metrics[2], 0.9)

    def test_openssl(self):
        minhash_similarities, jaccard_similarities = compute_matrices("openssl")
        print(
            "\n".join(
                [",".join([str(y) for y in list(x)]) for x in minhash_similarities]
            )
        )
        print()
        print(
            "\n".join(
                [",".join([str(y) for y in list(x)]) for x in jaccard_similarities]
            )
        )
        print()

        minhash_metrics = compute_metrics(minhash_similarities)
        print(
            f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
        )
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(
            f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
        )
        self.assertGreaterEqual(minhash_metrics[2], 0.9)

    def test_libcurl(self):
        minhash_similarities, jaccard_similarities = compute_matrices("libcurl")
        print(
            "\n".join(
                [",".join([str(y) for y in list(x)]) for x in minhash_similarities]
            )
        )
        print()
        print(
            "\n".join(
                [",".join([str(y) for y in list(x)]) for x in jaccard_similarities]
            )
        )
        print()

        minhash_metrics = compute_metrics(minhash_similarities)
        print(
            f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
        )
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(
            f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
        )
        self.assertGreaterEqual(minhash_metrics[2], 0.9)

    def test_zero_constants(self):
        minhash_similarities, jaccard_similarities, binaries = compute_matrices("net-snmp", regenerate=False,
                                                                                generate=False,
                                                                                _feature_mask=slice(4, 68))
        minhash_metrics = compute_metrics(minhash_similarities)
        print(
            f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
        )
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(
            f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
        )

    def test_split_large_int(self):
        from hashashin.utils import split_int_to_uint32, merge_uint32_to_int
        import random
        x = random.randint(2 ** 128, 2 ** 129)
        print(x, split_int_to_uint32(x))
        y = merge_uint32_to_int(split_int_to_uint32(x))
        self.assertEqual(x, y)
