import logging
from pathlib import Path

import numpy as np
import seaborn as sns  # type: ignore
from sklearn.preprocessing import normalize
from tqdm import tqdm
from typing import Tuple, List

from hashashin.classes import BinarySignature
from hashashin.utils import logger

logger.setLevel(logging.INFO)
logger = logger.getChild(Path(__file__).name)


def show_similarity_matrix(mat, labels, title=None, font_scale=0.7, figsize=None):
    if figsize is None:
        figsize = (14, 14)
    if isinstance(mat, list):
        mat = np.array(mat)
    if isinstance(labels, list):
        labels = np.array(labels)
    sns.set(font_scale=font_scale, rc={"figure.figsize": figsize, "axes.titlesize": 20})
    sns.heatmap(
        mat, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True
    ).set(title=title)


def hash_paths(
    base_binary,
    hasher: "BinaryHasherApplication",
    paths=None,
    binary_dir=Path(".").parent / "binary_data",
):
    if paths is None:
        paths = list(Path(f"{binary_dir}/{base_binary}").glob("*[0-9][._][0-9]*"))
    elif isinstance(paths, str):
        print(f"Globbing {binary_dir}/{base_binary}/{paths}")
        paths = list(Path(f"{binary_dir}/{base_binary}").glob(paths))
    else:
        assert isinstance(
            paths, list
        ), "paths must be a string regex or a list of paths"
    logger.info(f"Computing signatures for {paths}..")
    signatures = hasher.target(paths)
    logger.info("Done computing signatures.")
    return signatures


def compute_matrices(signatures) -> Tuple[np.ndarray, np.ndarray, list]:
    binaries = sorted(
        list([s.path.relative_to(Path(".").parent / "binary_data") for s in signatures])
    )
    minhash_similarities = np.zeros((len(binaries), len(binaries)))
    jaccard_similarities = np.zeros((len(binaries), len(binaries)))

    logger.info(f"Computing similarity matrix")
    logger.debug(",".join(b.name for b in binaries))
    np_signatures = [
        s.np_signature for s in tqdm(signatures, desc="Converting signatures to numpy")
    ]
    function_sigs = [
        [f.signature for f in s.functionFeatureList] for s in tqdm(signatures)
    ]
    for i, j in tqdm(
        np.ndindex(len(binaries), len(binaries)),
        total=len(binaries) ** 2,
        desc="Computing similarities",
    ):
        minhash_similarities[i, j] = BinarySignature.minhash_similarity(
            np_signatures[i], np_signatures[j]
        )
        jaccard_similarities[i, j] = BinarySignature.jaccard_estimate(
            function_sigs[i], function_sigs[j]
        )
    return minhash_similarities, jaccard_similarities, binaries


def compute_metrics(similarity_matrix) -> Tuple[float, float, float]:
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


def reciprocal_rank_fusion(scores):
    ranks = (1 + np.arange(len(scores))) / (1 + np.argsort(np.argsort(-scores)))
    return np.sum(1 / ranks)


def matrix_norms(fc1: np.ndarray, fc2: np.ndarray) -> float:
    scores = np.matmul(normalize(fc1, axis=1), normalize(fc2.T, axis=0))
    return (
        scores.max(axis=0).sum() / scores.shape[1]
        + scores.max(axis=1).sum() / scores.shape[0]
    )


def stacked_norms(fc1: List[np.ndarray], fc2: np.ndarray) -> List[float]:
    """Compute the norms of the stacked matrix of fc1 against fc2."""
    return [matrix_norms(x, fc2) for x in fc1]
    # if len(fc1.shape) != 3 or len(fc2.shape) != 3:
    #     raise ValueError("supplied matrices must be 3D")
    # breakpoint()
    # fc1_norm = np.stack([normalize(x, axis=1) for x in fc1])
    # fc2_norm = np.stack([normalize(x, axis=0) for x in fc2.T])
    # scores = np.matmul(fc1_norm, fc2_norm.T)
    # max_row_scores = np.max(scores, axis=2)
    # max_col_scores = np.max(scores, axis=1)
    # norms = (max_row_scores.sum(axis=1) + max_col_scores.sum(axis=0)) / (scores.shape[1] + scores.shape[0])
    # return norms


def generate_matrix_norms(base_path, hashApp, paths, rescale=None):
    signatures = sorted(
        hash_paths(base_path, hashApp, paths),
        key=lambda s: s.path.relative_to(Path(".").parent / "binary_data"),
    )
    binaries = list([s.path.relative_to(Path(".").parent / "binary_data") for s in signatures])
    logger.info("Converting signatures to function matrices..")
    np_signatures = [b.function_matrix for b in signatures]
    norms = np.zeros((len(binaries), len(binaries)))
    for i, j in tqdm(
        np.ndindex(len(binaries), len(binaries)),
        total=len(binaries) ** 2,
        desc="Computing norms",
    ):
        if norms[j, i] != 0:
            norms[i, j] = norms[j, i]
            continue
        norms[i, j] = matrix_norms(np_signatures[i], np_signatures[j])
    p, r, f1 = compute_metrics(norms)
    logger.info(f"Precision: {p}, Recall: {r}, F1: {f1}")
    if rescale is not None:
        _norms = rescale(norms)
    else:
        _norms = norms
    show_similarity_matrix(
        _norms,
        binaries,
        f"{base_path} matrix norms",
        figsize=(20, 20) if len(binaries) > 10 else None,
    )
    return norms, binaries


def main():
    from hashashin.main import ApplicationFactory, HashashinApplicationContext
    from hashashin.classes import BinjaFeatureExtractor
    from hashashin.db import HashRepository

    app_context = HashashinApplicationContext(
        extractor=BinjaFeatureExtractor(),
        hash_repo=HashRepository(),
        target_path=None,
        save_to_db=True,
        progress=True,
    )
    hashApp = ApplicationFactory.getHasher(app_context)

    run_snmp = False
    if run_snmp:
        signatures = hash_paths("net-snmp", hashApp, paths="*[0-9][.][0-9]*")

        minhash_similarities, jaccard_similarities, binaries = compute_matrices(
            signatures
        )
        minhash_metrics = compute_metrics(minhash_similarities)
        print(
            f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
        )
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(
            f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
        )

    run_curl = False
    if run_curl:
        signatures = hash_paths("libcurl", hashApp, paths="*[0-9][_][0-9]*")

        minhash_similarities, jaccard_similarities, binaries = compute_matrices(
            signatures
        )
        minhash_metrics = compute_metrics(minhash_similarities)
        print(
            f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
        )
        jaccard_metrics = compute_metrics(jaccard_similarities)
        print(
            f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
        )

    # norms = generate_matrix_norms("libcurl", hashApp, paths="*[0-9]_[0-9]*")
    norms = generate_matrix_norms("openssl", hashApp, paths="*[0-9].[0-9]*")
    print(norms)


if __name__ == "__main__":
    main()
