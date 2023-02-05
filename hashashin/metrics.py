import glob
import seaborn as sns  # type: ignore
import numpy as np
from pathlib import Path

from hashashin.main import BinaryHasherApplication


def show_similarity_matrix(mat, labels, title=None, font_scale=0.7, figsize=(14, 14)):
    if isinstance(mat, list):
        mat = np.array(mat)
    if isinstance(labels, list):
        labels = np.array(labels)
    sns.set(font_scale=font_scale, rc={"figure.figsize": figsize, "axes.titlesize": 20})
    sns.heatmap(
        mat, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True
    ).set(title=title)


def compute_matrices(
    base_binary,
    hasher: BinaryHasherApplication,
    version_paths=None,
    _feature_mask=None,
    binary_dir=Path(".").parent / "binary_data",
):
    if version_paths is None:
        version_paths = list(Path(f"{binary_dir}/{base_binary}").glob("*[0-9].[0-9]*"))
    elif isinstance(version_paths, str):
        print(f"Globbing {binary_dir}/{base_binary}/{version_paths}")
        version_paths = list(Path(f"{binary_dir}/{base_binary}").glob(version_paths))
    else:
        assert isinstance(
            version_paths, list
        ), "version_paths must be a string regex or a list of paths"
    print(f"Computing signatures for {version_paths}..")
    signatures = hasher.target(version_paths)
    binaries = sorted(list([s.path.relative_to(binary_dir) for s in signatures]))
    minhash_similarities = np.zeros((len(binaries), len(binaries)))
    jaccard_similarities = np.zeros((len(binaries), len(binaries)))

    print(f"Computing similarity matrix for {base_binary}:")
    print(",".join(binaries))
    print()
    for i, j in np.ndindex(len(binaries), len(binaries)):
        a, b = signatures[i], signatures[j]
        minhash_similarities[i, j] = a / b
        jaccard_similarities[i, j] = a - b
    return minhash_similarities, jaccard_similarities, binaries


def compute_metrics(similarity_matrix) -> tuple[float, float, float]:
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


def main():
    from hashashin.main import ApplicationFactory, HashashinApplicationContext
    from hashashin.classes import BinjaFeatureExtractor
    from hashashin.db import HashRepository

    app_context = HashashinApplicationContext(
        extractor=BinjaFeatureExtractor(),
        hash_repo=HashRepository(),
        target_path=None,
        save_to_db=True,
    )
    hashApp = ApplicationFactory.getHasher(app_context)
    minhash_similarities, jaccard_similarities, binaries = compute_matrices(
        "net-snmp", hashApp
    )
    minhash_metrics = compute_metrics(minhash_similarities)
    print(
        f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
    )
    jaccard_metrics = compute_metrics(jaccard_similarities)
    print(
        f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
    )


if __name__ == "__main__":
    main()
