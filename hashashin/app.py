from hashashin.db import RepositoryType
from hashashin.db import AbstractHashRepository
from hashashin.classes import FeatureExtractor
from hashashin.classes import BinarySignature
from hashashin.classes import FunctionFeatures
from hashashin.utils import str2path
from hashashin.utils import get_binaries
from hashashin.sqlalchemy import SQLAlchemyHashRepository
from enum import Enum
from pathlib import Path
from typing import Union
from collections import Counter

import logging

logger = logging.getLogger(__name__)


class HashApp:
    @staticmethod
    def _repo_from_type(repo_type: RepositoryType) -> AbstractHashRepository:
        if repo_type == RepositoryType.SQLALCHEMY:
            return SQLAlchemyHashRepository()
        else:
            raise ValueError(f"Invalid repository type: {repo_type}")

    def __init__(
        self,
        repository: AbstractHashRepository = SQLAlchemyHashRepository(),
        extractor: str = "binja",
    ):
        logger.debug("Making HashApp")
        self.repo: AbstractHashRepository = repository
        self.extractor: FeatureExtractor = FeatureExtractor.from_name(extractor)

    @classmethod
    def from_type(cls, repo_type: RepositoryType, extractor: str = "binja"):
        return cls(repository=cls._repo_from_type(repo_type), extractor=extractor)

    def hash_file(self, binary_path: Union[Path, str]) -> BinarySignature:
        binary_path = str2path(binary_path)
        bs: BinarySignature = self.extractor.extract_from_file(binary_path)
        return bs

    def hash_dir(self, binary_path: Union[Path, str]) -> list[BinarySignature]:
        binary_path = str2path(binary_path)
        if binary_path.is_dir():
            return [self.hash_file(p) for p in get_binaries(binary_path)]
        else:
            raise ValueError(f"{binary_path} is not a valid directory.")

    def save(self, binaries: Union[list[BinarySignature], BinarySignature]):
        if isinstance(binaries, BinarySignature):
            binaries = [binaries]
        self.repo.saveAll(binaries)

    def save_file(self, binary_path: Union[Path, str]) -> BinarySignature:
        self.save(bs := self.hash_file(binary_path))
        return bs

    def save_dir(self, binary_path: Union[Path, str]) -> list[BinarySignature]:
        self.save(sigs := self.hash_dir(binary_path))
        return sigs

    def match(self, sig: BinarySignature, n: int = 10):
        return self.repo.match(sig, n)
        # for bs in binaries:
        #     for fn in bs.functionFeatureList:
        #         matches = self.repo.function_repo.match(fn, n)
        #         bin_matches = [m.binary for m in matches]
        #         bin_counter = Counter(bin_matches)
        #         most_common_bin = bin_counter.most_common(1)[0][0]

    def match_file(self, binary_path: Union[Path, str], n: int = 10):
        return self.match(self.hash_file(binary_path), n)

    def match_dir(self, binary_path: Union[Path, str], n: int = 10):
        for b in self.hash_dir(binary_path):
            yield self.match(b, n)


if __name__ == "__main__":
    # testing
    from hashashin.metrics import (
        compute_metrics,
        compute_matrices,
        hash_paths,
    )

    hashApp = HashApp()

    signatures = hash_paths("openssl", hashApp, paths="*[0-9][.][0-9]*")

    minhash_similarities, jaccard_similarities, binaries = compute_matrices(signatures)
    minhash_metrics = compute_metrics(minhash_similarities)
    print(
        f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
    )
    jaccard_metrics = compute_metrics(jaccard_similarities)
    print(
        f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
    )
