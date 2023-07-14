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
import os
from typing import Iterable
from multiprocessing import Pool

import logging

logger = logging.getLogger(__name__)


class HashApp:
    @staticmethod
    def _repo_from_type(repo_type: RepositoryType) -> AbstractHashRepository:
        if repo_type == RepositoryType.SQLALCHEMY:
            return SQLAlchemyHashRepository()
        else:
            raise ValueError(f"Invalid repository type: {repo_type}")

    @staticmethod
    def _initialize_logger(level: int = logging.DEBUG):
        if not logger.hasHandlers():
            logging.basicConfig(
                format="%(asctime)s,%(msecs)03d %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d:%H:%M:%S",
                level=level,
            )

    def __init__(
        self,
        repository: AbstractHashRepository = SQLAlchemyHashRepository(),
        extractor: str = "binja",
        loglevel: int = logging.DEBUG,
        multiprocessing: int = 0,  # os.cpu_count() // 2 + 1,
    ):
        self._initialize_logger(loglevel)
        logger.debug("Making HashApp")
        self.repo: AbstractHashRepository = repository
        self.extractor: FeatureExtractor = FeatureExtractor.from_name(extractor)
        self._pool = Pool(processes=multiprocessing) if multiprocessing else None

    @classmethod
    def from_type(cls, repo_type: RepositoryType, extractor: str = "binja"):
        return cls(repository=cls._repo_from_type(repo_type), extractor=extractor)

    def hash_file(self, binary_path: Union[Path, str]) -> BinarySignature:
        print(binary_path)
        binary_path = str2path(binary_path)
        bs: BinarySignature = self.extractor.extract_from_file(binary_path)
        return bs

    def hash_dir(self, binary_path: Union[Path, str]) -> list[BinarySignature]:
        binary_path = str2path(binary_path)
        targets = get_binaries(binary_path)
        if self._pool:
            # TODO: fix pickling error
            raise NotImplementedError
            # submissions = [self._executor.submit(self.hash_file, p) for p in targets]
            # return [s.result() for s in futures.as_completed(submissions)]
        out = list()
        for p in targets:
            try:
                out.append(self.hash_file(p))
            except FeatureExtractor.NotABinaryError:
                logger.warning(f"Skipping non-binary file: {p}")
                continue
        return out

    def hash_path(self, binary_path: Union[Path, str]) -> list[BinarySignature]:
        binary_path = str2path(binary_path)
        if binary_path.is_dir():
            return self.hash_dir(binary_path)
        if binary_path.is_file():
            return [self.hash_file(binary_path)]
        raise ValueError(f"Invalid path: {binary_path}")

    def hash_list(self, bins: Iterable[Path]) -> list[BinarySignature]:
        out = list()
        if self._pool is not None:
            raise NotImplementedError
            # pooled_ret = self._pool.map(self.hash_path, bins)
            # for ret in pooled_ret:
            #     out.extend(ret)
            # return out
        for target in map(Path, bins):
            if not target.is_file() and not (
                target.is_dir() and len(get_binaries(target, progress=True)) > 0
            ):
                logger.debug(f"Skipping {target} as it is not a binary")
                continue
            logger.info(f"Hashing {target}")
            out.extend(self.hash_path(target))
        return out

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

    def match(self, sig: BinarySignature, n: int = 10) -> "QueryResult":
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

    @staticmethod
    def _log_summary(db: AbstractHashRepository, path_filter: str = ""):
        logger.debug("Printing database summary")
        num_binaries, num_functions = db.summary(path_filter)
        msg = f"*{path_filter}*" if path_filter else "all"
        logger.info(f"Summary for {msg} binary paths:")
        logger.info(f"\tBinaries: {num_binaries}")
        logger.info(f"\tFunctions: {num_functions}")

    def log_summary(self):
        self._log_summary(self.repo)


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
