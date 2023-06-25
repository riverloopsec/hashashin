from hashashin.db import HashRepository
from hashashin.db import RepositoryType
from hashashin.classes import FeatureExtractor
from hashashin.classes import BinarySignature
from hashashin.classes import FunctionFeatures
from hashashin.utils import str2path
from hashashin.utils import get_binaries
from enum import Enum
from pathlib import Path
from typing import Union
from collections import Counter

import logging

logger = logging.getLogger(__name__)

class HashApp:
    def __init__(self,
                 repo_type: RepositoryType = RepositoryType.SQLALCHEMY,
                 name: str = "binja"):
        logger.debug("Making HashApp")
        self.repo = HashRepository(repo_type=repo_type)
        self.extractor = FeatureExtractor.from_name(name)

    def _hash(self, binary_path: Union[Path, str]) -> BinarySignature:
        binary_path = str2path(binary_path)
        bs: BinarySignature = self.extractor.extract_from_file(binary_path)
        return bs

    def hash(self, binary_path: Union[Path, str]) -> list[BinarySignature]:
        binary_path = str2path(binary_path)
        if binary_path.is_dir():
            return [self._hash(p) for p in get_binaries(binary_path)]
        elif binary_path.is_file():
            return [self._hash(binary_path)]
        else:
            raise ValueError(f"Invalid path: {binary_path}")

    def save(self, binary_path: Union[Path, str]):
        for bs in [self.hash(p) for p in get_binaries(binary_path)]:
            self.repo.save(bs)

    # def _top_n(self, fn: FunctionFeatures, n: int = 5):
    #     return self.repo.function_repo.match(fn, n)

    def match(self, binary_path: Union[Path, str], n: int = 10):
        for bs in self.hash(binary_path):
            for fn in bs.functionFeatureList:
                matches = self.repo.function_repo.match(fn, n)
                bin_matches = [m.binary for m in matches]
                bin_counter = Counter(bin_matches)
                most_common_bin = bin_counter.most_common(1)[0][0]
