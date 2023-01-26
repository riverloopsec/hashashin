from hashashin.feature_extractors import FeatureExtractor, BinjaFeatureExtractor
from hashashin.db import (
    FunctionFeatureRepository,
    BinarySignatureRepository,
    SQLAlchemyFunctionFeatureRepository,
    SQLAlchemyBinarySignatureRepository,
)
from hashashin.classes import BinarySignature
from dataclasses import dataclass
from enum import Enum
from abc import ABC
from pathlib import Path
from typing import Optional
from hashashin.utils import get_binaries


class Task(Enum):
    HASH = 1
    MATCH = 2


@dataclass
class HashashinApplicationContext:
    extractor: FeatureExtractor
    feature_repo: FunctionFeatureRepository
    binary_repo: BinarySignatureRepository


@dataclass
class HashashinFactoryContext:
    context: HashashinApplicationContext
    task: Task


class HashashinApplication(ABC):
    def __init__(self, context: HashashinApplicationContext):
        self.context = context

    def run(self):
        raise NotImplementedError


class BinaryMatcherApplication(HashashinApplication):
    def __init__(
        self, context: HashashinApplicationContext, target_path: Optional[Path] = None
    ):
        super().__init__(context)
        self.target_path = target_path

    def _match(self, sig: BinarySignature, threshold: int = 0) -> list[BinarySignature]:
        """
        Match a binary signature against the database.
        :param sig: signature to match
        :param threshold: distance threshold to match
        :return:
        """
        return self.context.binary_repo.match_signature(sig, threshold)

    def run(self):
        if self.target_path is None:
            raise ValueError("No target path specified")
        if not self.target_path.exists():
            raise ValueError("Target path does not exist")
        targets = get_binaries(self.target_path)
        matches = []
        target_signature = self.context.extractor.extract_from_file(self.target_path)
        matches = self._match(target_signature)
        if len(matches) == 0:
            print(f"No matches found for {self.target_path}")
            return
        print(f"Matches for {self.target_path}:")
        for match in matches:
            print(f"\t{match}")
        return matches

    def match_target(self, target: Path):
        self.target_path = target
        return self.run()


class BinaryHasherApplication(HashashinApplication):
    def __init__(
        self, context: HashashinApplicationContext, target_binary: Optional[Path]
    ):
        super().__init__(context)
        self.target_binary = target_binary

    def _hash(self, binary: Path) -> BinarySignature:
        """
        Hash a binary and save it to the database.
        :param binary: binary to hash
        :return: signature of the binary
        """
        sig = self.context.extractor.extract_from_file(binary)
        if self.context.feature_repo is not None:
            self.context.feature_repo.store_features(sig.features)
        if self.context.binary_repo is not None:
            self.context.binary_repo.store_signature(sig)
        return sig

    def run(self):
        if self.target_binary is None:
            raise ValueError("No target binary specified")
        if not self.target_binary.exists():
            raise ValueError("Target binary does not exist")
        if self.target_binary.is_dir():
            for binary in self.target_binary.iterdir():
                if binary.is_file():
                    self._hash(binary)
        else:
            self._hash(self.target_binary)


class HashashinFactory:
    def __init__(self, context: HashashinFactoryContext):
        self.context = context

    def create(self) -> BinaryMatcherApplication:
        if self.context.task == Task.MATCH:
            return BinaryMatcherApplication(
                feature_extractor=self.context.extractor,
                feature_repository=self.context.feature_repo,
                binary_repository=self.context.binary_repo,
            )
        elif self.context.task == Task.HASH:
            return BinaryMatcherApplication(
                feature_extractor=self.context.extractor,
                feature_repository=self.context.feature_repo,
                binary_repository=self.context.binary_repo,
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import argparse
    import os
    from hashashin.utils import get_binja_path
    from hashashin.db import init_db

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", "-t", type=str, choices=["hash", "match"], required=True
    )
    parser.add_argument("--target", "-b", type=str, required=True)
    parser.add_argument("--db", "-d", type=str, required=True)
    parser.add_argument("--binja", "-p", type=str, required=True)
    parser.add_argument("--threshold", "-r", type=int, default=0)
    args = parser.parse_args()

    binja_path = get_binja_path(args.binja)
    if binja_path is None:
        raise ValueError("Could not find Binary Ninja executable")

    init_db(args.db)
    extractor = BinjaFeatureExtractor(binja_path)
    feature_repo = SQLAlchemyFunctionFeatureRepository()
    binary_repo = SQLAlchemyBinarySignatureRepository()
    context = HashashinApplicationContext(extractor, feature_repo, binary_repo)
    factory_context = HashashinFactoryContext(context, Task[args.task.upper()])
    factory = HashashinFactory(factory_context)
    app = factory.create()
    app.run()
