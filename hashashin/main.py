import glob
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Iterable, Optional, Union

import magic
from tqdm import tqdm

from hashashin.classes import BinarySignature
from hashashin.db import (BinarySignatureRepository, FunctionFeatureRepository,
                          SQLAlchemyBinarySignatureRepository,
                          SQLAlchemyFunctionFeatureRepository)
from hashashin.feature_extractors import (BinjaFeatureExtractor,
                                          FeatureExtractor)
from hashashin.utils import get_binaries
import logging


logger = logging.getLogger(os.path.basename(__file__))


class Task(Enum):
    HASH = "hash"
    MATCH = "match"


@dataclass
class HashashinApplicationContext:
    extractor: FeatureExtractor
    feature_repo: FunctionFeatureRepository
    binary_repo: BinarySignatureRepository
    target_path: Optional[Path]
    save_to_db: Optional[bool]
    sig_match_threshold: float = 0.5
    # feat_match_threshold: int = 0
    progress: bool = False


@dataclass
class HashashinFactoryContext:
    app_context: HashashinApplicationContext
    task: Task


class HashashinApplication(ABC):
    def __init__(self, context: HashashinApplicationContext):
        self.context = context

    def run(self):
        raise NotImplementedError

    def saveToDB(
        self,
        signatures: Union[Collection[BinarySignature], BinarySignature],
    ):
        if isinstance(signatures, BinarySignature):
            signatures = [signatures]
        for sig in signatures:
            binary = self.context.binary_repo.store_signature(sig)
            for feat in sig.functionFeatureList:
                feat.binary_id = binary.id
                self.context.feature_repo.store_feature(feat)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.context})"


class BinaryMatcherApplication(HashashinApplication):

    def _match(self, sig: BinarySignature) -> list[BinarySignature]:
        """
        Match a binary signature against the database.
        :param sig: signature to match
        :return: a list of matching signatures sorted by distance
        """
        return self.context.binary_repo.match_signature(sig, self.context.sig_match_threshold)

    def step(self) -> Iterable[tuple[BinarySignature, list[BinarySignature]]]:
        """
        Match all binaries in the target path against the database.
        :return: a generator of tuples containing the target signature and a list of matches sorted by distance
        """
        if self.context.target_path is None:
            raise ValueError("No target path specified")
        if not self.context.target_path.exists():
            raise ValueError("Target path does not exist")
        targets = get_binaries(self.context.target_path)
        for target_path in targets:
            target_signature = self.context.extractor.extract_from_file(target_path)
            if self.context.save_to_db:
                self.saveToDB(target_signature)
            matches = self._match(target_signature)
            if len(matches) == 0:
                print(f"No matches found for {target_path}")
                continue
            print(f"Matches for {target_path}:")
            for match in matches:
                print(f"\t{match}")
            yield target_signature, matches
    
    def run(self) -> list[tuple[BinarySignature, list[BinarySignature]]]:
        return list(self.step())

    def match_target(self, target: Path):
        self.context.target_path = target
        return self.run()


class BinaryHasherApplication(HashashinApplication):

    def _hash(self, binary: Path) -> BinarySignature:
        """
        Hash a binary and save it to the database.
        :param binary: binary to hash
        :return: signature of the binary
        """
        sig = self.context.extractor.extract_from_file(binary)
        if self.context.save_to_db:
            self.saveToDB(sig)
        return sig

    def step(self) -> Iterable[BinarySignature]:
        if self.context.target_path is None:
            raise ValueError("No target binary specified")
        if not self.context.target_path.exists():
            raise ValueError("Target binary does not exist")
        targets = get_binaries(self.context.target_path, progress=self.context.progress)
        logger.info(f"Hashing {len(targets)} binaries")
        for target_path in targets:
            target_signature = self.context.extractor.extract_from_file(target_path)
            if self.context.save_to_db:
                self.saveToDB(target_signature)
            yield target_signature
    
    def run(self) -> list[BinarySignature]:
        return list(self.step())


class ApplicationFactory:
    def __init__(self, context: HashashinFactoryContext):
        self.context = context

    def create(self) -> HashashinApplication:
        if self.context.task == Task.MATCH:
            return BinaryMatcherApplication(
                self.context.app_context
            )
        elif self.context.task == Task.HASH:
            return BinaryHasherApplication(
                self.context.app_context
            )
        else:
            raise NotImplementedError


def main():
    import argparse

    parser = argparse.ArgumentParser()
    db_group = parser.add_argument_group()
    db_group.add_argument(        
        "--status", "-db", action="store_true", help="Print database status"
    )
    app_group = parser.add_argument_group()
    app_group.add_argument(
        "--task", "-t", type=str, choices=[t.value for t in Task])
    app_group.add_argument("--target", "-b", type=str)
    app_group.add_argument("--save", "-s", action="store_true")
    app_group.add_argument("--threshold", "-r", type=int, default=0.5)
    app_group.add_argument("--progress", "-p", action="store_true")
    args = parser.parse_args()

    if not (args.status or (args.task and args.target)):
        parser.error("--status or --task and --target are required options")


    if args.status:
        print("Database status:")
        print(f"\t{len(SQLAlchemyBinarySignatureRepository())} signatures")
        print(f"\t{len(SQLAlchemyFunctionFeatureRepository())} features")
        return

    factory = ApplicationFactory(
        HashashinFactoryContext(
            HashashinApplicationContext(
                extractor=BinjaFeatureExtractor(),
                feature_repo=SQLAlchemyFunctionFeatureRepository(),
                binary_repo=SQLAlchemyBinarySignatureRepository(),
                target_path=Path(args.target),
                save_to_db=args.save,
                sig_match_threshold=args.threshold,
                progress=args.progress,
            ),
            task=Task(args.task),
        )
    )
    app = factory.create()
    app.run()
    print("Done!")


if __name__ == "__main__":
    main()
