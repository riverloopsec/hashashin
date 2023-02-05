import glob
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Iterable, Optional, Union, Generator
import argparse

import magic
from tqdm import tqdm

from hashashin.classes import BinarySignature
from hashashin.db import (
    BinarySignatureRepository,
    FunctionFeatureRepository,
    SQLAlchemyBinarySignatureRepository,
    SQLAlchemyFunctionFeatureRepository,
    HashRepository,
)
from hashashin.classes import BinjaFeatureExtractor, FeatureExtractor
from hashashin.utils import get_binaries
from hashashin.utils import logger

logger = logger.getChild(Path(__file__).name)


class Task(Enum):
    HASH = "hash"
    MATCH = "match"


@dataclass
class HashashinApplicationContext:
    extractor: FeatureExtractor
    hash_repo: HashRepository
    target_path: Optional[Union[Path, Iterable[Path]]]
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

    def target(self, target: Union[Path, Iterable[Path]]):
        self.context.target_path = target
        return self.run()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.context})"


class BinaryMatcherApplication(HashashinApplication):
    def _match(self, sig: BinarySignature) -> list[BinarySignature]:
        """
        Match a binary signature against the database.
        :param sig: signature to match
        :return: a list of matching signatures sorted by distance
        """
        return self.context.hash_repo.binary_repo.match_signature(
            sig, self.context.sig_match_threshold
        )

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
        show_progress = self.context.app_context.progress
        for target_path in tqdm(targets, disable=not show_progress):
            target_signature = self.context.extractor.extract_from_file(target_path)
            if self.context.save_to_db:
                self.context.hash_repo.save(target_signature)
            matches = self._match(target_signature)
            if len(matches) == 0:
                print(f"No matches found for {target_path}")
                continue
            print(f"Matches for {target_path}:")
            for match in matches:
                print(f"\t{match}")
            yield target_signature, matches

    def run(self) -> list[tuple[BinarySignature, list[BinarySignature]]]:
        logger.info("Matching binaries")
        return list(self.step())


class BinaryHasherApplication(HashashinApplication):
    def _hash(self, binary: Path) -> BinarySignature:
        """
        Hash a binary and save it to the database.
        :param binary: binary to hash
        :return: signature of the binary
        """
        sig = self.context.extractor.extract_from_file(binary)
        if self.context.save_to_db:
            self.context.hash_repo.save(sig)
        return sig

    def step(self) -> Iterable[BinarySignature]:
        if self.context.target_path is None:
            raise ValueError("No target binary specified")
        target_path: Union[Path, list[Path]] = (
            list(self.context.target_path)
            if isinstance(self.context.target_path, Generator)
            else self.context.target_path
        )
        if isinstance(target_path, list) and not all(p.exists() for p in target_path):
            raise ValueError(
                f"List of target binaries contains non-existent paths: {target_path}"
            )
        if isinstance(target_path, Path) and not target_path.exists():
            raise ValueError(
                f"Target binary does not exist: {self.context.target_path}"
            )
        targets = get_binaries(
            target_path, progress=self.context.progress, recursive=True
        )
        logger.info(f"Hashing {len(targets)} binaries")
        pbar = tqdm(targets, disable=not self.context.progress)
        for t in pbar:
            pbar.set_description(f"Hashing {t}")
            cached = self.context.hash_repo.get(t)
            if cached is not None:
                pbar.set_description(f"Retrieved {t} from db")
                logger.debug(f"Binary {t} already hashed, skipping")
                yield cached
                continue
            target_signature = self.context.extractor.extract_from_file(t)
            if self.context.save_to_db:
                self.context.hash_repo.save(target_signature)
            yield target_signature

    def run(self) -> list[BinarySignature]:
        logger.info(f"Hashing {self.context.target_path}")
        return list(self.step())


class ApplicationFactory:
    def __init__(self, context: HashashinFactoryContext):
        self.context = context

    def create(self) -> HashashinApplication:
        if self.context.task == Task.MATCH:
            return BinaryMatcherApplication(self.context.app_context)
        elif self.context.task == Task.HASH:
            return BinaryHasherApplication(self.context.app_context)
        else:
            raise NotImplementedError

    @classmethod
    def fromAppContext(
        cls, context: HashashinApplicationContext, task: Task
    ) -> HashashinApplication:
        return ApplicationFactory(HashashinFactoryContext(context, task)).create()

    @classmethod
    def getHasher(cls, context: HashashinApplicationContext) -> BinaryHasherApplication:
        hasher = ApplicationFactory.fromAppContext(context, Task.HASH)
        if not isinstance(hasher, BinaryHasherApplication):
            raise ValueError("Invalid hasher")
        return hasher

    @classmethod
    def getMatcher(
        cls, context: HashashinApplicationContext
    ) -> BinaryMatcherApplication:
        matcher = ApplicationFactory.fromAppContext(context, Task.MATCH)
        if not isinstance(matcher, BinaryMatcherApplication):
            raise ValueError("Invalid matcher")
        return matcher


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser()
        db_group = parser.add_argument_group("Database operations")
        db_group.add_argument(
            "--status", "-db", action="store_true", help="Print database status"
        )
        db_group.add_argument(
            "--drop", action="store_true", help="Drop database tables"
        )
        app_group = parser.add_argument_group("Application operations")
        app_group.add_argument(
            "--task",
            "-t",
            type=str,
            choices=[
                t.value for t in Task if isinstance(t.value, str)
            ],  # mypy type fix
            help="Application task",
        )
        app_group.add_argument(
            "--target", "-b", type=str, help="Target binary or directory"
        )
        app_group.add_argument(
            "--save", "-s", action="store_true", help="Save results to database"
        )
        app_group.add_argument(
            "--threshold", "-r", type=int, default=0.5, help="Signature match threshold"
        )
        app_group.add_argument(
            "--progress", "-p", action="store_true", help="Show progress bar"
        )
        args = parser.parse_args()
        if not ((args.status or args.drop) or (args.task and args.target)):
            parser.error("--status or --task and --target are required options")

    if args.status:
        print("Database status:")
        print(f"\t{len(SQLAlchemyBinarySignatureRepository())} signatures")
        print(f"\t{len(SQLAlchemyFunctionFeatureRepository())} features")
        return

    if args.drop:
        print("Dropping database tables...")
        SQLAlchemyBinarySignatureRepository().drop()
        SQLAlchemyFunctionFeatureRepository().drop()
        print("Done!")
        args.drop, args.status = False, True
        main(args)
        return

    factory = ApplicationFactory(
        HashashinFactoryContext(
            HashashinApplicationContext(
                extractor=BinjaFeatureExtractor(),
                hash_repo=HashRepository(),
                target_path=Path(args.target),
                save_to_db=args.save,
                sig_match_threshold=args.threshold,
                progress=args.progress,
            ),
            task=Task(args.task),
        )
    )
    app = factory.create()
    print(app.run())
    print("Done!")


if __name__ == "__main__":
    main()
