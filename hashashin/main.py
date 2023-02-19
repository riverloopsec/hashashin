import glob
import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection, Iterable, Optional, Union, Generator
import argparse
from collections import Counter

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
from hashashin.metrics import matrix_norms
from hashashin.utils import get_binaries
from hashashin.utils import logger
import numpy as np

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
        return self.context.hash_repo.binary_repo.fast_match(
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
        show_progress = self.context.progress
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
            try:
                target_signature = self.context.extractor.extract_from_file(
                    t,
                    progress_kwargs={
                        "desc": f"Extracting {t}",
                        "position": 1,
                        "leave": False,
                    },
                )
            except BinjaFeatureExtractor.NotABinaryError:
                logger.info(f"Skipping non-binary {t}")
                continue
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
            "--drop", type=str, default=None, help="D"+"rop database tables. Accepts \"all\" or binary path"
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
            "--threshold", "-r", type=float, default=0.5, help="Signature match threshold"
        )
        app_group.add_argument(
            "--progress", "-p", action="store_true", help="Show progress bar"
        )
        demo_group = parser.add_argument_group("Demo operations")
        demo_group.add_argument(
            "--fast-match", type=str, metavar="BINARY_PATH", help="Fast match a binary against the database"
        )
        demo_group.add_argument(
            "--function-match", type=str, metavar="BINARY_PATH",
            help="Match a given function against database"
        )
        demo_group.add_argument(
            "--matches", type=int, metavar="N", help="Number of matches to return"
        )
        args = parser.parse_args()
        if not ((args.status or args.drop) or (args.task and args.target) or (args.fast_match or args.function_match)):
            parser.error("--status or --task and --target are required options")

    if args.status:
        print("Database status:")
        print(f"\t{len(SQLAlchemyBinarySignatureRepository())} binaries")
        print(f"\t{len(SQLAlchemyFunctionFeatureRepository())} functions")
        return

    if args.drop is not None:
        # Confirm drop
        if not input(f"Confirm drop {args.drop}? [y/N] ").lower().startswith("y"):
            print("Aborting")
            return
        print("Dropping database tables...")
        SQLAlchemyBinarySignatureRepository().drop(args.drop)
        print("Done!")
        args.drop, args.status = False, True
        main(args)
        return

    if args.fast_match:
        args.fast_match = Path(__file__).parent / args.fast_match
        if not args.fast_match.is_file() or len(get_binaries(args.fast_match, silent=True)) == 0:
            logger.error(f"{args.fast_match} is not a valid binary")
            print()
            return
        hash_repo = HashRepository()
        target = hash_repo.get(args.fast_match)
        if target is None:
            logger.error(f"Failed to hash {args.fast_match}")
            return
        logger.info(f"Fast matching target {target.path.name}: {target.signature.hex()}")
        matches = hash_repo.binary_repo.fast_match(target)
        matches = list(filter(lambda x: x.path != str(target.path), matches))
        logger.info(f"Found {len(matches)} matches")
        for match in matches:
            print(f"\t{match.path} {match.sig.hex()}")
        print()
        return

    if args.function_match:
        hash_repo = HashRepository()
        target = hash_repo.get(args.function_match)
        # logger.info("Choosing top 10 functions by dominator signature")
        # candidate_functions = sorted(target.functionFeatureList, key=lambda f: f.dominator_signature)
        logger.info("Stacking candidate functions into matrix")
        candidate_np_functions = np.stack([f.asArray() for f in target.functionFeatureList])
        binary_scores = {}
        for binary in tqdm(hash_repo.binary_repo.getAll(), desc="Scoring binaries"):
            features_matrix, features = hash_repo.function_repo.get_feature_matrix(binary.path)
            binary_scores[binary.path] = matrix_norms(candidate_np_functions, features_matrix)
        logger.info(f"Top 5 matches:\n{sorted(binary_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")
        # target_fn = list(filter(lambda x: args.function_match[1] in x.function.name, target.functionFeatureList))
        # breakpoint()
        # if len(target_fn) > 1:
        #     raise NotImplementedError("Multiple matches not implemented")
        # if len(target_fn) == 0:
        #     raise ValueError("Function name not found")
        # target_fn = target_fn[0]
        # logger.info(f"Target {target.path}: {target_fn.function.name} {target_fn.signature.hex()}")
        # matches = hash_repo.function_repo.match(target_fn, threshold=args.threshold)
        # logger.info(f"Found {len(matches)} matches above threshold")
        # logger.info(f"Closest match ({matches[0] ^ target_fn}): {matches[0]} {matches[0].function.path} {matches[0].signature.hex()}")
        # binary_matches = Counter([x.function.path for x in matches])
        # logger.info(f"Found {len(binary_matches)} binaries with matching functions:\n{sorted(binary_matches.items())}")
        # breakpoint()
        # likely_matches = {}
        # for match in tqdm(matches, desc="Calculating likely matches"):
        #     likely_matches[match.function.path] = likely_matches.get(match.function.path, 0) + (match ^ target_fn)
        # for b, score in tqdm(likely_matches.items(), desc="Normalizing matches"):
        #     if not Path(b).is_file():
        #         if Path(b.replace("hashashin/", "")).is_file():
        #             b = b.replace("hashashin/", "")
        #         else:
        #             logger.warning(f"Binary {b} not found, skipping")
        #             continue
        #     likely_matches[b] = score / len(hash_repo.binary_repo.get(b).functionFeatureList)
        #
        # percentage_matches = {}
        # for b, count in tqdm(binary_matches.items(), desc="Calculating percentage matches", disable=not args.progress):
        #     if not Path(b).is_file():
        #         if Path(b.replace("hashashin/", "")).is_file():
        #             b = b.replace("hashashin/", "")
        #         else:
        #             logger.warning(f"Binary {b} not found, skipping")
        #             continue
        #     funcs = len(hash_repo.binary_repo.get(b).functionFeatureList)
        #     percentage_matches[b] = count / funcs
        # breakpoint()
        # binary_likelihood = sorted(percentage_matches.items(), key=lambda x: x[1], reverse=True)
        # logger.info(f"Top 3 most likely binaries (path, likelihood):\n{binary_likelihood[:3]}")
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
