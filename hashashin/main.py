import argparse
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import Union
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from hashashin.classes import BinarySignature
from hashashin.classes import BinjaFeatureExtractor
from hashashin.classes import FeatureExtractor
from hashashin.db import HashRepository
from hashashin.db import SQLAlchemyBinarySignatureRepository
from hashashin.db import SQLAlchemyFunctionFeatureRepository
from hashashin.metrics import stacked_norms
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
    def _match(self, sig: BinarySignature) -> List[BinarySignature]:
        """
        Match a binary signature against the database.
        :param sig: signature to match
        :return: a list of matching signatures sorted by distance
        """
        return self.context.hash_repo.binary_repo.fast_match(
            sig, self.context.sig_match_threshold
        )

    def step(self) -> Iterable[Tuple[BinarySignature, List[BinarySignature]]]:
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

    def run(self) -> List[Tuple[BinarySignature, List[BinarySignature]]]:
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

    def run(self) -> List[BinarySignature]:
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
            "--fast-match", type=str, nargs="+", metavar="BINARY_PATH", help="Fast match a binary against the database"
        )
        demo_group.add_argument(
            "--robust-match", type=str, nargs="+", metavar="BINARY_PATH",
            help="Match a given binary against database"
        )
        demo_group.add_argument(
            "--matches", type=int, metavar="N", help="Number of matches to return"
        )
        args = parser.parse_args()
        if not ((args.status or args.drop) or (args.task and args.target) or (args.fast_match or args.robust_match)):
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
        for target in args.fast_match:
            target = Path(__file__).parent / target
            if not target.is_file() or len(get_binaries(target, silent=True)) == 0:
                logger.debug(f"{target} is not a valid binary")
                continue
            hash_repo = HashRepository()
            target = hash_repo.get(target)
            if target is None:
                logger.error(f"Failed to hash {target}")
                continue
            logger.info(f"Fast matching target {target.path.absolute().relative_to(Path(__file__).parent / 'binary_data')}")
            matches = hash_repo.binary_repo.fast_match(target, args.threshold)
            matches = list(filter(lambda x: x.path != str(target.path), matches))
            logger.info(f"Found {len(matches)} matches above {args.threshold} similarity")
            for match in matches:
                print(f"\t{match.path}: {target ^ match.sig}")
            print()
        return

    if args.robust_match:
        hash_repo = HashRepository()
        targets = [hash_repo.get(path) for path in args.robust_match if len(get_binaries(Path(path), silent=True)) > 0]
        if any(target is None for target in targets):
            logger.error(f"Failed to hash {args.robust_match}")
            return
        # target = hash_repo.get(args.robust_match)
        # logger.info("Stacking target functions into matrix")
        target_np_functions = [
                np.stack([f.asArray() for f in target.functionFeatureList])
                for target in tqdm(targets, desc="Collecting target functions")
            ]
        binary_scores: list[dict[str, float]] = [{} for _ in range(len(target_np_functions))]
        logger.info("Collecting binaries from database...")
        for binary in tqdm(hash_repo.binary_repo.getAll(), desc="Scoring binaries"):
            features_matrix, features = hash_repo.function_repo.get_feature_matrix(binary.path)
            scores = stacked_norms(target_np_functions, features_matrix)
            for i, score in enumerate(scores):
                binary_scores[i][binary.path] = score

        for i, target in enumerate(targets):
            print(f"Top 5 scores for {target.path.absolute().relative_to(Path(__file__).parent / 'binary_data')}:")
            top5 = sorted(binary_scores[i].items(), key=lambda x: x[1], reverse=True)[:5]
            spaces = max([len(str(x[0])) for x in top5])
            format = f"{{path:{{spaces}}}}: {{score}}"
            print("\t" + "\n\t".join([format.format(path=str(x[0]), score=x[1], spaces=spaces) for x in top5]) + "\n")

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
