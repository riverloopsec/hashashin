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
from hashashin.classes import FunctionFeatures
from hashashin.db import HashRepository
from hashashin.db import BinarySignatureRepository
from hashashin.db import SQLAlchemyBinarySignatureRepository
from hashashin.db import SQLAlchemyFunctionFeatureRepository
from hashashin.metrics import stacked_norms
from hashashin.utils import get_binaries
from hashashin.utils import list_rindex
import logging

from hashashin.db import NET_SNMP_BINARY_CACHE

logger = logging.getLogger(__name__)


class Task(Enum):
    HASH = "hash"
    MATCH = "match"


@dataclass
class HashashinApplicationContext:
    extractor: FeatureExtractor
    hash_repo: HashRepository
    target_path: Optional[Union[Path, Iterable[Path]]]
    target_func: Optional[str]
    save_to_db: Optional[bool]
    sig_match_threshold: float = 0.5
    # feat_match_threshold: int = 0
    progress: bool = False

    @staticmethod
    def _parse_args(args: list):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--extractor",
            type=str,
            help="Feature extractor to use",
            choices=FeatureExtractor.get_extractor_names(),
            default="binja",
        )
        parser.add_argument(
            "--target-path",
            type=str,
            help="Path to the binary to hash",
        )
        parser.add_argument(
            "--target-func",
            type=str,
            help="Function to hash",
        )
        parser.add_argument(
            "--save-to-db",
            action="store_true",
            help="Save to database",
        )
        parser.add_argument(
            "--sig-match-threshold",
            type=float,
            help="Signature match threshold",
            default=0.5,
        )
        parser.add_argument(
            "--progress",
            action="store_true",
            help="Show progress bar",
        )
        return parser.parse_args(args)

    @classmethod
    def from_args(cls, args: list):
        args = cls._parse_args(args)
        extractor = FeatureExtractor.from_name(args.extractor)
        hash_repo = HashRepository()
        target_path = args.target_path
        target_func = args.target_func
        save_to_db = args.save_to_db
        sig_match_threshold = args.sig_match_threshold
        # feat_match_threshold = args.feat_match_threshold
        progress = args.progress
        return cls(
            extractor,
            hash_repo,
            target_path,
            target_func,
            save_to_db,
            sig_match_threshold,
            # feat_match_threshold,
            progress,
        )


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

    def step(self) -> Iterable[Union[BinarySignature, FunctionFeatures]]:
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
        if len(targets) == 1 and self.context.target_func is not None:
            logger.info(f"Extracting {self.context.target_func} from {targets[0]}")
            bs = self.context.extractor.extract_function(targets[0], self.context.target_func)
            yield bs
            return
        logger.info(f"Loading {len(targets)} binaries...")
        pbar = tqdm(targets, disable=not self.context.progress)  # type: ignore
        for t in pbar:
            pbar.set_description(f"Hashing {t}")  # type: ignore
            cached = self.context.hash_repo.get(t)
            if cached is not None:
                pbar.set_description(f"Retrieved {t} from db")  # type: ignore
                logger.debug(f"Binary {t} already hashed, skipping")
                yield cached
                continue
            else:
                logger.debug(f"{t} not found in cache")
            try:
                target_signature = self.context.extractor.extract_from_file(
                    t,
                    progress_kwargs={
                        "desc": lambda func: f"Extracting fn {func.name} @ {func.start:#x}",
                        "position": 1,
                        "leave": False,
                    },
                )
            except BinjaFeatureExtractor.NotABinaryError as e:
                logger.warning(f"Skipping non-binary {t}:\n{e}")
                continue
            if self.context.save_to_db:
                self.context.hash_repo.save(target_signature)
            yield target_signature

    def run(self) -> List[Union[BinarySignature, FunctionFeatures]]:
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


def get_parser():
    parser = argparse.ArgumentParser()
    db_group = parser.add_argument_group("Database operations")
    db_group.add_argument(
        "--status", "-db", action="store_true", help="Print database status"
    )
    db_group.add_argument(
        "--drop", type=str, default=None, help="D" + "rop database tables. Accepts \"all\" or binary path"
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
        "--function", "-f", type=str, help="Target function name or address"
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
    demo_group.add_argument(
        "--stdlib", action="store_true", help="Standard library matching"
    )
    demo_group.add_argument(
        "--snmp", action="store_true", help="net-snmp matching"
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )
    return parser


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

        if not ((args.status or args.drop) or (args.task and args.target) or (args.fast_match or args.robust_match)):
            parser.error("--status or --task and --target are required options")

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s,%(msecs)03d %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=level,
    )

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
            _log = f"Found {len(matches)} matches above {args.threshold} similarity"
            if len(matches) == 0:
                logger.info(_log + ".\n")
                continue
            logger.info(_log + ":\n" + "\n".join([f"\t{match.path}: {target ^ match.sig}" for match in matches]) + "\n")
            # matches = "\n".join([f"\t{match.path}: {target ^ match.sig}" for match in matches]) + "\n"
            # logger.info(f"Found {len(matches)} matches above {args.threshold} similarity" + ("\n" if len(matches) == 0 else ""))
            # for match in matches:
            #     newline = "\n" if matches.index(match) == len(matches) - 1 else ""
            #     logger.info(f"\t{match.path}: {target ^ match.sig}{newline}")
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
                target_func=args.function,
                save_to_db=args.save,
                sig_match_threshold=args.threshold,
                progress=args.progress,
            ),
            task=Task(args.task),
        )
    )
    app = factory.create()
    ret = app.run()

    if args.stdlib:
        logger.info("Computing stdlib signatures...")
        # Get stdlib binary
        stdlib_path = "binary_data/libc.so.6"
        stdlib_matrix, stdlib_features = app.context.hash_repo.function_repo.get_feature_matrix(stdlib_path)
        if isinstance(ret[0], FunctionFeatures):
            # Single function
            stdlib_score = np.linalg.norm(stdlib_matrix - ret[0].asArray(), axis=1)
            # get top 5 stdlib_features
            top5 = sorted(zip(stdlib_score, stdlib_features), key=lambda x: x[0], reverse=False)[:5]
            print(f"Top 5 stdlib functions for {ret[0].function.name}:")
            spaces = max([len(str(x[1])) for x in top5])
            format = f"{{path:{{spaces}}}}: {{score}}"
            print("\t" + "\n\t".join([format.format(path=x[1].name, score=x[0], spaces=spaces) for x in top5]) + "\n")
            breakpoint()
        elif isinstance(ret[0], BinarySignature):
            for bin in ret:
                # best_matches = []
                logger.debug(f"Matching {bin.path} to stdlib...")
                hits = 0
                for func in bin.functionFeatureList:
                    if func.cyclomatic_complexity <= 2:
                        continue
                    stdlib_score = np.linalg.norm(stdlib_matrix - func.asArray(), axis=1)
                    if min(stdlib_score) == 0:
                        hits += 1
                        logger.info(f"{bin.path}:\t{func.function.name}")
                        logger.info(f"{Path(stdlib_path).name}:\t{stdlib_features[np.argmin(stdlib_score)].name}\n")
                logger.info(f"Found {hits} perfect matches for {bin.path} out of {len(bin.functionFeatureList)} functions")

                # z_scores = (np.array(best_matches) - np.mean(best_matches)) / np.std(best_matches)
                # target_z_score = -0.9
                # match_idx = np.where(z_scores < -0.9)[0]
                # match_idx = np.where(np.array(best_matches) == 0)[0]
                # logger.info(f"Found {len(match_idx)} matches for {bin.path}")
                # for i in match_idx:
                #     logger.debug(f"{bin.path}:\t{bin.functionFeatureList[i].function.name}")
                #     logger.debug(f"{Path(stdlib_path).name}:\t{stdlib_features[i].name}\n")
                #     # logger.debug(f"Dist:\t\t{best_matches[i]}\n")
                #     breakpoint()
    if args.snmp:
        if not isinstance(ret[0], BinarySignature):
            raise NotImplementedError("SNMP matching not implemented for single function yet")
        logger.info("Gathering SNMP signatures...")
        snmp_signatures = app.context.hash_repo.get_snmp_signatures()
        try:
            for target_bin in ret:
                logger.info(f"Calculating distances for {target_bin.path}")
                try:
                    jaccard_estimate = [target_bin // sig for sig in tqdm(snmp_signatures, desc="Jaccard Estimate")]
                    minhash_similarity = [target_bin ^ sig for sig in tqdm(snmp_signatures, desc="Minhash Similarity")]
                except KeyboardInterrupt:
                    breakpoint()
                logger.info("Sorting results...")
                jaccard_estimate = sorted(zip(jaccard_estimate, snmp_signatures), key=lambda x: x[0], reverse=True)
                minhash_similarity = sorted(zip(minhash_similarity, snmp_signatures), key=lambda x: x[0], reverse=True)
                jaccard_closest_bins = jaccard_estimate[:list_rindex([x[0] for x in jaccard_estimate], max(jaccard_estimate, key=lambda x: x[0])[0]) + 1]
                jaccard_closest_paths = [x[1].path for x in jaccard_closest_bins]
                minhash_closest_bins = minhash_similarity[:list_rindex([x[0] for x in minhash_similarity], max(minhash_similarity, key=lambda x: x[0])[0]) + 1]
                minhash_closest_paths = [x[1].path for x in minhash_closest_bins]
                if any(target_bin.path.name in str(path) for path in jaccard_closest_paths):
                    likely_matches = [x for x in jaccard_closest_bins if target_bin.path.name in str(x[1].path)]
                    print(f"Jaccard likely matches for {target_bin.path}:")
                    spaces = max([len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE))) for x in likely_matches])
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print("\t" + "\n\t".join([format.format(path=str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)), score=x[0], spaces=spaces) for x in likely_matches]) + "\n")
                else:
                    # print top 5 closest matches
                    print(f"Top 5 Jaccard Estimate matches for {target_bin.path}:")
                    spaces = max([len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE))) for x in jaccard_estimate[:5]])
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print("\t" + "\n\t".join([format.format(path=str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)), score=x[0], spaces=spaces) for x in jaccard_estimate[:5]]) + "\n")
            
                if any(target_bin.path.name in str(path) for path in minhash_closest_paths):
                    likely_matches = [x for x in minhash_closest_bins if target_bin.path.name in str(x[1].path)]
                    print(f"Minhash likely matches for {target_bin.path}:")
                    spaces = max([len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE))) for x in likely_matches])
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print("\t" + "\n\t".join([format.format(path=str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)), score=x[0], spaces=spaces) for x in likely_matches]) + "\n")
                else:
                    # print top 5 closest matches
                    print(f"Top 5 Minhash Similarity matches for {target_bin.path}:")
                    spaces = max([len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE))) for x in minhash_similarity[:5]])
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print("\t" + "\n\t".join([format.format(path=str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)), score=x[0], spaces=spaces) for x in minhash_similarity[:5]]) + "\n")

        except Exception as e:
            print(e)
            breakpoint()


            # print(f"Top {args.matches} Jaccard Estimate matches for {target_bin.path}:")
            # spaces = max([len(str(x[1])) for x in jaccard_estimate[:args.matches]])
            # format = f"{{path:{{spaces}}}}: {{score}}"
            # print("\t" + "\n\t".join([format.format(path=x[1].path, score=x[0], spaces=spaces) for x in jaccard_estimate[:args.matches]]) + "\n")
            # print(f"Top {args.matches} Minhash Similarity matches for {target_bin.path}:")
            # spaces = max([len(str(x[1])) for x in minhash_similarity[:args.matches]])
            # format = f"{{path:{{spaces}}}}: {{score}}"
            # print("\t" + "\n\t".join([format.format(path=x[1].path, score=x[0], spaces=spaces) for x in minhash_similarity[:args.matches]]) + "\n")

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborting")
    except Exception as e:
        print(f"Error: {e}")
        breakpoint()
